import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from analysis.report import write_json
from evaluation.contracts import validate_partition
from inference.unet import prediction_from_probabilities, _synchronize
from models.unet import build_model
from preprocessing.transforms import transform_aligned_arrays
from training.unet import _check_config, create_optimizer, create_scheduler, evaluate_loss, load_checkpoint, make_batch, save_checkpoint, set_seed, train_step


def augment_sample(sample, rotation, flip):
    """입력·정답·유효성 마스크에 같은 D4 회전·반전을 적용해 새 표본을 반환한다.

    공통 변환에 맞춰 CHW를 HWC로 잠시 바꿨다가 되돌린다. GT 도형을 다시
    래스터화하지 않으며, 존재 정답과 증강 전 ROI 원점 등 메타데이터는 유지한다.
    """
    arrays = {"input": sample["input"].transpose(1, 2, 0),
              "target": sample["target"].transpose(1, 2, 0),
              "segmentation_valid": sample["segmentation_valid"].transpose(1, 2, 0)}
    arrays.update({key: sample[key] for key in
                   ("prior", "gt", "official_valid", "numeric_valid", "inside")})
    if "index_valid" in sample:
        arrays["index_valid"] = sample["index_valid"]
    if "boundary_distance" in sample:
        arrays["boundary_distance"] = sample["boundary_distance"].transpose(1, 2, 0)
    if "coverage_target" in sample:
        arrays["coverage_target"] = sample["coverage_target"].transpose(1, 2, 0)
    transformed = transform_aligned_arrays(arrays, rotation, flip)
    result = dict(sample)
    for key, value in transformed.items():
        result[key] = np.ascontiguousarray(value.transpose(2, 0, 1) if key in
                                          ("input", "target", "segmentation_valid", "boundary_distance", "coverage_target") else value)
    if "index_valid_by_name" in sample:
        result["index_valid_by_name"] = transform_aligned_arrays(
            sample["index_valid_by_name"], rotation, flip)
    return result


def predict_samples(model, samples, device, batch_size):
    """증강 없는 표본을 추론해 (예측 행, 분할 확률 N×S×S, 존재 확률 N)을 반환한다.

    sigmoid로 확률을 만든 뒤 추론 모듈과 같은 polygon 후처리를 적용한다.
    미라벨을 포함해 전달된 모든 표본의 키와 예측 순서를 보존한다.
    """
    model.eval()
    rows, probabilities, presence = [], [], []
    with torch.inference_mode():
        for offset in range(0, len(samples), batch_size):
            chunk = samples[offset:offset + batch_size]
            tensor = torch.from_numpy(np.stack([sample["input"] for sample in chunk])).to(device)
            output = model(tensor)
            masks = torch.sigmoid(output["segmentation"]).cpu().numpy()[:, 0]
            heads = torch.sigmoid(output["presence"]).cpu().numpy()
            for sample, mask, head in zip(chunk, masks, heads, strict=True):
                rows.append({"key": sample["key"],
                             **prediction_from_probabilities(sample, mask, head)})
            probabilities.extend(masks)
            presence.extend(heads)
    return rows, np.asarray(probabilities, dtype=np.float32), np.asarray(presence, dtype=np.float32)


def _checkpoint(path, model, optimizer, update, config, metadata, scheduler=None):
    """임시 저장이 끝난 뒤 현재 run에서 갱신하는 best/last 파일만 교체한다."""
    temporary = path.with_suffix(".pending.pt")
    save_checkpoint(temporary, model, optimizer, update, config=config, metadata=metadata,
                    scheduler=scheduler)
    temporary.replace(path)


def fit_holdout(train, validation, config, artifacts, *, progress=print):
    """주어진 분할을 학습하고 주기적으로 검증한 뒤 best 복구 결과·이력을 반환한다.

    update는 배치 한 번의 가중치 갱신이다. epoch 수 대신 설정된 update 수까지
    학습하며, 검증 loss가 가장 낮은 best와 최근 검증 시점의 last를 따로 저장한다.
    작은 update 수와 CPU 설정을 전달하면 같은 실행 경로를 합성 검사에 사용할 수 있다.
    """
    from evaluation.scorer import score_predictions

    _check_config(config)
    validate_partition(train, validation)
    d4_selection = config.get('checkpoint_selection') == 'bce_and_d4_reference'
    if d4_selection:
        from training.d4_selection import DENOMINATORS, evaluate_selection, selection_population
        selection_counts = selection_population(validation)
    train = [sample for sample in train if sample["presence_valid"]]
    if not train:
        raise ValueError("No supervised training samples")
    device = torch.device(config["device"])
    if device.type == "mps" and (not torch.backends.mps.is_available()
                                  or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"):
        raise RuntimeError("MPS with CPU fallback disabled is required")
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    set_seed()
    torch.set_num_threads(2)
    model = build_model(config['model'], config['roi_size'], len(config['input_bands'])).to(device)
    optimizer = create_optimizer(model.parameters(), config)
    scheduler = create_scheduler(optimizer, config)
    # 매 실행 모델 초기화를 seed42로 맞추고, 표본 순서와 D4 추첨은 독립 RNG로 만든다.
    # 같은 표본 순서의 ROI 후보들은 같은 초기 가중치·표본 순서·증강 추첨을 사용한다.
    order_rng = np.random.default_rng(42)
    d4_rng = np.random.default_rng(42)
    order, position = [], 0
    sampling_trace = hashlib.sha256()
    history, validations = [], []
    best_loss, best_update = float("inf"), None
    best_d4_value, best_d4_update = float('-inf'), None
    seen = 0
    training_seconds = validation_seconds = saving_seconds = 0.0
    observed_driver = observed_tensor = 0
    started = time.perf_counter()
    with (artifacts / "updates.jsonl").open("x", encoding="utf-8") as update_log:
        for update in range(1, config["updates"] + 1):
            _synchronize(device)
            before = time.perf_counter()
            if position >= len(order):
                order, position = order_rng.permutation(len(train)), 0
            # 한 순열을 차례로 소진한다. 끝에 남은 작은 배치를 버리거나 다음 순열로 채우지 않는다.
            indices = order[position:position + config["batch_size"]]
            position += len(indices)
            batch_samples = []
            for batch_index, index in enumerate(indices):
                # 0~7을 네 회전 × 좌우 반전 여부로 나누어 D4의 여덟 변환 중 하나를 고른다.
                transform = int(d4_rng.integers(8))
                sampling_trace.update(json.dumps([list(train[index]["key"]), int(transform)], separators=(",", ":")).encode())
                sample = augment_sample(train[index], transform % 4, transform >= 4)
                batch_samples.append(sample)
            batch = make_batch(batch_samples, device)
            metrics = train_step(model, optimizer, batch, config)
            if scheduler is not None:
                scheduler.step()
            if config.get("diagnostics", False):
                metrics["next_learning_rate"] = optimizer.param_groups[0]["lr"]
            _synchronize(device)
            seconds = time.perf_counter() - before
            training_seconds += seconds
            seen += len(indices)
            # 이 loss는 이번 배치의 값이다. equivalent_epochs는 누적 처리 표본/학습 표본 수다.
            row = {"update": update, "batch_samples": len(indices), "samples_seen": seen,
                   "equivalent_epochs": seen / len(train), "seconds": seconds, **metrics}
            history.append(row)
            update_log.write(json.dumps(row, allow_nan=False) + "\n")
            if device.type == "mps":
                observed_driver = max(observed_driver, torch.mps.driver_allocated_memory())
                observed_tensor = max(observed_tensor, torch.mps.current_allocated_memory())
            if update % config["validation_every"] and update != config["updates"]:
                continue
            update_log.flush()
            before = time.perf_counter()
            # 검증은 지정 간격과 마지막 update에 수행한다. 전체 유효 화소·표본으로
            # 가중 평균한 loss와 최종 polygon 기반 지표를 함께 기록한다.
            # D49의 학습 손실과 무관하게 best 선택은 기존 비가중 검증 BCE 합이다.
            loss = evaluate_loss(model, validation, device, config["batch_size"])
            predictions, masks, heads = predict_samples(model, validation, device, config["batch_size"])
            scores = score_predictions(validation, predictions)["summary"]
            if d4_selection:
                d4_scores, d4_predictions = evaluate_selection(model, validation, device)
                if {k: d4_scores[k] for k in DENOMINATORS} != selection_counts:
                    raise ValueError('Inner D4 selection population changed')
            _synchronize(device)
            validation_seconds += time.perf_counter() - before
            validation_row = {"update": update, "equivalent_epochs": seen / len(train),
                              **loss, **scores, "presence_macro_f1": scores["presence"]}
            if d4_selection:
                validation_row['selection_d4'] = d4_scores
            validations.append(validation_row)
            before = time.perf_counter()
            metadata = {"split_id": config["split_id"], "samples_seen": seen,
                        "equivalent_epochs": seen / len(train), "validation_loss": loss["loss"]}
            _checkpoint(artifacts / "last.pt", model, optimizer, update, config, metadata, scheduler)
            # best 선택 기준은 F1이 아닌 검증 loss다. 동률이면 먼저 저장한 best를 유지한다.
            if loss["loss"] < best_loss:
                best_loss, best_update = loss["loss"], update
                _checkpoint(artifacts / "best.pt", model, optimizer, update, config, metadata, scheduler)
                np.savez_compressed(artifacts / "best_probabilities.npz", masks=masks, presence=heads)
                write_json(artifacts / "best_predictions.json", predictions)
            if d4_selection and d4_scores['reference_total'] > best_d4_value:
                best_d4_value, best_d4_update = d4_scores['reference_total'], update
                _checkpoint(artifacts / 'best_d4.pt', model, optimizer, update, config,
                            {**metadata, 'selection_d4': d4_scores}, scheduler)
                write_json(artifacts / 'best_d4_predictions.json', d4_predictions)
            write_json(artifacts / "validation_history.json", validations)
            saving_seconds += time.perf_counter() - before
            progress(json.dumps({"update": update, "of": config["updates"],
                                 "train_loss": metrics["loss"], "validation_loss": loss["loss"],
                                 "best_update": best_update,
                                 "elapsed_seconds": round(time.perf_counter() - started, 2)},
                                ensure_ascii=False, allow_nan=False), flush=True)
    del model, optimizer, scheduler
    if device.type == "mps":
        torch.mps.empty_cache()
    # 학습 재개가 아니라 저장·복구의 재현 확인이다. best를 다시 추론해 확률·손실은
    # 정해진 허용오차로, 최종 polygon은 정확히 일치하는지 확인한다.
    model, saved = load_checkpoint(artifacts / "best.pt", device=device)
    predictions, masks, heads = predict_samples(model, validation, device, config["batch_size"])
    with np.load(artifacts / "best_probabilities.npz", allow_pickle=False) as expected:
        np.testing.assert_allclose(masks, expected["masks"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(heads, expected["presence"], rtol=1e-4, atol=1e-5)
    before_predictions = json.loads((artifacts / "best_predictions.json").read_text())
    if [r["polygons"] for r in predictions] != [r["polygons"] for r in before_predictions]:
        raise ValueError("Reloaded checkpoint polygon predictions differ")
    loss = evaluate_loss(model, validation, device, config["batch_size"])
    if not np.isclose(loss["loss"], best_loss, rtol=1e-4, atol=1e-5):
        raise ValueError("Reloaded checkpoint validation loss differs")
    score = score_predictions(validation, predictions)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    summary = {"sampling_d4_trace_sha256": sampling_trace.hexdigest(), "updates": config["updates"], "validation_count": len(validations),
               "best_update": best_update, "best_validation_loss": loss["loss"],
               "last_validation_loss": validations[-1]["loss"], "samples_seen": seen,
               "equivalent_epochs": seen / len(train), "device": str(device), "dtype": "float32",
               "training_seconds": training_seconds, "validation_seconds": validation_seconds,
               "checkpoint_saving_seconds": saving_seconds, "fit_and_reload_seconds": time.perf_counter() - started,
               "observed_mps_driver_bytes": int(observed_driver), "observed_mps_tensor_bytes": int(observed_tensor),
               "mps_memory_kind": "post-update observation, not allocation peak",
               "process_peak_rss_bytes": int(rss), "checkpoint_reload_verified": True,
               "checkpoint_update": saved["updates"], "best_scores": score["summary"]}
    if d4_selection:
        summary.update(best_d4_update=best_d4_update, best_d4_reference_total=best_d4_value,
                       d4_validation_count=len(validations), selection_population=selection_counts)
    return {"summary": summary, "train_history": history, "validation_history": validations,
            "predictions": predictions, "score_rows": score["rows"]}
