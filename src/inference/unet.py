"""정답 라벨 없이 공통 U-Net 입력·모델·후처리를 연결해 예측 CSV를 만든다.

checkpoint의 ROI 설정을 복구해 학습 때와 같은 전처리를 사용한다.
확률을 polygon으로 바꾸는 함수는 Holdout 검증에서도 공유한다.
"""

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import torch

from postprocessing.polygons import masks_to_polygons
from preprocessing.model_input import prepare_sample, input_parameters, normalize_sample
from models.checkpoint import inference_config, load_checkpoint


CSV_FIELDS = ("location_id", "date", "site_id", "polygons")


def postprocessing_options(config=None):
    """제출 전용 임계값·선택적 등위선을 읽고 checkpoint 설정은 유지한다."""
    defaults = {"presence_threshold": 0.5, "mask_threshold": 0.5}
    options = defaults if config is None else config.get("postprocessing", defaults)
    if (not isinstance(options, dict) or not defaults.keys() <= options.keys()
            or options.keys() - defaults.keys() - {"method"}
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not np.isfinite(value) or not 0 <= value <= 1
                   for key, value in options.items() if key in defaults)):
        raise ValueError("Postprocessing must specify finite presence/mask thresholds within [0, 1]")
    if options.get("method", "pixel_union") not in ("pixel_union", "probability_contour"):
        raise ValueError("Postprocessing method must be pixel_union or probability_contour")
    return {**{key: float(options[key]) for key in defaults},
            **({"method": options["method"]} if "method" in options else {})}


def _inputs(input_dir):
    """location/sites.json과 location/bands/*.npy만 확인해 장면 목록을 만든다.

    정답 파일은 읽지 않는다. 날짜·관심지 키를 확인하고 정렬된 순서로 반환해
    이후 각 location/date/site 조합에 예측 행 하나를 만들 수 있게 한다.
    """
    root = Path(input_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Inference input must be a directory, not a symlink")
    scenes = []
    for location in sorted(root.iterdir()):
        if location.is_symlink():
            raise ValueError("Symlinks are not supported in inference input")
        if not location.is_dir():
            continue
        site_path, bands = location / "sites.json", location / "bands"
        if (not site_path.is_file() or site_path.is_symlink()
                or not bands.is_dir() or bands.is_symlink()):
            raise ValueError("Each input location needs sites.json and bands/")
        sites = json.loads(site_path.read_text(encoding="utf-8"))
        if not isinstance(sites, list) or not sites:
            raise ValueError("sites.json must contain a nonempty list")
        ids = set()
        for site in sites:
            if not isinstance(site, dict) or "prior_polygon" not in site:
                raise ValueError("Each site needs site_id and prior_polygon")
            site_id = site.get("site_id")
            if (isinstance(site_id, bool) or not isinstance(site_id, (str, int))
                    or str(site_id) == ""):
                raise ValueError("site_id must be a nonempty string or integer")
            if str(site_id) in ids:
                raise ValueError("Duplicate site identifiers within one location")
            ids.add(str(site_id))
        paths = sorted(bands.glob("*.npy"))
        if not paths:
            raise ValueError("Every location must contain at least one band array")
        for path in paths:
            if not path.is_file() or path.is_symlink():
                raise ValueError("Band arrays must be ordinary files")
            date = path.stem
            if len(date) != 8 or datetime.strptime(date, "%Y%m%d").strftime("%Y%m%d") != date:
                raise ValueError("Band filenames must use YYYYMMDD.npy")
            scenes.append((location.name, date, sites, path))
    if not scenes:
        raise ValueError("No inference scenes found")
    return scenes


def _synchronize(device):
    """비동기 GPU 작업이 끝날 때까지 기다려 구간별 실행 시간을 측정한다."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_probabilities(model, tensor, *, tta="none"):
    """공통 FP32 추론으로 (N,S,S) 분할 확률·(N,) 존재 확률과 시간을 반환한다.

    D4는 회전 후 좌우 반전하며, 확률을 반전 해제 후 역회전해 평균한다.
    sigmoid는 기존 none 경로처럼 CPU에서 계산하고 모든 방향을 순차 처리한다.
    """
    if tta not in ("none", "d4"):
        raise ValueError("tta must be none or d4")
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 4
            or tensor.shape[1] != getattr(model, "input_channels", 5) or tensor.shape[0] < 1
            or tensor.shape[2] < 1 or tensor.shape[2] != tensor.shape[3]
            or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()):
        raise ValueError("Expected a finite FP32 N x model_channels x S x S input")
    model.eval()
    device = tensor.device
    count, _, size, _ = tensor.shape
    timings = {"forward": 0.0, "augmentation_aggregation": 0.0}
    transforms = ((0, False),) if tta == "none" else tuple(
        (rotation, flip) for flip in (False, True) for rotation in range(4))
    mask_sum = presence_sum = None
    with torch.inference_mode():
        for rotation, flip in transforms:
            augmented = tensor
            if tta == "d4":
                before = perf_counter()
                augmented = torch.rot90(tensor, rotation, dims=(-2, -1))
                if flip:
                    augmented = torch.flip(augmented, dims=(-1,))
                _synchronize(device)
                timings["augmentation_aggregation"] += perf_counter() - before
            _synchronize(device)
            before = perf_counter()
            output = model(augmented)
            _synchronize(device)
            if (not isinstance(output, dict)
                    or not isinstance(output.get("segmentation"), torch.Tensor)
                    or not isinstance(output.get("presence"), torch.Tensor)):
                raise ValueError("Model must return segmentation and presence tensors")
            segmentation = output["segmentation"].detach().cpu()
            presence = output["presence"].detach().cpu()
            if (segmentation.shape != (count, 1, size, size)
                    or presence.shape != (count,)
                    or segmentation.dtype != torch.float32 or presence.dtype != torch.float32):
                raise ValueError("Unexpected FP32 model output shapes or dtypes")
            if not torch.isfinite(segmentation).all() or not torch.isfinite(presence).all():
                raise ValueError("Model produced nonfinite predictions")
            presence_probability = torch.sigmoid(presence).numpy()
            mask_probability = torch.sigmoid(segmentation)[:, 0]
            timings["forward"] += perf_counter() - before
            if tta == "none":
                return mask_probability.numpy(), presence_probability, timings
            before = perf_counter()
            if flip:
                mask_probability = torch.flip(mask_probability, dims=(-1,))
            mask_probability = torch.rot90(mask_probability, -rotation, dims=(-2, -1)).numpy()
            if mask_sum is None:
                mask_sum = mask_probability.copy()
                presence_sum = presence_probability.copy()
            else:
                mask_sum += mask_probability
                presence_sum += presence_probability
            timings["augmentation_aggregation"] += perf_counter() - before
    before = perf_counter()
    mask_sum /= np.float32(len(transforms))
    presence_sum /= np.float32(len(transforms))
    timings["augmentation_aggregation"] += perf_counter() - before
    return mask_sum, presence_sum, timings


def predict_fold_probabilities(model, sample, config, *, tta="d4"):
    """Equal mean of three independently normalized members' D4 probabilities."""
    device = next(model.parameters()).device
    masks, heads = [], []
    timings = {'prepare': 0., 'forward': 0., 'augmentation_aggregation': 0.}
    for member, stats in zip(model.members, config['member_normalization_statistics'], strict=True):
        before = perf_counter()
        normalized = normalize_sample(sample, 'zscore', stats)
        tensor = torch.from_numpy(normalized['input']).unsqueeze(0).to(device)
        _synchronize(device)
        timings['prepare'] += perf_counter() - before
        mask, head, measured = predict_probabilities(member, tensor, tta=tta)
        masks.append(mask)
        heads.append(head)
        for name, seconds in measured.items():
            timings[name] += seconds
    before = perf_counter()
    result = np.mean(masks, axis=0), np.mean(heads, axis=0)
    timings['augmentation_aggregation'] += perf_counter() - before
    return *result, timings


def prediction_from_probabilities(sample, mask_probability, presence_probability, *,
                                  presence_threshold=0.5, mask_threshold=0.5, method="pixel_union"):
    """분할 확률 (S, S)·존재 확률 하나를 공통 규칙으로 polygon 예측으로 바꾼다.

    두 임계값의 기본값은 0.5이며 모두 확률 ≥ 임계값으로 판정한다.
    존재 head가 양성이어도 남은 유효 화소가 없으면 polygon은 빈 목록이다.
    이 경우를 별도 표시하며, 산사태 없음 판정을 prior 도형으로 대체하지 않는다.
    """
    if method == "probability_contour":
        # 경계 모듈은 기존 화소 함수를 공유하므로 사용 시점에만 가져온다.
        from postprocessing.boundary_processing import boundary_prediction
        return boundary_prediction(sample, mask_probability, presence_probability,
                                   method=method, presence_threshold=presence_threshold,
                                   mask_threshold=mask_threshold)
    if method != "pixel_union":
        raise ValueError("Unknown prediction postprocessing method")
    probabilities = np.asarray(mask_probability)
    probability = float(presence_probability)
    if any(not np.isfinite(value) or not 0 <= value <= 1
           for value in (presence_threshold, mask_threshold)):
        raise ValueError("Prediction thresholds must be finite and within [0, 1]")
    if probabilities.shape != sample["inside"].shape or not np.isfinite(probabilities).all() or not np.isfinite(probability):
        raise ValueError("Invalid prediction probability array")
    is_present = probability >= presence_threshold
    polygons = []
    if is_present:
        # 원영상 밖 padding과 무효 화소를 제거한 뒤 원영상 좌표로 복원한다.
        usable = sample["official_valid"] & sample["numeric_valid"] & sample["inside"]
        polygons = masks_to_polygons((probabilities >= mask_threshold) & usable,
                                     sample["origin"], sample["original_shape"])
    return {"polygons": polygons, "presence_probability": probability,
            "presence_positive": is_present,
            "positive_without_polygon": is_present and not polygons}


def run_inference(input_dir, output_path, checkpoint_path, *, device="cpu", config=None, tta="none"):
    """입력된 모든 관심지·날짜를 예측하고 전체 성공 후 CSV를 저장한다.

    빈 polygon은 공식 CSV 예시처럼 빈 필드로 쓰고, 날짜는 밴드 파일명의
    YYYYMMDD를 유지한다. 정답 라벨과 미리보기 이미지는 읽지 않는다.
    반환값은 예측 배열 대신 처리 건수·장치·시간 등의 실행 요약이다.
    """
    if tta not in ("none", "d4"):
        raise ValueError("tta must be none or d4")
    postprocessing = postprocessing_options(config)
    output_path = Path(output_path)
    if (output_path.resolve().is_relative_to(Path(input_dir).resolve())
            or output_path.resolve() == Path(checkpoint_path).resolve()):
        raise ValueError("Prediction output must be separate from inputs and checkpoint")
    started = perf_counter()
    timings = dict.fromkeys(("load", "prepare", "forward", "augmentation_aggregation", "polygon", "write"), 0.0)
    before = perf_counter()
    scenes = _inputs(input_dir)
    model, saved = load_checkpoint(checkpoint_path, device=device)
    saved_config = inference_config(saved["config"])
    if config is not None and inference_config(config) != saved_config:
        raise ValueError("Packaged inference configuration differs from the checkpoint")
    # 기본 96px를 강제하지 않고 저장 당시의 96/128px·원점 정책을 그대로 적용한다.
    roi = input_parameters(saved_config if config is None else config)
    model.eval()
    actual_device = next(model.parameters()).device
    timings["load"] += perf_counter() - before
    rows = []
    presence_positive = positive_predictions = positive_without_polygon = 0
    contour_degenerate_ring_count = 0
    for location, date, sites, path in scenes:
        before = perf_counter()
        raw = np.load(path, allow_pickle=False)
        if (raw.ndim != 3 or raw.shape[-1] != 4 or min(raw.shape[:2]) < 1
                or raw.dtype.kind not in "uif"):
            raise ValueError("Expected a numeric H x W x 4 band array")
        timings["load"] += perf_counter() - before
        for site in sites:
            before = perf_counter()
            sample = prepare_sample(raw, site["prior_polygon"], **roi)
            # (5, S, S)에 배치 축을 붙여 관심지 하나를 (1, 5, S, S)로 추론한다.
            tensor = torch.from_numpy(sample['input']).unsqueeze(0).to(actual_device)
            _synchronize(actual_device)
            timings["prepare"] += perf_counter() - before
            if saved_config['model'] in ('small_unet_coverage_fold3_v1', 'small_unet_coverage_baseline_presence_fold3_v1'):
                (masks, probabilities, prediction_timings) = predict_fold_probabilities(model, sample, saved_config, tta=tta)
            else:
                (masks, probabilities, prediction_timings) = predict_probabilities(model, tensor, tta=tta)
            for name, seconds in prediction_timings.items():
                timings[name] += seconds
            before = perf_counter()
            prediction = prediction_from_probabilities(sample, masks[0], probabilities[0], **postprocessing)
            contour_degenerate_ring_count += prediction.get("contour_degenerate_ring_count", 0)
            polygons = prediction["polygons"]
            if prediction["presence_positive"]:
                presence_positive += 1
                if polygons:
                    positive_predictions += 1
                else:
                    positive_without_polygon += 1
            rows.append({"location_id": location, "date": date,
                         "site_id": site["site_id"],
                         "polygons": json.dumps(polygons, separators=(",", ":"), allow_nan=False)
                         if polygons else ""})
            timings["polygon"] += perf_counter() - before
    before = perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # 완성된 CSV를 같은 폴더의 임시 파일에 먼저 쓴다. 중간 실패로 일부 행만
        # 있는 결과가 최종 경로에 남지 않도록 저장 성공 후 파일을 교체한다.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=output_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    timings["write"] = perf_counter() - before
    return {"device": str(actual_device), "scenes": len(scenes),
            "locations": len({scene[0] for scene in scenes}), "rows": len(rows),
            "presence_positive": presence_positive,
            "positive_predictions": positive_predictions,
            "positive_without_polygon": positive_without_polygon,
            "empty_predictions": len(rows) - positive_predictions,
            "tta": tta, "directions": 8 if tta == "d4" else 1,
            "postprocessing_method": postprocessing.get("method", "pixel_union"),
            "contour_degenerate_ring_count": contour_degenerate_ring_count,
            "thresholds": {"presence": postprocessing["presence_threshold"],
                           "mask": postprocessing["mask_threshold"]},
            "timings_seconds": timings, "total_seconds": perf_counter() - started}


def run_from_environment(assets_dir="assets", *, device="cpu"):
    """공식 경로 환경변수와 assets/config.json의 설정·가중치로 추론한다."""
    input_dir = os.environ.get("AIF_INPUT_DIR")
    output_path = os.environ.get("AIF_PREDICTION_PATH")
    if not input_dir or not output_path:
        raise ValueError("AIF_INPUT_DIR and AIF_PREDICTION_PATH are required")
    assets = Path(assets_dir).resolve()
    config = json.loads((assets / "config.json").read_text(encoding="utf-8"))
    inference_config(config)
    checkpoint_name = config.get("checkpoint")
    if not isinstance(checkpoint_name, str) or not checkpoint_name:
        raise ValueError("Packaged checkpoint path is required")
    relative = Path(checkpoint_name)
    checkpoint = (assets / relative).resolve()
    if relative.is_absolute() or not checkpoint.is_relative_to(assets):
        raise ValueError("Packaged checkpoint must be inside assets")
    return run_inference(input_dir, output_path, checkpoint, device=device, config=config,
                         tta=config.get("tta", "none"))
