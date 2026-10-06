import random
import copy
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models.unet import SmallUNet


CONFIG = {"model": "small_unet_v1", "channels": [16, 32, 64, 128],
          "roi_size": 96, "input_bands": ["R", "G", "B", "NIR", "prior"],
          "reflectance_divisor": 10000.0, "normalization": "none", "seed": 42,
          "optimizer": "Adam", "learning_rate": 1e-3,
          "loss": "segmentation_BCE_plus_presence_BCE", "scope": "synthetic_only",
          "presence_threshold": 0.5, "mask_threshold": 0.5}


BCE_DICE = "segmentation_BCE_Dice_plus_presence_BCE"


COVERAGE_BCE_DICE = "coverage_BCE_binary_Dice_plus_presence_BCE"


SITE_BCE_DICE = "binary_BCE_site_Dice_plus_presence_BCE"


DICE_PARAMETERS = {"epsilon": 1e-6, "bce_weight": 0.5, "dice_weight": 0.5,
                   "aggregation": "positive_roi_mean", "empty_positive": "bce_only"}


def set_seed():
    """사용하는 난수 생성기의 seed를 전역 기준 42로 맞춘다."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)


def make_batch(samples, device="cpu"):
    """표본 배열 앞에 배치 축 N을 쌓고 지정 장치로 옮긴다.

    input은 (N, 5, S, S), 분할 정답·마스크는 (N, 1, S, S),
    존재 정답·감독 여부는 (N,)가 된다. 좌표 등 메타데이터는 배치에 넣지 않는다.
    """
    if not samples:
        raise ValueError("Empty batch")
    names = ["input", "target", "segmentation_valid", "presence", "presence_valid"]
    for field in ("coverage_target", "dice_site_weight"):
        if any(field in sample for sample in samples):
            if not all(field in sample for sample in samples):
                raise ValueError(f"Batch mixes present and missing {field}")
            names.append(field)
    return {name: torch.from_numpy(np.stack([sample[name] for sample in samples])).to(device)
            for name in names}


def supervised_loss(outputs, batch, config=None, *, diagnostics=None):
    """감독 가능한 화소·표본에서 선택한 손실을 계산해 (합계, 분할, 존재)를 반환한다.

    BCE는 이진 정답에 대한 손실이다. with_logits 함수가 sigmoid 계산을 포함하므로
    모델의 logit을 그대로 받는다. 분할은 유효 화소 수, 존재는 유효 표본 수로
    각각 평균한 뒤 두 손실을 같은 가중치로 더한다.
    """
    config = CONFIG if config is None else config
    _validate_loss(config)
    seg_valid = batch["segmentation_valid"].bool()
    pre_valid = batch["presence_valid"].bool()
    if outputs["segmentation"].shape != batch["target"].shape or outputs["presence"].shape != batch["presence"].shape:
        raise ValueError("Output/target shapes differ")
    if not all(torch.isfinite(value).all() for value in outputs.values()):
        raise ValueError("Nonfinite model output")
    if not pre_valid.any() and not seg_valid.any():
        raise ValueError("Batch has no supervision")
    # BCE 계산 전에 제외 대상을 걸러 NaN 정답이나 불필요한 gradient가 섞이지 않게 한다.
    # 한 head만 감독이 없으면 계산 그래프와 연결된 0을 만들어 그 head의 gradient도 0으로 둔다.
    bce_target = batch["target"]
    if config["loss"] == COVERAGE_BCE_DICE:
        bce_target = batch.get("coverage_target")
        if (bce_target is None or bce_target.shape != batch["target"].shape
                or bce_target.dtype != torch.float32
                or not torch.isfinite(bce_target[seg_valid]).all()
                or ((bce_target[seg_valid] < 0) | (bce_target[seg_valid] > 1)).any()):
            raise ValueError("Coverage BCE requires aligned finite FP32 targets in [0,1]")
    seg = (F.binary_cross_entropy_with_logits(outputs["segmentation"][seg_valid], bce_target[seg_valid])
           if seg_valid.any() else outputs["segmentation"].sum() * 0)
    segmentation_bce, dice_loss = seg, None
    bce_contribution, dice_contribution = seg, None
    dice_roi_count = 0
    auxiliary = outputs["segmentation"].sum() * 0
    if config['loss'] in (BCE_DICE, COVERAGE_BCE_DICE, SITE_BCE_DICE) and seg_valid.any():
        target = torch.where(seg_valid, batch['target'], 0).flatten(1)
        probability = torch.where(seg_valid, outputs['segmentation'].sigmoid(), 0).flatten(1)
        positive = target.sum(1) > 0
        if positive.any():
            (target, probability) = (target[positive], probability[positive])
            epsilon = config['loss_parameters']['epsilon']
            dice = 1 - (2 * (target * probability).sum(1) + epsilon) / (target.sum(1) + probability.sum(1) + epsilon)
            dice_loss = dice.mean()
            if config['loss'] == SITE_BCE_DICE:
                weights = batch.get('dice_site_weight')
                if weights is None or weights.shape != positive.shape or weights.dtype != torch.float32 or (not torch.isfinite(weights).all()) or (weights <= 0).any():
                    raise ValueError('Site Dice requires finite positive FP32 sample weights')
                if diagnostics is not None:
                    diagnostics['dice_loss_unweighted'] = dice_loss.detach().item()
                    diagnostics['dice_batch_weight_mean'] = weights[positive].mean().item()
                dice_loss = (dice * weights[positive]).mean()
            (bce_contribution, dice_contribution) = (0.5 * seg, 0.5 * dice_loss)
            seg = bce_contribution + dice_contribution
            if diagnostics is not None:
                dice_roi_count = int(positive.sum().item())
    pre = (F.binary_cross_entropy_with_logits(outputs["presence"][pre_valid], batch["presence"][pre_valid])
           if pre_valid.any() else outputs["presence"].sum() * 0)
    if diagnostics is not None:
        diagnostics.update({"segmentation_bce": segmentation_bce.detach().item(),
                            "dice_loss": None if dice_loss is None else dice_loss.detach().item(),
                            "bce_contribution": bce_contribution.detach().item(),
                            "dice_contribution": 0.0 if dice_contribution is None else dice_contribution.detach().item(),
                            "presence_samples": int(pre_valid.sum().item()),
                            "presence_positive_samples": int((batch["presence"][pre_valid] > 0).sum().item()),
                            "dice_roi_count": dice_roi_count,
                            "segmentation_pixels": int(seg_valid.sum().item())})
    total = seg + pre + auxiliary
    return total, seg, pre


def train_step(model, optimizer, batch, config=None, *, gradient_observer=None):
    """배치 하나로 순전파 → 손실 → 역전파 → 가중치 갱신을 한 번 수행한다."""
    model.train()
    # 이전 update의 gradient를 비워 이번 배치의 변화량만 optimizer에 전달한다.
    optimizer.zero_grad(set_to_none=True)
    diagnostics = {} if config is not None and config.get("diagnostics", False) else None
    (total, seg, pre) = supervised_loss(model(batch['input']), batch, config, diagnostics=diagnostics)
    if not torch.isfinite(total):
        raise ValueError("Nonfinite loss")
    total.backward()
    if gradient_observer is not None:
        # 선택한 update에서 이미 계산한 gradient만 읽는다. 추가 forward/backward는 없다.
        gradient_observer(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable):
        raise ValueError("Missing or nonfinite gradient")
    if diagnostics is not None:
        # 이미 계산한 gradient를 읽기만 한다. clipping이나 추가 backward는 없다.
        gradient_l2 = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(p.grad.detach()) for p in trainable])).item()
        if not math.isfinite(gradient_l2):
            raise ValueError("Nonfinite gradient norm")
        diagnostics.update(gradient_l2=gradient_l2,
                           learning_rate=optimizer.param_groups[0]["lr"],
                           next_learning_rate=optimizer.param_groups[0]["lr"])
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite updated parameter")
    return {"loss": total.item(), "segmentation_loss": seg.item(), "presence_loss": pre.item(),
            **(diagnostics or {})}


def evaluate_loss(model, samples, device="cpu", batch_size=8):
    """가중치를 갱신하지 않고 검증 표본 전체의 두 손실과 감독 수를 집계한다."""
    model.eval()
    sums = [0.0, 0.0]
    counts = [0, 0]
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = make_batch(samples[start:start + batch_size], device)
            if not batch["segmentation_valid"].any() and not batch["presence_valid"].any():
                continue
            _, seg, pre = supervised_loss(model(batch["input"]), batch)
            for index, (loss, valid) in enumerate(((seg, batch["segmentation_valid"]), (pre, batch["presence_valid"]))):
                count = int(valid.sum().item())
                # 배치 평균을 단순 평균하면 작은 마지막 배치의 비중이 커진다.
                # 각 평균을 감독 수로 가중해 전체 유효 화소/표본 기준 평균을 복원한다.
                sums[index] += loss.item() * count
                counts[index] += count
    if not any(counts):
        raise ValueError("Validation has no supervision")
    values = [total / count if count else None for total, count in zip(sums, counts)]
    return {"loss": sum(value for value in values if value is not None),
            "segmentation_loss": values[0], "presence_loss": values[1],
            "segmentation_pixels": counts[0], "presence_samples": counts[1]}


def train_single_split(train_samples, validation_samples, *, updates, batch_size, device="cpu"):
    """주어진 단일 분할을 간단히 학습·검증하는 경로로, 합성 검사에서 사용한다.

    기본 96px 모델을 쓰고 update마다 표본을 무작위 선택한다. 실제 Holdout의
    D4·순차 배치·주기적 검증·best 저장 루프는 training.holdout.fit_holdout에 있다.
    """
    from evaluation.contracts import validate_partition

    validate_partition(train_samples, validation_samples)
    if updates < 1 or batch_size < 1:
        raise ValueError("updates and batch_size must be positive")
    # 미라벨·ignore를 음성으로 바꾸지 않고, 감독이 있는 표본만 학습에 넣는다.
    supervised = [s for s in train_samples if s["presence_valid"] or s["segmentation_valid"].any()]
    if not supervised:
        raise ValueError("Training split has no supervision")
    set_seed()
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    rng = np.random.default_rng(42)
    history = []
    for update in range(updates):
        selected = rng.choice(len(supervised), size=batch_size, replace=batch_size > len(supervised))
        metrics = train_step(model, optimizer, make_batch([supervised[i] for i in selected], device))
        history.append({"update": update + 1, **metrics})
    return model, optimizer, history, evaluate_loss(model, validation_samples, device, batch_size)


def save_checkpoint(path, model, optimizer, updates, *, config=None, metadata=None, scheduler=None):
    """가중치·optimizer 상태·설정·진행량을 새 checkpoint 파일에 함께 저장한다.

    'xb'로 기존 파일의 덮어쓰기를 막는다. Holdout의 best/last 갱신은 호출자가
    임시 파일을 완전히 저장한 뒤 현재 run의 해당 파일을 교체하는 방식이다.
    """
    config = copy.deepcopy(CONFIG if config is None else config)
    _check_config(config)
    if (getattr(model, "model_id", None) != config["model"] or model.roi_size != config["roi_size"]
            or model.input_channels != len(config["input_bands"])):
        raise ValueError("Model and checkpoint architecture or input configuration differ")
    scheduler_state = {} if scheduler is None else {"scheduler": scheduler.state_dict()}
    _validate_scheduler_state(config, scheduler_state, updates)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config, "updates": updates, "seed": 42,
                    "metadata": copy.deepcopy(metadata or {}), **scheduler_state}, handle)


def load_checkpoint(path, device="cpu"):
    """설정을 확인해 모델을 복구하고 (평가 모드 모델, 저장된 정보)를 반환한다.

    파일은 먼저 CPU로 읽고 가중치를 넣은 모델만 요청 장치로 옮긴다.
    optimizer 상태는 반환 정보에 포함되지만 이 함수가 optimizer를 재생성하지는 않는다.
    """
    from models.checkpoint import load_checkpoint as restore_model
    model, saved = restore_model(path, device=device)
    _check_config(saved.get("config"))
    _validate_scheduler_state(saved["config"], saved, saved["updates"])
    if saved.get("seed") != 42:
        raise ValueError("Checkpoint configuration differs from the implemented contract")
    return model, saved


def _validate_loss(config):
    name, parameters = config.get('loss'), config.get('loss_parameters', {})
    if name == CONFIG['loss'] and parameters == {}:
        return
    if name in (BCE_DICE, COVERAGE_BCE_DICE, SITE_BCE_DICE) and parameters == DICE_PARAMETERS:
        return
    raise ValueError('Unsupported or inconsistent training loss configuration')


def _scheduler_configuration(config):
    settings = config.get('scheduler', {'name': 'constant'})
    if settings != {'name': 'constant'}:
        raise ValueError('Only the fixed learning rate is supported')
    return settings


def create_scheduler(optimizer, config):
    _scheduler_configuration(config)
    return None


def create_optimizer(parameters, config):
    if config.get('optimizer', 'Adam') != 'Adam' or config.get('optimizer_parameters', {}):
        raise ValueError('Only default Adam is supported')
    return torch.optim.Adam([p for p in parameters if p.requires_grad], lr=config['learning_rate'])


def _validate_scheduler_state(config, saved, updates):
    _scheduler_configuration(config)
    if saved.get('scheduler') is not None:
        raise ValueError('Constant learning-rate checkpoint has scheduler state')


def _check_config(config):
    from models.checkpoint import validate_inference_config
    if not isinstance(config, dict) or config.get('scope') not in ('synthetic_only', 'holdout'):
        raise ValueError('Unsupported checkpoint scope')
    validate_inference_config(config)
    if config['model'] != SmallUNet.model_id:
        raise ValueError('Training supports SmallUNet only')
    if any(config.get(key) != value for key, value in CONFIG.items()
           if key not in ('model', 'channels', 'scope', 'roi_size', 'loss', 'input_bands', 'normalization')):
        raise ValueError('Checkpoint configuration differs from the implemented contract')
    _validate_loss(config)
    _scheduler_configuration(config)
    if any(config.get(k) is not None for k in ('refinement', 'common_error_method', 'brightness_augmentation', 'prior_augmentation')):
        raise ValueError('Unsupported auxiliary training method')
    if type(config.get('diagnostics', False)) is not bool:
        raise ValueError('Training diagnostics must be boolean')
