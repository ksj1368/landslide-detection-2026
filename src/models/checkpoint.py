"""학습 실행 모듈 없이 U-Net과 추론에 필요한 설정을 공통 복구한다."""

import copy

import torch

from models.unet import SmallUNet, build_model
from preprocessing.model_input import roi_parameters, index_configuration, normalization_configuration


INFERENCE_VALUES = {
    "model": "small_unet_v1", "channels": [16, 32, 64, 128],
    "input_bands": ["R", "G", "B", "NIR", "prior"],
    "reflectance_divisor": 10000.0, "normalization": "none", "seed": 42,
    "presence_threshold": 0.5, "mask_threshold": 0.5,
}


def validate_inference_config(config):
    """지원하는 모델·ROI·입력·임계값만 검사하며 학습 손실은 검사하지 않는다.

    origin_policy가 없는 기존 checkpoint는 E001의 중심 crop으로 복구한다.
    optimizer·손실·학습 범위의 검사는 학습 모듈에 남겨 둔다.
    """
    if not isinstance(config, dict) or "roi_size" not in config:
        raise ValueError("Missing inference checkpoint configuration")
    roi_parameters(config)
    if (config.get('model') not in (SmallUNet.model_id, 'small_unet_coverage_fold3_v1',
                                  'small_unet_coverage_baseline_presence_fold3_v1')
            or config.get('channels') != INFERENCE_VALUES['channels']):
        raise ValueError('Unsupported checkpoint model architecture/channels')
    if (type(config.get("seed")) is not int
            or any(config.get(key) != value for key, value in INFERENCE_VALUES.items()
                   if key not in ("model", "channels", "input_bands", "normalization"))):
        raise ValueError("Checkpoint configuration differs from the implemented inference contract")
    normalization_configuration(config)
    if config.get('model') in ('small_unet_coverage_fold3_v1', 'small_unet_coverage_baseline_presence_fold3_v1'):
        stats = config.get('member_normalization_statistics')
        if (config.get('normalization') != 'none' or config.get('input_index') != 'NDVI'
                or config['roi_size'] != 96 or config.get('origin_policy') != 'shift_inside'
                or not isinstance(stats, list) or len(stats) != 3):
            raise ValueError('Invalid E090 fold ensemble input contract')
        for item in stats:
            normalization_configuration({'normalization': 'zscore', 'normalization_statistics': item})
    expected = index_configuration(config.get("input_index", "none"))
    if config.get("input_bands") != expected["input_bands"] or config.get("index_calculation") != expected.get("index_calculation"):
        raise ValueError("Checkpoint input index, channels or calculation contract differs")


def inference_config(config):
    """Project a checkpoint to its path-free inference contract."""
    validate_inference_config(config)
    return copy.deepcopy({**{key: config[key] for key in INFERENCE_VALUES if key != 'normalization'},
                          **roi_parameters(config), **index_configuration(config.get('input_index', 'none')),
                          **normalization_configuration(config),
                          **({'member_normalization_statistics': config['member_normalization_statistics']}
                             if 'member_normalization_statistics' in config else {})})


def load_checkpoint(path, device="cpu"):
    """기존 학습 checkpoint와 optimizer 없는 제출 checkpoint를 같은 모델로 복구한다."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint must contain a model and configuration")
    validate_inference_config(saved.get("config"))
    if type(saved.get("seed")) is not int or saved["seed"] != 42:
        raise ValueError("Checkpoint seed differs from the implemented inference contract")
    if type(saved.get("updates")) is not int or saved["updates"] < 0:
        raise ValueError("Checkpoint updates must be a nonnegative integer")
    model = build_model(saved["config"]["model"], saved["config"]["roi_size"],
                        len(saved["config"]["input_bands"]))
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), saved
