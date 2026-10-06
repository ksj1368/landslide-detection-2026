import torch
from torch import nn
from torch.nn import functional as F


def _block(inputs, outputs):
    """공간 크기를 유지하면서 채널 수를 바꾸는 합성곱 두 층을 만든다."""
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class SmallUNet(nn.Module):
    """픽셀 분할과 대상 관심지의 산사태 양성 여부를 함께 학습하는 두 출력 모델."""

    model_id = "small_unet_v1"

    def __init__(self, roi_size=96, input_channels=5):
        super().__init__()
        if type(roi_size) is not int or roi_size not in (96, 128):
            raise ValueError("Unsupported U-Net ROI size")
        if type(input_channels) is not int or input_channels not in (5, 6, 7):
            raise ValueError("Unsupported U-Net input channels")
        self.roi_size, self.input_channels = roi_size, input_channels
        # 입력 채널은 R/G/B/NIR/prior다. 인코더는 공간을 줄이며 특징 채널을 늘린다.
        self.encoder = nn.ModuleList([_block(5, 16), _block(16, 32),
                                      _block(32, 64), _block(64, 128)])
        self.decoder = nn.ModuleList([_block(128 + 64, 64), _block(64 + 32, 32),
                                      _block(32 + 16, 16)])
        # 분할 head는 위치별 점수, 존재 head는 ROI 전체를 요약한 점수 하나를 만든다.
        self.segmentation = nn.Conv2d(16, 1, 1)
        self.presence = nn.Linear(128, 1)
        self.boundary = nn.Identity()
        if input_channels > 5:
            # 5채널 전체 모델 초기화를 먼저 마쳐 공통 가중치와 RNG 흐름을 보존한다.
            previous = self.encoder[0][0]
            with torch.random.fork_rng(devices=[]):
                expanded = nn.Conv2d(input_channels, 16, 3, padding=1)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :5].copy_(previous.weight)
                expanded.bias.copy_(previous.bias)
            self.encoder[0][0] = expanded

    def forward(self, x):
        """(N, C, S, S)를 분할 (N, 1, S, S)·존재 (N,) logit으로 변환한다.

        N은 배치 크기, C는 5·6·7, S는 96 또는 128이다. logit은 sigmoid 적용 전의 점수로,
        학습 손실에는 그대로 전달하고 추론 단계에서 0~1 확률로 변환한다.
        """
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.input_channels, self.roi_size, self.roi_size):
            raise ValueError(f"U-Net expects N x {self.input_channels} x {self.roi_size} x {self.roi_size} input")
        return self._forward_impl(x)

    def _forward_impl(self, x, return_features=False):
        features = []
        # 해상도는 S → S/2 → S/4 → S/8로 줄어든다. 복원에 쓸 특징도 보관한다.
        for index, block in enumerate(self.encoder):
            x = block(F.max_pool2d(x, 2) if index else x)
            features.append(x)
        # 가장 깊은 특징의 높이·너비를 평균하여 대상 관심지의 산사태 양성 여부를 판정한다.
        presence = self.presence(x.mean(dim=(-2, -1))).squeeze(-1)
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            # skip connection: 같은 해상도의 인코더 특징을 채널 축으로 연결해
            # 축소 과정에서 잃기 쉬운 위치·경계 정보를 공간 복원에 활용한다.
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        x = self.boundary(x)
        outputs = {"segmentation": self.segmentation(x), "presence": presence}
        if return_features:
            outputs["features"] = {"first_encoder16": features[0], "final_decoder16": x}
        return outputs

    def forward_with_features(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.input_channels, self.roi_size, self.roi_size):
            raise ValueError(f"U-Net expects N x {self.input_channels} x {self.roi_size} x {self.roi_size} input")
        return self._forward_impl(x, return_features=True)

    def forward_with_shallow_features(self, x):
        """Return legacy outputs plus the first 16-channel encoder feature."""
        outputs = self.forward_with_features(x)
        return {key: value for key, value in outputs.items() if key != "features"}, outputs["features"]["first_encoder16"]


def build_model(model_id, roi_size=96, input_channels=5):
    if model_id == SmallUNet.model_id:
        return SmallUNet(roi_size, input_channels)
    if model_id in ('small_unet_coverage_fold3_v1', 'small_unet_coverage_baseline_presence_fold3_v1'):
        from models.fold_ensemble import CoverageFoldEnsemble
        return CoverageFoldEnsemble(baseline_presence=model_id == 'small_unet_coverage_baseline_presence_fold3_v1')
    raise ValueError('Unsupported public-release model')
