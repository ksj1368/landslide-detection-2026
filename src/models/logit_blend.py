"""Use candidate segmentation and baseline Presence when candidate_only=True."""
from torch import nn
from models.unet import SmallUNet


class BaselineLogitBlend(nn.Module):
    def __init__(self, roi_size=96, input_channels=5, *, candidate_only=False):
        super().__init__()
        self.roi_size, self.input_channels = roi_size, input_channels
        self.model_id = 'small_unet_logit_blend_v1'
        self.candidate_only = candidate_only
        if candidate_only:
            self.model_id = 'small_unet_baseline_presence_v1'
        self.baseline_model = SmallUNet(roi_size, input_channels)
        self.candidate_model = SmallUNet(roi_size, input_channels)

    def forward(self, x):
        baseline = self.baseline_model(x)
        candidate = self.candidate_model(x)
        return {'segmentation': (candidate['segmentation'] if self.candidate_only else
                                 .5 * (baseline['segmentation'] + candidate['segmentation'])),
                'presence': baseline['presence']}
