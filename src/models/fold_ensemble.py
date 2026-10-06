"""Three preserved E090 models; each keeps its own training normalization."""
from torch import nn

from models.unet import SmallUNet


class CoverageFoldEnsemble(nn.Module):
    model_id = 'small_unet_coverage_fold3_v1'

    def __init__(self, roi_size=96, input_channels=6, *, baseline_presence=False):
        super().__init__()
        if roi_size != 96 or input_channels != 6:
            raise ValueError('E090 ensemble requires 96px RGBNIR/prior/NDVI')
        self.roi_size, self.input_channels = roi_size, input_channels
        if baseline_presence:
            from models.logit_blend import BaselineLogitBlend
            self.model_id = 'small_unet_coverage_baseline_presence_fold3_v1'
            self.members = nn.ModuleList(BaselineLogitBlend(roi_size, input_channels, candidate_only=True)
                                         for _ in range(3))
        else:
            self.members = nn.ModuleList(SmallUNet(roi_size, input_channels) for _ in range(3))
