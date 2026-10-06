"""E007 합성 검증. 실제 데이터·학습 가중치·기존 제출본에는 접근하지 않는다."""

import copy
import os
from pathlib import Path
import unittest

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import numpy as np
from shapely.geometry import box
import torch

from analysis.data import geometry
from inference.unet import predict_probabilities, prediction_from_probabilities
from models.unet import SmallUNet
from preprocessing.model_input import prepare_sample
from training.unet import set_seed

PROJECT = Path(__file__).resolve().parents[2]


class DirectionalModel(torch.nn.Module):
    """위치에 민감한 합성 모델로 잘못된 역변환·logit 평균을 검출한다."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.seen = []

    def forward(self, value):
        self.seen.append(value.detach().cpu().numpy().copy())
        vertical = torch.arange(value.shape[-2], device=value.device).float()[None, :, None] * .17
        segmentation = value[:, 0] - .7 * value[:, 1] + .3 * value[:, 4] + vertical + self.anchor
        presence = value[:, 2, 0, 0] * 2 - .9 + self.anchor
        return {'segmentation': segmentation[:, None], 'presence': presence}


class PriorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, value):
        return {'segmentation': (value[:, 4:5] - .5) * 20 + self.anchor,
                'presence': torch.full((len(value),), 9., device=value.device) + self.anchor}


class TTACPU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        set_seed()

    def test_eight_transforms_manual_probability_average_and_original(self):
        source = np.random.default_rng(42).normal(size=(1, 5, 8, 8)).astype(np.float32)
        tensor = torch.from_numpy(source.copy())
        model = DirectionalModel().eval()
        weights = copy.deepcopy(model.state_dict())
        masks, heads, timing = predict_probabilities(model, tensor, tta='d4')
        self.assertEqual(len(model.seen), 8)
        expected_masks, expected_heads, presence_logits = [], [], []
        for code in range(8):
            transformed = np.rot90(source, code % 4, axes=(-2, -1))
            if code >= 4:
                transformed = np.flip(transformed, -1)
            np.testing.assert_array_equal(model.seen[code], transformed)
            logits = (transformed[:, 0] - .7 * transformed[:, 1] + .3 * transformed[:, 4]
                      + np.arange(8, dtype=np.float32)[None, :, None] * .17)
            probability = (1 / (1 + np.exp(-logits))).astype(np.float32)
            if code >= 4:
                probability = np.flip(probability, -1)
            expected_masks.append(np.rot90(probability, -(code % 4), axes=(-2, -1)))
            logit = transformed[:, 2, 0, 0] * 2 - .9
            presence_logits.append(logit)
            expected_heads.append(1 / (1 + np.exp(-logit)))
        np.testing.assert_allclose(masks, np.mean(expected_masks, axis=0, dtype=np.float32), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(heads, np.mean(expected_heads, axis=0, dtype=np.float32), rtol=1e-6, atol=1e-7)
        self.assertGreater(abs(float(heads[0]) - float(1 / (1 + np.exp(-np.mean(presence_logits))))), .001)
        self.assertEqual(masks.dtype, np.float32)
        self.assertEqual(heads.dtype, np.float32)
        np.testing.assert_array_equal(tensor.numpy(), source)
        for name, value in weights.items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(timing[name] >= 0 for name in ('forward', 'augmentation_aggregation')))

    def test_none_preserves_existing_cpu_sigmoid_results(self):
        model = SmallUNet().eval()
        tensor = torch.randn(1, 5, 96, 96)
        with torch.inference_mode():
            previous = model(tensor)
            expected_mask = torch.sigmoid(previous['segmentation'].cpu())[:, 0].numpy()
            expected_head = torch.sigmoid(previous['presence'].cpu()).numpy()
        masks, heads, _ = predict_probabilities(model, tensor)
        np.testing.assert_array_equal(masks, expected_mask)
        np.testing.assert_array_equal(heads, expected_head)

    def test_padding_small_positive_invalid_pixels_and_coordinates(self):
        raw = np.full((64, 64, 4), 4000, dtype=np.float32)
        raw[30, 30, 1:3] = 0  # Green/Blue의 0만으로 공식 무효가 되지 않는다.
        original = raw.copy()
        prior = box(30, 30, 31, 31)
        sample = prepare_sample(raw, prior, roi_size=96, origin_policy='shift_inside')
        before = copy.deepcopy(sample)
        self.assertTrue((~sample['inside']).any())
        self.assertEqual(sample['state'], 'missing')
        model = PriorModel().eval()
        masks, heads, _ = predict_probabilities(model, torch.from_numpy(sample['input'])[None], tta='d4')
        prediction = prediction_from_probabilities(sample, masks[0], heads[0])
        self.assertTrue(geometry(prediction['polygons']).equals(prior))
        for field in ('input', 'prior', 'inside', 'official_valid', 'numeric_valid'):
            np.testing.assert_array_equal(sample[field], before[field])
        np.testing.assert_array_equal(raw, original)
        for bad in (0., float('nan')):
            invalid = raw.copy()
            invalid[30, 30, 0] = bad
            invalid_sample = prepare_sample(invalid, prior, roi_size=96, origin_policy='shift_inside')
            m, h, _ = predict_probabilities(model, torch.from_numpy(invalid_sample['input'])[None], tta='d4')
            result = prediction_from_probabilities(invalid_sample, m[0], h[0])
            self.assertEqual(result['polygons'], [])
            self.assertTrue(result['positive_without_polygon'])
        self.assertEqual(prediction_from_probabilities(sample, masks[0], .1)['polygons'], [])

    def test_invalid_mode_nonfinite_and_shape_fail(self):
        tensor = torch.zeros(1, 5, 8, 8)
        with self.assertRaises(ValueError):
            predict_probabilities(DirectionalModel(), tensor, tta='d8')
        class BrokenModel(DirectionalModel):
            def forward(self, value):
                output = super().forward(value)
                output['segmentation'].fill_(float('nan'))
                return output
        with self.assertRaises(ValueError):
            predict_probabilities(BrokenModel(), tensor, tta='d4')
        class ShapeModel(DirectionalModel):
            def forward(self, value):
                output = super().forward(value)
                output['presence'] = output['presence'][:, None]
                return output
        with self.assertRaises(ValueError):
            predict_probabilities(ShapeModel(), tensor, tta='d4')




@unittest.skipUnless(os.environ.get('LANDSLIDE_TEST_MPS') == '1', 'Set LANDSLIDE_TEST_MPS=1 to opt into MPS checks')
class TTAMPS(unittest.TestCase):
    def test_cpu_mps_eight_direction_probabilities(self):
        self.assertTrue(torch.backends.mps.is_available(), 'MPS required')
        self.assertEqual(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'], '0')
        torch.set_num_threads(2)
        set_seed()
        cpu = SmallUNet().eval()
        mps = copy.deepcopy(cpu).to('mps').eval()
        tensor = torch.randn(1, 5, 96, 96)
        before = copy.deepcopy(cpu.state_dict())
        for mode in ('none', 'd4'):
            cm, ch, _ = predict_probabilities(cpu, tensor, tta=mode)
            mm, mh, _ = predict_probabilities(mps, tensor.to('mps'), tta=mode)
            np.testing.assert_allclose(mm, cm, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(mh, ch, rtol=1e-4, atol=1e-5)
        for name, value in mps.state_dict().items():
            torch.testing.assert_close(value.cpu(), before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in mps.parameters()))


if __name__ == '__main__':
    unittest.main()
