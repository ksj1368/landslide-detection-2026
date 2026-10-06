"""합성 입력의 네 밴드 정규화와 checkpoint 추론 계약을 검사한다."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from shapely.geometry import box

from models.checkpoint import INFERENCE_VALUES, inference_config, validate_inference_config
from preprocessing.model_input import (append_input_index, index_configuration, input_parameters,
                                       normalization_configuration, normalize_sample, prepare_sample)
from preprocessing.transforms import denormalize, transform_array


STATS = {"bands": ["R", "G", "B", "NIR"], "mean": [.2, .3, .1, .4],
         "std": [.1, .2, .05, .1], "median": [.15, .25, .08, .35],
         "iqr": [.2, .3, .1, .25]}


def fixture():
    raw = np.random.default_rng(42).integers(100, 10000, (64, 70, 4)).astype(np.float32)
    raw[0, 0, 0] = 0
    raw[1, 1, 1] = np.nan
    prior = box(25, 25, 34, 34)
    label = {"visible": True, "polygon": list(box(30, 30, 31, 31).exterior.coords)}
    sample = prepare_sample(raw, prior, label, input_index="NDVI", origin_policy="shift_inside")
    return raw, prior, label, sample


class TrainingNormalizationTests(unittest.TestCase):
    def test_fit_uses_only_supervised_train_pixels(self):
        from training.preprocessing_experiment import fit_frozen_statistics
        _, _, _, sample = fixture()
        sample.update(key=('train', '20200101', 'positive'))
        missing = copy.deepcopy(sample)
        missing.update(key=('train', '20200101', 'missing'), presence_valid=False)
        missing['input'][:4].fill(1000)
        manifest = {'seed':42, 'split_id':'synthetic', 'train_locations':['train'],
                    'validation_locations':['selection']}
        stats = fit_frozen_statistics([sample, missing], manifest)
        valid = sample['inside'] & sample['official_valid'] & sample['numeric_valid']
        pixels = sample['input'][:4].transpose(1,2,0)[valid]
        np.testing.assert_array_equal(stats['mean'], pixels.mean(axis=0,dtype=np.float64))
        np.testing.assert_array_equal(stats['std'], pixels.std(axis=0,dtype=np.float64))
        self.assertEqual(stats['bands'], ['R','G','B','NIR'])
        self.assertEqual(stats['fit_samples'], 1)
        with self.assertRaisesRegex(ValueError, 'non-train'):
            fit_frozen_statistics([sample, dict(sample,key=('selection','20200101','x'))],manifest)

    def test_manual_values_inverse_invalid_padding_and_other_channels(self):
        raw, _, label, base = fixture()
        before, label_before = raw.copy(), copy.deepcopy(label)
        source = base["input"].copy()
        eligible = base["inside"] & base["official_valid"] & base["numeric_valid"]
        for method, center, scale in (("zscore", "mean", "std"), ("robust", "median", "iqr")):
            actual = normalize_sample(base, method, STATS)
            expected = ((source[:4].astype(np.float64) - np.array(STATS[center])[:, None, None])
                        / np.array(STATS[scale])[:, None, None]).astype(np.float32)
            expected[:, ~eligible] = 0
            np.testing.assert_array_equal(actual["input"][:4], expected)
            np.testing.assert_array_equal(actual["input"][4:], source[4:])
            self.assertEqual(actual["input"].dtype, np.float32)
            self.assertGreater(float(actual["input"][:4].max()), 1.)  # 계산값 무클리핑
            for field in ("target", "segmentation_valid", "prior", "gt", "inside", "official_valid",
                          "numeric_valid", "index_valid", "presence", "presence_valid"):
                np.testing.assert_array_equal(actual[field], base[field])
            restored = denormalize(actual["input"][:4].transpose(1, 2, 0), STATS, method)
            np.testing.assert_allclose(restored[eligible], source[:4].transpose(1, 2, 0)[eligible],
                                       rtol=1e-4, atol=1e-5)
            actual["input"][0, 0, 0] = 99
            np.testing.assert_array_equal(base["input"], source)
        np.testing.assert_array_equal(raw, before)
        self.assertEqual(label, label_before)

    def test_d4_and_label_free_repreparation(self):
        raw, prior, label, base = fixture()
        for method in ("zscore", "robust"):
            original = normalize_sample(base, method, STATS)
            parameters = input_parameters(original)
            inference = prepare_sample(raw, prior, **parameters)
            np.testing.assert_array_equal(inference["input"], original["input"])
            for rotation in range(4):
                for flip in (False, True):
                    actual = prepare_sample(raw, prior, label, rotation=rotation, flip=flip, **parameters)
                    expected = transform_array(original["input"].transpose(1, 2, 0), rotation, flip)
                    np.testing.assert_array_equal(actual["input"], expected.transpose(2, 0, 1))
                    restored = transform_array(expected, 0, flip)
                    restored = transform_array(restored, (-rotation) % 4)
                    np.testing.assert_array_equal(restored.transpose(2, 0, 1), original["input"])
                    self.assertEqual(actual["origin"], original["origin"])

    def test_invalid_statistics_and_repeated_or_wrong_order_transform_fail(self):
        _, _, _, base = fixture()
        for method, scale in (("zscore", "std"), ("robust", "iqr")):
            for value in (0, -1, np.nan, np.inf):
                stats = copy.deepcopy(STATS); stats[scale][0] = value
                with self.assertRaises(ValueError): normalize_sample(base, method, stats)
        for stats in (None, {**STATS, "bands": ["B", "G", "R", "NIR"]},
                      {**STATS, "mean": [1, 2, 3]}, {**STATS, "mean": [np.inf, 1, 2, 3]}):
            with self.assertRaises(ValueError): normalize_sample(base, "zscore", stats)
        with self.assertRaises(ValueError): normalize_sample(base, "unknown", STATS)
        with self.assertRaises(ValueError): normalize_sample(base, "none", STATS)
        normalized = normalize_sample(base, "zscore", STATS)
        with self.assertRaises(ValueError): normalize_sample(normalized, "zscore", STATS)
        with self.assertRaises(ValueError): append_input_index(normalized, "NDVI")
        extreme = copy.deepcopy(STATS); extreme["std"] = [1e-100] * 4
        with self.assertRaisesRegex(ValueError, "finite FP32"):
            normalize_sample(base, "zscore", extreme)

    def test_checkpoint_inference_exports_only_needed_statistics_and_legacy_none(self):
        base = {**copy.deepcopy(INFERENCE_VALUES), "roi_size": 96, "origin_policy": "shift_inside",
                **index_configuration("NDVI")}
        for method, expected_fields in (("zscore", {"bands", "mean", "std"}),
                                       ("robust", {"bands", "median", "iqr"})):
            config = {**base, "normalization": method, "normalization_statistics": copy.deepcopy(STATS)}
            config["normalization_statistics"]["image_keys"] = ["local-only"]
            validate_inference_config(config)
            exported = inference_config(config)
            self.assertEqual(set(exported["normalization_statistics"]), expected_fields)
            self.assertEqual(exported["normalization_statistics"]["bands"], STATS["bands"])
            changed = copy.deepcopy(exported)
            changed["normalization_statistics"][next(iter(expected_fields - {"bands"}))][0] = 99
            self.assertNotEqual(exported, changed)
            bad = copy.deepcopy(config); bad["reflectance_divisor"] = 1
            with self.assertRaises(ValueError): validate_inference_config(bad)
        self.assertEqual(normalization_configuration({}), {"normalization": "none"})
        self.assertEqual(inference_config(base)["normalization"], "none")
        del base["normalization"]
        self.assertEqual(inference_config(base)["normalization"], "none")
        _, _, _, sample = fixture()
        np.testing.assert_array_equal(normalize_sample(sample)["input"], sample["input"])


if __name__ == "__main__":
    unittest.main()
