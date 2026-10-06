"""Synthetic-only actual-loader and fixed group Holdout contract tests."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from shapely.geometry import box

from analysis.data import file_snapshot
from preprocessing.model_input import prepare_sample
from training.holdout_data import load_holdout_data, _count_samples
from training.group_cv_experiment import make_folds


def _ring(polygon):
    return [list(point) for point in polygon.exterior.coords]


def _fixture(root, *, invalid=False, empty=False, clipped=False):
    """Five independent locations, each with positive/negative/missing sites."""
    for index in range(5):
        folder = root / f"location{index}"
        (folder / "bands").mkdir(parents=True)
        (folder / "labels").mkdir()
        raw = np.full((96, 96, 4), 2500 + index, dtype=np.uint16)
        np.save(folder / "bands" / "20220101.npy", raw)
        sites = [{"site_id": site, "prior_polygon": _ring(box(0, 0, 96, 96))}
                 for site in ("001", "002", "003")]
        positive = _ring(box(20, 20, 24, 24))
        if index == 0:
            if invalid:
                positive = [[20, 20], [24, 24], [20, 24], [24, 20], [20, 20]]
            elif empty:
                positive = []
            elif clipped:
                positive = _ring(box(90, 90, 100, 100))
        labels = [{"site_id": "001", "visible": 1, "polygon": positive},
                  {"site_id": "002", "visible": 0, "polygon": []}]
        (folder / "sites.json").write_text(json.dumps(sites), encoding="utf-8")
        (folder / "labels" / "20220101.json").write_text(json.dumps(labels), encoding="utf-8")




class HoldoutDataTests(unittest.TestCase):
    def test_raw_scale_validity_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            paths = sorted(path for path in root.rglob("*") if path.is_file())
            before = file_snapshot(paths, root)
            loaded = load_holdout_data(root, expected_invalid_gt=0)
            self.assertEqual(loaded["summary"]["samples"], 15)
            self.assertEqual(loaded["summary"]["positive"], 5)
            self.assertEqual(loaded["summary"]["missing"], 5)
            self.assertEqual(loaded["summary"]["minimum_roi_gt_fraction"], 1)
            self.assertEqual(loaded["duplicate_groups"], [])
            self.assertEqual(loaded["snapshot"], before)
            self.assertEqual(file_snapshot(loaded["input_files"], root.resolve()), before)
            first = loaded["samples"][0]
            self.assertEqual(first["input"][0, 0, 0], .25)
            self.assertEqual(first["key"], ("location0", "20220101", "001"))
            first["input"][:] = 0
            source = np.load(first["image_path"], allow_pickle=False)
            self.assertTrue((source == 2500).all())

    def test_invalid_positive_preserves_presence_and_requires_approved_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root, invalid=True)
            loaded = load_holdout_data(root, expected_invalid_gt=1)
            first = loaded["samples"][0]
            self.assertEqual(first["state"], "positive")
            self.assertEqual(first["presence"], 1)
            self.assertTrue(first["presence_valid"])
            self.assertFalse(first["segmentation_valid"].any())
            self.assertFalse(first["gt_geometry"].is_valid)
            self.assertTrue(first["segmentation_excluded"])
            self.assertEqual(loaded["summary"]["shape_evaluable"], 4)
            self.assertEqual(loaded["exclusions"][0]["reason"], "invalid_nonempty_positive_gt")
            with self.assertRaisesRegex(ValueError, "frozen count"):
                load_holdout_data(root, expected_invalid_gt=3)

    def test_segmentation_exclusion_is_explicit_and_does_not_relabel(self):
        raw = np.full((96, 96, 4), 2500, dtype=np.uint16)
        label = {"visible": 1, "polygon": [[1, 1], [4, 4], [1, 4], [4, 1], [1, 1]]}
        before = (raw.copy(), copy.deepcopy(label))
        with self.assertRaises(ValueError):
            prepare_sample(raw, box(0, 0, 96, 96), label)
        sample = prepare_sample(raw, box(0, 0, 96, 96), label,
                                segmentation_supervision=False, rotation=1, flip=True)
        self.assertEqual(sample["state"], "positive")
        self.assertTrue(sample["presence_valid"])
        self.assertEqual(sample["presence"], 1)
        self.assertFalse(sample["segmentation_valid"].any())
        np.testing.assert_array_equal(raw, before[0])
        self.assertEqual(label, before[1])

    def test_unapproved_empty_or_clipped_gt_stops_preflight(self):
        for option in ("empty", "clipped"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _fixture(root, **{option: True})
                with self.assertRaises(ValueError):
                    load_holdout_data(root, expected_invalid_gt=0)





    def test_fold_excludes_outer_and_rejects_connected_leakage(self):
        locations = [f'loc{i:02}' for i in range(12)]
        samples = [{'key': (loc, '20200101', 'site'), 'state': 'positive', 'presence_valid': True,
                    'gt_geometry': box(0, 0, 5, 5)} for loc in locations]
        parent = {'split_id': 'parent', 'seed': 42, 'train_locations': locations,
                  'validation_locations': ['outer'], 'connected_groups': {loc: [loc] for loc in locations},
                  'counts': {'train': _count_samples(samples)}}
        protocol = {'split_id': 'synthetic', 'seed': 42, 'parent_split_id': 'parent',
                    'validation_locations': [locations[i:i+4] for i in (0,4,8)],
                    'expected_validation_counts': [[4,4,4,0,0,4,4]] * 3, 'selection': 'fixture'}
        outer = {**samples[0], 'key': ('outer','20200101','site')}
        folds = make_folds(samples + [outer], parent, protocol)
        self.assertEqual([len(f['train']) for f in folds], [8,8,8])
        self.assertEqual(len({s['key'] for f in folds for s in f['validation']}), 12)
        parent['connected_groups']['loc00'] += parent['connected_groups'].pop('loc04')
        with self.assertRaisesRegex(ValueError, 'Connected-group'):
            make_folds(samples, parent, protocol)


if __name__ == "__main__":
    unittest.main()
