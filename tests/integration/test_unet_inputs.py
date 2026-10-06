"""Synthetic-only ROI, supervision and raster-to-polygon contract checks."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from shapely.geometry import box

from analysis.data import geometry, pixel_mask
from postprocessing.polygons import masks_to_polygons
from preprocessing.model_input import prepare_sample
from preprocessing.transforms import transform_array


class UnetInputTests(unittest.TestCase):
    def test_roi_origin_padding_scale_and_source_immutability(self):
        raw = np.full((20, 30, 4), 2500, dtype=np.uint16)
        prior = list(box(0, 0, 3, 5).exterior.coords)
        label = {"visible": 1, "polygon": list(box(1, 1, 2, 2).exterior.coords)}
        before = (raw.copy(), copy.deepcopy(prior), copy.deepcopy(label))
        sample = prepare_sample(raw, prior, label)
        self.assertEqual(sample["origin"], (-47, -46))
        self.assertEqual(sample["original_shape"], (20, 30))
        self.assertEqual(sample["input"].shape, (5, 96, 96))
        self.assertEqual(sample["input"].dtype, np.float32)
        self.assertEqual(sample["target"].shape, (1, 96, 96))
        self.assertEqual(sample["inside"].sum(), 600)
        self.assertEqual(sample["target"].sum(), 1)
        self.assertTrue(sample["target"][0, 47, 48])
        self.assertEqual(sample["input"][0, 46, 47], .25)
        self.assertTrue((sample["input"][:, ~sample["inside"]] == 0).all())
        self.assertFalse(sample["segmentation_valid"][0, 0, 0])
        sample["input"][:] = 99
        np.testing.assert_array_equal(raw, before[0])
        self.assertEqual(prior, before[1]); self.assertEqual(label, before[2])

    def test_official_numeric_and_padding_masks_stay_separate(self):
        raw = np.full((96, 96, 4), 5000, dtype=np.float32)
        raw[0, 0, 0] = 0
        raw[0, 1, 3] = 0
        raw[0, 2, 1:3] = 0
        raw[0, 3, 1] = np.nan
        raw[0, 4, 0] = np.inf
        sample = prepare_sample(raw, box(0, 0, 96, 96), {"visible": 0, "polygon": []})
        np.testing.assert_array_equal(sample["official_valid"][0, :5], [False, False, True, True, True])
        np.testing.assert_array_equal(sample["numeric_valid"][0, :5], [True, True, True, False, False])
        np.testing.assert_array_equal(sample["segmentation_valid"][0, 0, :5], [False, False, True, False, False])
        self.assertTrue(sample["inside"].all())
        self.assertTrue(np.isfinite(sample["input"]).all())
        self.assertEqual(sample["input"][0, 0, 2], .5)
        self.assertTrue((sample["input"][:4, 0, [0, 1, 3, 4]] == 0).all())
        self.assertTrue((sample["input"][4, 0, :5] == 1).all())

    def test_d4_transfers_masks_and_channels_without_rerasterization(self):
        rng = np.random.default_rng(42)
        raw = rng.integers(1, 10000, size=(96, 96, 4), dtype=np.uint16)
        raw[3, 4, 0] = 0
        prior = box(20, 21, 76, 75)
        label = {"visible": 1, "polygon": list(box(23, 27, 31, 38).exterior.coords)}
        baseline = prepare_sample(raw, prior, label)
        for rotation in range(4):
            for flip in (False, True):
                transformed = prepare_sample(raw, prior, label, rotation=rotation, flip=flip)
                for name in ("prior", "gt", "official_valid", "numeric_valid", "inside"):
                    np.testing.assert_array_equal(transformed[name], transform_array(baseline[name], rotation, flip))
                for name in ("input", "target", "segmentation_valid"):
                    expected = transform_array(baseline[name].transpose(1, 2, 0), rotation, flip).transpose(2, 0, 1)
                    np.testing.assert_array_equal(transformed[name], expected)
                    restored = transform_array(transformed[name].transpose(1, 2, 0), 0, flip)
                    restored = transform_array(restored, (-rotation) % 4).transpose(2, 0, 1)
                    np.testing.assert_array_equal(restored, baseline[name])
                self.assertEqual(transformed["origin"], baseline["origin"])

    def test_small_positive_missing_ignore_and_negative_are_distinct(self):
        raw = np.full((96, 96, 4), 1000, dtype=np.uint16)
        prior = box(0, 0, 96, 96)
        positive = prepare_sample(raw, prior, {"visible": 1, "polygon": list(box(1, 1, 2, 2).exterior.coords)})
        self.assertEqual(positive["presence"], 1)
        self.assertEqual(positive["target"].sum(), 1)
        self.assertTrue(positive["presence_valid"])
        plural = prepare_sample(raw, prior, {"visible": 1, "polygons": [[list(box(1, 1, 2, 2).exterior.coords)]]})
        np.testing.assert_array_equal(plural["target"], positive["target"])
        for label, state in ((None, "missing"), ({"visible": 1, "ignore": True, "polygon": []}, "ignore")):
            sample = prepare_sample(raw, prior, label)
            self.assertEqual(sample["state"], state)
            self.assertFalse(sample["presence_valid"])
            self.assertFalse(sample["segmentation_valid"].any())
        negative = prepare_sample(raw, prior, {"visible": 0, "polygon": []})
        self.assertEqual(negative["state"], "negative")
        self.assertTrue(negative["presence_valid"])
        self.assertTrue(negative["segmentation_valid"].all())
        self.assertFalse(negative["target"].any())

    def test_positive_geometry_and_supervision_fail_explicitly(self):
        raw = np.full((96, 96, 4), 1000, dtype=np.uint16)
        prior = box(0, 0, 96, 96)
        invalid = [[1, 1], [3, 3], [1, 3], [3, 1], [1, 1]]
        for polygon in ([], invalid, list(box(.01, .01, .2, .2).exterior.coords),
                        list(box(100, 100, 102, 102).exterior.coords)):
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                prepare_sample(raw, prior, {"visible": 1, "polygon": polygon})
        positive = {"visible": 1, "polygon": list(box(1, 1, 2, 2).exterior.coords)}
        raw[1, 1, 0] = 0
        with self.assertRaisesRegex(ValueError, "no supervised raster pixel"):
            prepare_sample(raw, prior, positive)
        for bad_prior in ([], invalid):
            with self.assertRaises(ValueError):
                prepare_sample(raw, bad_prior)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            prepare_sample(raw, prior, {**positive, "visible": 0})

    def test_polygon_cells_restore_holes_multipart_and_one_pixel(self):
        mask = np.zeros((8, 9), dtype=bool)
        mask[1:6, 1:6] = True
        mask[2:5, 2:5] = False
        mask[7, 8] = True
        before = mask.copy()
        result = masks_to_polygons(mask, (10, 20), (40, 40))
        polygon = geometry(result)
        self.assertTrue(polygon.is_valid)
        self.assertEqual(len(polygon.geoms), 2)
        self.assertEqual(sum(len(part.interiors) for part in polygon.geoms), 1)
        self.assertEqual(polygon.area, 17)
        self.assertTrue(any(part.area == 1 for part in polygon.geoms))
        np.testing.assert_array_equal(pixel_mask(polygon, (40, 40))[20:28, 10:19], mask)
        np.testing.assert_array_equal(mask, before)

    def test_polygon_clips_padding_without_area_filter_or_fallback(self):
        mask = np.ones((4, 4), dtype=bool)
        polygon = geometry(masks_to_polygons(mask, (-3, -3), (3, 3)))
        self.assertEqual(polygon.area, 1)
        self.assertEqual(polygon.bounds, (0, 0, 1, 1))
        self.assertEqual(masks_to_polygons(mask, (8, 8), (3, 3)), [])
        self.assertEqual(masks_to_polygons(np.zeros((4, 4), bool), (0, 0), (4, 4)), [])
        with self.assertRaises(ValueError):
            masks_to_polygons(np.array([[np.nan]]), (0, 0), (4, 4))


if __name__ == "__main__":
    unittest.main()
