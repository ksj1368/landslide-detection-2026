"""Synthetic boundary comparisons; never open real imagery or stored probabilities."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

from inference.unet import prediction_from_probabilities
from postprocessing.boundary_processing import boundary_prediction


def fixture(size=11, *, origin=(0, 0), original_shape=(40, 40)):
    valid = np.ones((size, size), dtype=bool)
    return {"inside": valid.copy(), "official_valid": valid.copy(),
            "numeric_valid": valid.copy(), "origin": origin,
            "original_shape": original_shape}, np.zeros((size, size))


def geometry(prediction):
    return MultiPolygon([Polygon(rings[0], rings[1:]) for rings in prediction["polygons"]])


def contour(sample, mask, head=.8, **kwargs):
    return boundary_prediction(sample, mask, head, method="probability_contour", **kwargs)


class BoundaryProcessingTests(unittest.TestCase):
    def test_single_pixel_interpolates_centers_and_preserves_inputs(self):
        sample, mask = fixture(size=3)
        mask[1, 1] = 1
        before = mask.copy()
        predicted = geometry(contour(sample, mask))
        self.assertTrue(predicted.is_valid)
        np.testing.assert_allclose(predicted.bounds, (.9, .9, 2.1, 2.1))
        self.assertAlmostEqual(predicted.area, .72)
        self.assertEqual(len(predicted.geoms), 1)
        np.testing.assert_array_equal(mask, before)

    def test_low_diagonal_connectivity_keeps_high_pixels_separate(self):
        sample, mask = fixture(size=4)
        mask[1, 1] = mask[2, 2] = 1
        predicted = geometry(contour(sample, mask))
        self.assertTrue(predicted.is_valid)
        self.assertEqual(len(predicted.geoms), 2)

    def test_nested_hole_and_island_are_preserved(self):
        sample, mask = fixture()
        mask[1:10, 1:10] = 1
        mask[3:8, 3:8] = 0
        mask[5, 5] = 1
        predicted = geometry(contour(sample, mask))
        self.assertTrue(predicted.is_valid)
        self.assertEqual(len(predicted.geoms), 2)
        self.assertEqual(sorted(len(part.interiors) for part in predicted.geoms), [0, 1])
        self.assertTrue(predicted.covers(box(5.3, 5.3, 5.7, 5.7)))
        self.assertEqual(predicted.intersection(box(3.5, 3.5, 4.5, 4.5)).area, 0)

    def test_nonzero_origin_padding_image_clipping_and_usable_exclusion(self):
        sample, mask = fixture(size=3, origin=(7, 9), original_shape=(12, 10))
        mask[0, 0] = 1
        shifted = geometry(contour(sample, mask))
        np.testing.assert_allclose(shifted.bounds, (7, 9, 8.1, 10.1))
        sample, mask = fixture(size=3, origin=(-1, -1), original_shape=(2, 2))
        mask[:] = 1
        for field in ("inside", "official_valid", "numeric_valid"):
            sample[field][1, 1] = False
        predicted = geometry(contour(sample, mask))
        self.assertTrue(box(0, 0, 2, 2).covers(predicted))
        self.assertEqual(predicted.intersection(box(0, 0, 1, 1)).area, 0)
        self.assertGreater(predicted.area, 0)

    def test_head_gate_empty_and_exact_mask_threshold_distinguish_presence(self):
        sample, mask = fixture(size=3)
        mask[1, 1] = 1
        self.assertFalse(contour(sample, mask, np.nextafter(.2, 0))["presence_positive"])
        self.assertTrue(contour(sample, mask, .2)["polygons"])
        mask[:] = 0
        self.assertTrue(contour(sample, mask)["positive_without_polygon"])
        mask[1, 1] = .4
        exact = contour(sample, mask)
        self.assertTrue(exact["presence_positive"])
        self.assertTrue(exact["positive_without_polygon"])
        # Existing >= pixel classification and an exactly level-only contour
        # have intentionally different zero-area semantics.
        self.assertTrue(prediction_from_probabilities(sample, mask, .8,
                                                     presence_threshold=.2, mask_threshold=.4)["polygons"])

    def test_simplification_matches_native_topology_preserving_operation(self):
        sample, mask = fixture()
        mask[1:10, 1:10] = 1
        mask[3:8, 3:8] = 0
        mask[5, 5] = 1
        mask[1, 1] = mask[9, 9] = 0
        baseline = geometry(prediction_from_probabilities(sample, mask, .8,
                                                          presence_threshold=.2, mask_threshold=.4))
        for tolerance in (.25, .5, .75):
            with self.subTest(tolerance=tolerance):
                result = boundary_prediction(sample, mask, .8, method="simplify", tolerance=tolerance)
                actual = geometry(result)
                self.assertTrue(actual.equals(baseline.simplify(tolerance, preserve_topology=True)))
                self.assertEqual(len(actual.geoms), 2)
                self.assertEqual(sorted(len(part.interiors) for part in actual.geoms), [0, 1])
                self.assertEqual(result["contour_degenerate_ring_count"], 0)
        self.assertEqual(boundary_prediction(sample, mask, .1, method="simplify", tolerance=.5)["polygons"], [])

    def test_degenerate_contours_are_counted_and_invalid_contours_fail(self):
        sample, mask = fixture(size=3)
        with patch("postprocessing.boundary_processing.find_contours", return_value=[
            np.array([[1, 1], [1, 2], [1, 3], [1, 1]])
        ]):
            result = contour(sample, mask)
            self.assertEqual(result["contour_degenerate_ring_count"], 1)
            self.assertTrue(result["positive_without_polygon"])
        for ring, message in (
            ([[1, 1], [2, 2], [1, 2], [2, 1], [1, 1]], "Invalid probability contour"),
            ([[1, 1], [1, 2], [2, 2], [2, 1]], "Open probability contour"),
        ):
            with self.subTest(message=message), patch(
                "postprocessing.boundary_processing.find_contours", return_value=[np.array(ring)]
            ), self.assertRaisesRegex(ValueError, message):
                contour(sample, mask)

    def test_invalid_inputs_and_invalid_baseline_geometry_fail_without_fallback(self):
        sample, mask = fixture(size=3)
        for value in (np.nan, np.inf, -.01, 1.01):
            bad = mask.copy()
            bad[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                contour(sample, bad)
        for kwargs in ({"method": "unknown"}, {"method": "simplify", "tolerance": -1},
                       {"method": "probability_contour", "tolerance": .5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                boundary_prediction(sample, mask, .8, **kwargs)
        with self.assertRaises(ValueError):
            contour(sample, mask, mask_threshold=np.nan)
        sample["inside"] = np.ones((2, 2), dtype=bool)
        with self.assertRaises(ValueError):
            contour(sample, mask)
        sample, mask = fixture(size=3)
        bad_prediction = {"polygons": [[[[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]]],
                          "presence_probability": .8, "presence_positive": True,
                          "positive_without_polygon": False}
        with patch("postprocessing.boundary_processing.prediction_from_probabilities",
                   return_value=bad_prediction), self.assertRaisesRegex(ValueError, "Invalid boundary geometry"):
            boundary_prediction(sample, mask, .8, method="simplify", tolerance=.5)


if __name__ == "__main__":
    unittest.main()
