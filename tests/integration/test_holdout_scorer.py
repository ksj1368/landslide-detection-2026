"""Manual synthetic checks for the document-based scorer, never official parity."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shapely.geometry import MultiPolygon, Polygon, box, mapping

from evaluation.scorer import score_predictions


def sample(site, state="positive", gt=None, *, date="20220101", location="L"):
    return {"key": (location, date, str(site)), "state": state, "gt_geometry": gt}


def prediction(row, geom=None, probability=None):
    coords = [] if geom is None else mapping(MultiPolygon([geom]) if isinstance(geom, Polygon) else geom)["coordinates"]
    return {"key": row["key"], "polygons": coords, "presence_probability": probability}


class HoldoutScorerTests(unittest.TestCase):
    def test_perfect_including_exact_ten_pixel_boundary(self):
        positive = sample(1, gt=box(0, 0, 5, 2))
        negative = sample(2, "negative")
        result = score_predictions([positive, negative], [prediction(positive, positive["gt_geometry"]), prediction(negative)])
        summary = result["summary"]
        self.assertEqual((summary["presence"], summary["shape"], summary["total"]), (1., 1., 1.))
        self.assertEqual(summary["confusion"], dict(tp=1, tn=1, fp=0, fn=0))
        self.assertEqual(summary["scorer_id"], "scorer-doc-v1")
        self.assertEqual(summary["official_compatibility"], "not_verified")
        self.assertEqual(summary["shape_eligible_count"], 1)
        self.assertNotIn("coordinates", json.dumps(result, allow_nan=False))

    def test_presence_uses_serialized_polygon_instead_of_head(self):
        positive = sample(1, gt=box(0, 0, 4, 4))
        negative = sample(2, "negative")
        result = score_predictions([positive, negative], [prediction(positive, probability=.99), prediction(negative, box(10, 10, 11, 11), .01)])
        summary = result["summary"]
        self.assertEqual((summary["presence"], summary["shape"], summary["total"]), (0., 0., 0.))
        self.assertEqual(summary["confusion"], dict(tp=0, tn=0, fp=1, fn=1))
        self.assertEqual(summary["head_positive_without_polygon_count"], 1)

    def test_empty_predictions_have_manual_macro_f1(self):
        rows = [sample(1, gt=box(0, 0, 4, 4)), sample(2, "negative")]
        summary = score_predictions(rows, [prediction(row) for row in rows])["summary"]
        self.assertAlmostEqual(summary["presence"], 1 / 3)
        self.assertEqual(summary["shape"], 0)
        self.assertAlmostEqual(summary["total"], .2)

    def test_small_positive_keeps_presence_missing_ignore_excluded(self):
        tiny = sample(1, gt=box(0, 0, 1, 1))
        rows = [tiny, sample(2, "negative"), sample(3, "missing"), sample(4, "ignore")]
        preds = [prediction(tiny, tiny["gt_geometry"])] + [prediction(row) for row in rows[1:]]
        result = score_predictions(rows, preds)
        summary = result["summary"]
        self.assertEqual(summary["presence"], 1)
        self.assertIsNone(summary["shape"])
        self.assertIsNone(summary["total"])
        self.assertEqual(summary["shape_reason"], "no_shape_eligible_samples")
        self.assertEqual(summary["small_positive_count"], 1)
        self.assertEqual(summary["labeled_count"], 2)
        self.assertEqual((summary["missing_count"], summary["ignored_count"]), (1, 1))
        self.assertIsNone(result["rows"][2]["gt_present"])

    def test_invalid_gt_keeps_presence_and_exposes_shape_subset(self):
        invalid = Polygon([(0, 0), (4, 4), (4, 0), (0, 4), (0, 0)])
        bad = sample(1, gt=invalid)
        good = sample(2, gt=box(0, 0, 4, 4))
        negative = sample(3, "negative")
        before = invalid.wkb
        result = score_predictions([bad, good, negative], [prediction(bad, box(0, 0, 1, 1)), prediction(good, good["gt_geometry"]), prediction(negative)])
        summary = result["summary"]
        self.assertEqual(summary["presence"], 1)
        self.assertIsNone(summary["shape"])
        self.assertIsNone(summary["total"])
        self.assertEqual(summary["shape_diagnostic"], 1)
        self.assertEqual(summary["shape_reason"], "invalid_positive_gt")
        self.assertEqual(summary["invalid_gt_count"], 1)
        self.assertEqual(summary["shape_eligible_count"], 1)
        self.assertEqual(summary["shape_site_count"], 1)
        self.assertEqual(before, invalid.wkb)

    def test_unexpected_empty_positive_never_becomes_negative(self):
        rows = [sample(1, gt=Polygon()), sample(2, "negative")]
        result = score_predictions(rows, [prediction(row) for row in rows])
        self.assertEqual(result["summary"]["positive_count"], 1)
        self.assertEqual(result["summary"]["confusion"]["fn"], 1)
        self.assertEqual(result["summary"]["empty_or_nonpolygon_gt_count"], 1)
        self.assertIsNone(result["summary"]["shape"])

    def test_holes_multipart_continuous_coordinates(self):
        hole = Polygon([(0.25, .5), (10.25, .5), (10.25, 10.5), (.25, 10.5)], [[(2.25, 2.5), (4.25, 2.5), (4.25, 4.5), (2.25, 4.5)]])
        gt = MultiPolygon([hole, box(20.25, .5, 22.25, 2.5)])
        positive = sample(1, gt=gt)
        negative = sample(2, "negative")
        preds = [prediction(positive, MultiPolygon([box(.25, .5, 10.25, 10.5), box(20.25, .5, 22.25, 2.5)])), prediction(negative)]
        result = score_predictions([positive, negative], preds)
        self.assertAlmostEqual(result["summary"]["shape"], 100 / 104)
        exact = score_predictions([positive, negative], [prediction(positive, gt), prediction(negative)])
        self.assertEqual(exact["summary"]["shape"], 1)

    def test_site_weighting_differs_from_sample_weighting(self):
        gt = box(0, 0, 4, 4)
        rows = [sample(1, gt=gt, date=f"2022010{i}") for i in (1, 2, 3)]
        rows += [sample(1, gt=gt, location="other"), sample(2, "negative")]
        preds = [prediction(row, gt) for row in rows[:3]] + [prediction(row) for row in rows[3:]]
        summary = score_predictions(rows, preds)["summary"]
        self.assertEqual(summary["shape"], .5)
        self.assertEqual(summary["shape_site_count"], 2)
        self.assertEqual(summary["shape_eligible_count"], 4)

    def test_duplicate_missing_unexpected_keys_fail(self):
        rows = [sample(1), sample(2, "negative")]
        preds = [prediction(row) for row in rows]
        cases = [(rows + rows[:1], preds), (rows, preds + preds[:1]), (rows, preds[:1]), (rows[:1], preds)]
        for samples, predictions in cases:
            with self.subTest(sample_count=len(samples), prediction_count=len(predictions)):
                with self.assertRaises(ValueError):
                    score_predictions(samples, predictions)

    def test_empty_and_single_class_are_diagnostic_incomplete(self):
        empty = score_predictions([], [])["summary"]
        self.assertEqual(empty["presence_reason"], "no_labeled_samples")
        self.assertIsNone(empty["total"])
        positive = sample(1, gt=box(0, 0, 4, 4))
        summary = score_predictions([positive], [prediction(positive, positive["gt_geometry"])])["summary"]
        self.assertIsNone(summary["presence"])
        self.assertEqual(summary["presence_reason"], "single_gt_class")
        self.assertEqual(summary["shape"], 1)
        self.assertEqual(summary["status"], "diagnostic_incomplete")

    def test_invalid_prediction_is_rejected_without_geometry_repair(self):
        positive = sample(1, gt=box(0, 0, 4, 4))
        invalid = Polygon([(0, 0), (4, 4), (4, 0), (0, 4), (0, 0)])
        for coords in (prediction(positive, invalid)["polygons"], [[[]]], None):
            with self.subTest(coords_type=type(coords).__name__):
                with self.assertRaises(ValueError):
                    score_predictions([positive], [{"key": positive["key"], "polygons": coords}])


if __name__ == "__main__":
    unittest.main()
