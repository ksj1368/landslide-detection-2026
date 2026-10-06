"""Synthetic CSV and copied-source inference checks; never read real data."""

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import torch
from shapely.geometry import MultiPolygon, Polygon

from inference.unet import CSV_FIELDS, run_from_environment, run_inference
from models.unet import SmallUNet
from training.unet import save_checkpoint, set_seed


def _checkpoint(path, *, presence=8.0, segmentation=8.0):
    set_seed()
    model = SmallUNet()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.presence.bias.fill_(presence)
        model.segmentation.bias.fill_(segmentation)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(path, model, optimizer, updates=0)
    return path


def _fixture(root):
    prior = [[[24, 24], [40, 24], [40, 40], [24, 40], [24, 24]]]
    edge = [[[0, 0], [8, 0], [8, 8], [0, 8], [0, 0]]]
    for name in ("synthetic_a", "synthetic_b"):
        folder = root / name
        (folder / "bands").mkdir(parents=True)
        (folder / "sites.json").write_text(json.dumps([
            {"site_id": "001", "prior_polygon": prior},
            {"site_id": 7, "prior_polygon": edge},
        ]), encoding="utf-8")
        raw = np.full((64, 64, 4), 1000, dtype=np.uint16)
        raw[10, 10, 0] = 0
        np.save(folder / "bands/20220603.npy", raw, allow_pickle=False)
        np.save(folder / "bands/20220604.npy", np.zeros_like(raw), allow_pickle=False)
        (folder / "labels").mkdir()
        (folder / "labels/20220603.json").write_text("FORBIDDEN: NOT JSON", encoding="utf-8")
        (folder / "images").mkdir()
        (folder / "images/20220603.npy").write_bytes(b"FORBIDDEN: NOT NPY")
    return root


class UNetInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_complete_csv_uses_only_bands_sites_and_excludes_invalid_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _fixture(root / "inputs")
            checkpoint = _checkpoint(root / "checkpoint.pt")
            original_read, original_load = Path.read_text, np.load
            reads = []

            def only_sites(path, *args, **kwargs):
                self.assertEqual(path.name, "sites.json")
                reads.append(path)
                return original_read(path, *args, **kwargs)

            def only_bands(path, *args, **kwargs):
                self.assertEqual(Path(path).parent.name, "bands")
                return original_load(path, *args, **kwargs)

            with patch.object(Path, "read_text", only_sites), patch("inference.unet.np.load", only_bands):
                summary = run_inference(inputs, root / "predictions.csv", checkpoint)
            self.assertEqual(len(reads), 2)
            self.assertEqual(summary["device"], "cpu")
            self.assertEqual((summary["rows"], summary["locations"], summary["scenes"]), (8, 2, 4))
            self.assertEqual(summary["presence_positive"], 8)
            self.assertEqual(summary["positive_predictions"], 4)
            self.assertEqual(summary["positive_without_polygon"], 4)
            with (root / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames), CSV_FIELDS)
                rows = list(reader)
            self.assertEqual(len({tuple(row[key] for key in CSV_FIELDS[:3]) for row in rows}), 8)
            self.assertEqual({row["site_id"] for row in rows}, {"001", "7"})
            for row in rows:
                if row["date"] == "20220604":
                    self.assertEqual(row["polygons"], "")
                    continue
                coordinates = json.loads(row["polygons"])
                geometry = MultiPolygon([Polygon(rings[0], rings[1:]) for rings in coordinates])
                self.assertTrue(geometry.is_valid)
                self.assertEqual(geometry.bounds, (0.0, 0.0, 64.0, 64.0)
                                 if row["site_id"] == "001" else (0.0, 0.0, 52.0, 52.0))
                self.assertEqual(geometry.area, 64 * 64 - 1 if row["site_id"] == "001" else 52 * 52 - 1)
            self.assertTrue(all(value >= 0 for value in summary["timings_seconds"].values()))

    def test_negative_presence_and_empty_mask_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _fixture(root / "inputs")
            for name, presence, segmentation, expected_empty in (
                ("negative", -8, 8, 0), ("empty_mask", 8, -8, 8)
            ):
                checkpoint = _checkpoint(root / f"{name}.pt", presence=presence, segmentation=segmentation)
                summary = run_inference(inputs, root / f"{name}.csv", checkpoint)
                self.assertEqual(summary["empty_predictions"], 8)
                self.assertEqual(summary["positive_without_polygon"], expected_empty)

    def test_invalid_inputs_do_not_publish_partial_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _fixture(root / "inputs")
            checkpoint = _checkpoint(root / "checkpoint.pt")
            output = root / "predictions.csv"
            output.write_text("existing complete output", encoding="utf-8")
            for protected in (inputs / "synthetic_a/sites.json", checkpoint):
                with self.assertRaisesRegex(ValueError, "separate from inputs and checkpoint"):
                    run_inference(inputs, protected, checkpoint)
            # A late unreadable scene must not replace an earlier complete output.
            (inputs / "synthetic_b/bands/20220604.npy").write_bytes(b"invalid NPY")
            with self.assertRaises(ValueError):
                run_inference(inputs, output, checkpoint)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing complete output")
            sites = inputs / "synthetic_a/sites.json"
            repeated = json.loads(sites.read_text(encoding="utf-8"))[:1] * 2
            sites.write_text(json.dumps(repeated), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate site"):
                run_inference(inputs, output, checkpoint)
            with self.assertRaises(ValueError):
                run_inference(root / "missing", output, checkpoint)


    def test_environment_requires_official_path_variables(self):
        with patch.dict(os.environ, {"AIF_INPUT_DIR": "", "AIF_PREDICTION_PATH": ""}):
            with self.assertRaisesRegex(ValueError, "AIF_INPUT_DIR and AIF_PREDICTION_PATH"):
                run_from_environment()


if __name__ == "__main__":
    unittest.main()
