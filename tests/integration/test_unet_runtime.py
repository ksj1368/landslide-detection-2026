"""Actual U-Net CPU/MPS checks on generated inputs; never a performance score."""

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from evaluation.contracts import validate_oof, validate_partition
from models.unet import SmallUNet
from training.synthetic import fixture_samples
from training.unet import (CONFIG, evaluate_loss, load_checkpoint, make_batch, save_checkpoint,
                           set_seed, supervised_loss, train_single_split, train_step)


class UnetCPU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        set_seed()
        self.train, self.validation = fixture_samples()

    def check_device(self, device):
        model = SmallUNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = make_batch(self.train, device)
        before = [p.detach().clone() for p in model.parameters()]
        metrics = train_step(model, optimizer, batch)
        self.assertGreater(metrics["loss"], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        for parameter in model.parameters():
            self.assertEqual(parameter.device.type, device)
            self.assertEqual(parameter.grad.device.type, device)
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        model.eval()
        with torch.no_grad():
            expected = model(batch["input"])
        self.assertEqual(tuple(expected["segmentation"].shape), (4, 1, 96, 96))
        self.assertEqual(tuple(expected["presence"].shape), (4,))
        with tempfile.TemporaryDirectory(prefix="unet-checkpoint-") as directory:
            path = Path(directory) / "synthetic.pt"
            save_checkpoint(path, model, optimizer, 1)
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, optimizer, 2)
            restored, saved = load_checkpoint(path, device)
            self.assertEqual(saved["config"], CONFIG)
            self.assertEqual(saved["updates"], 1)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
            restored_optimizer.load_state_dict(saved["optimizer"])
            cpu_restored, _ = load_checkpoint(path, "cpu")
            with torch.no_grad():
                for key, value in restored(batch["input"]).items():
                    torch.testing.assert_close(value, expected[key], rtol=1e-4, atol=1e-5)
                for key, value in cpu_restored(batch["input"].cpu()).items():
                    torch.testing.assert_close(value, expected[key].cpu(), rtol=1e-4, atol=1e-5)
            train_step(model, optimizer, batch)
            train_step(restored, restored_optimizer, batch)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        return model

    def test_cpu_forward_updates_and_restore(self):
        self.check_device("cpu")

    def test_loss_ignores_unsupervised_targets(self):
        model = SmallUNet()
        batch = make_batch(self.train)
        outputs = model(batch["input"])
        expected = supervised_loss(outputs, batch)[0]
        changed = {k: value.clone() for k, value in batch.items()}
        changed["target"][~changed["segmentation_valid"]] = float("nan")
        changed["presence"][~changed["presence_valid"]] = float("nan")
        torch.testing.assert_close(supervised_loss(outputs, changed)[0], expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "no supervision"):
            ignored = make_batch(self.train[2:])
            supervised_loss(model(ignored["input"]), ignored)

    def test_one_split_validation_and_no_stat_fitting(self):
        model, _, history, result = train_single_split(self.train, self.validation, updates=2, batch_size=2)
        self.assertEqual(len(history), 2)
        self.assertEqual(result["presence_samples"], 2)
        self.assertEqual(CONFIG["normalization"], "none")
        self.assertNotIn("statistics", CONFIG)
        unchanged = [p.detach().clone() for p in model.parameters()]
        small_batches = evaluate_loss(model, self.validation, batch_size=1)
        self.assertAlmostEqual(result["loss"], small_batches["loss"], places=5)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(unchanged, model.parameters())))

    def test_partition_and_oof_checks(self):
        validate_partition(self.train, self.validation)
        leaked = [dict(s, group=self.train[0]["group"]) for s in self.validation]
        with self.assertRaisesRegex(ValueError, "Connected-group"):
            validate_partition(self.train, leaked)
        with self.assertRaises(ValueError):
            validate_partition(self.train, self.train)
        same_location = [dict(s, key=(self.train[0]["key"][0], "20220202", s["key"][2])) for s in self.validation]
        with self.assertRaisesRegex(ValueError, "Location"):
            validate_partition(self.train, same_location)
        expected = [s["key"] for s in self.validation]
        validate_oof(self.validation, expected)
        for bad in (self.validation[:-1], self.validation + self.validation[:1], self.train):
            with self.assertRaises(ValueError):
                validate_oof(bad, expected)


@unittest.skipUnless(os.environ.get('LANDSLIDE_TEST_MPS') == '1', 'Set LANDSLIDE_TEST_MPS=1 to opt into MPS checks')
class UnetMPS(UnetCPU):
    # This class deliberately runs only GPU checks; CPU checks above run once.
    test_cpu_forward_updates_and_restore = None
    test_loss_ignores_unsupervised_targets = None
    test_one_split_validation_and_no_stat_fitting = None
    test_partition_and_oof_checks = None

    def setUp(self):
        super().setUp()
        self.assertTrue(torch.backends.mps.is_available(), "MPS unavailable; cannot complete GPU verification")
        self.assertEqual(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
        self.assertEqual(torch.__version__.split("+")[0], "2.14.0")

    def test_mps_forward_updates_and_restore(self):
        self.check_device("mps")
        torch.mps.synchronize()

    def test_mps_cpu_identical_initial_weights(self):
        model = SmallUNet().eval()
        mps_model = copy.deepcopy(model).to("mps")
        inputs = make_batch(self.train)["input"]
        with torch.no_grad():
            expected = model(inputs)
            observed = mps_model(inputs.to("mps"))
        for key in expected:
            self.assertEqual(observed[key].device.type, "mps")
            torch.testing.assert_close(observed[key].cpu(), expected[key], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
