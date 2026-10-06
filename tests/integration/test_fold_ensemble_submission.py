"""Member-specific normalization, exact probability mean and isolated export."""
from pathlib import Path
import tempfile
import json
import os
import subprocess
import sys
import unittest

import numpy as np
import torch

from inference.unet import predict_fold_probabilities, predict_probabilities
from models.checkpoint import INFERENCE_VALUES, validate_inference_config
from models.fold_ensemble import CoverageFoldEnsemble
from preprocessing.model_input import index_configuration, prepare_sample


class FoldEnsembleTests(unittest.TestCase):

    def test_retrained_train_assemble_and_cli(self):
        from scripts.prepare_coverage_submission import assemble
        from tests.integration.test_d4_selection import fixture
        from tests.integration.test_unet_inference import _fixture
        from training.ab_supervision import attach_coverage
        from training.checkpoint_selection_experiment import train_trajectory
        from training.unet import COVERAGE_BCE_DICE
        from models.checkpoint import load_checkpoint
        project = Path(__file__).resolve().parents[2]
        public = project / 'configs/inference_s030.json'
        train, selection, _, config = fixture()
        config.update(updates=1, checkpoint_selection='bce_and_d4_reference')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {'segmentation': [], 'presence': []}
            for fold in (1, 2, 3):
                for role in paths:
                    data, candidate = train, {**config, 'fold_id': fold}
                    if role == 'segmentation':
                        data, _ = attach_coverage(train)
                        candidate.update(experiment_id='E090', loss=COVERAGE_BCE_DICE)
                    run = root / f'{role}{fold}'
                    train_trajectory(data, selection, candidate, run)
                    paths[role].append(run / 'best_d4.pt')
            output = root / 'retrained.pt'
            with self.assertRaisesRegex(ValueError, 'Selected updates'):
                assemble(paths['segmentation'], paths['presence'], output, public)
            with self.assertRaisesRegex(ValueError, 'ordered folds'):
                assemble(paths['segmentation'][::-1], paths['presence'], output, public, retrained=True)
            result = assemble(paths['segmentation'], paths['presence'], output, public, retrained=True)
            _, saved = load_checkpoint(output)
            self.assertTrue(saved['metadata']['retrained'])
            self.assertEqual(saved['config']['experiment_id'], 'retrained_role_ensemble')
            self.assertEqual(saved['metadata']['segmentation_updates'], [1, 1, 1])
            # A pre-existing sidecar must survive without creating a checkpoint.
            collision = root / 'collision.pt'
            collision.with_suffix('.json').write_text('preserve')
            with self.assertRaises(FileExistsError):
                assemble(paths['segmentation'], paths['presence'], collision, public, retrained=True)
            self.assertFalse(collision.exists())
            inputs = _fixture(root / 'input')
            env = {k:v for k,v in os.environ.items() if k != 'PYTHONPATH'}
            command = [sys.executable, '-I', '-B', str(project / 'scripts/run_inference.py'),
                       '--input', str(inputs), '--output', str(root / 'predictions.csv'),
                       '--checkpoint', str(output), '--config', result['inference_config'], '--device', 'cpu']
            completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((root / 'predictions.csv').read_text().startswith('location_id,date,site_id,polygons'))

    def test_baseline_presence_ensemble_and_payload(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = CoverageFoldEnsemble(baseline_presence=True).eval()
        stats = [{'bands': ['R', 'G', 'B', 'NIR'], 'mean': [.03 * i] * 4,
                  'std': [.1 + i * .02] * 4} for i in (1, 2, 3)]
        config = {**INFERENCE_VALUES, **index_configuration('NDVI'), 'model': model.model_id,
                  'roi_size': 96, 'origin_policy': 'shift_inside', 'member_normalization_statistics': stats}
        raw = np.random.default_rng(42).integers(1, 4000, (64, 64, 4), dtype=np.uint16)
        raw[10:15, 10:15, 0] = 0
        prior = [[[0, 0], [8, 0], [8, 8], [0, 8], [0, 0]]]
        sample = prepare_sample(raw, prior, input_index='NDVI', origin_policy='shift_inside')
        masks, heads, _ = predict_fold_probabilities(model, sample, config, tta='d4')
        expected_masks, expected_heads = [], []
        for member, normalization in zip(model.members, stats, strict=True):
            direct = prepare_sample(raw, prior, input_index='NDVI', origin_policy='shift_inside',
                                    normalization='zscore', normalization_statistics=normalization)
            tensor = torch.from_numpy(direct['input'])[None]
            expected_masks.append(predict_probabilities(member.candidate_model, tensor, tta='d4')[0])
            expected_heads.append(predict_probabilities(member.baseline_model, tensor, tta='d4')[1])
        np.testing.assert_array_equal(masks, np.mean(expected_masks, axis=0))
        np.testing.assert_array_equal(heads, np.mean(expected_heads, axis=0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'ensemble.pt'
            torch.save({'model': model.state_dict(), 'config': config, 'seed': 42, 'updates': 0}, checkpoint)
            from models.checkpoint import load_checkpoint
            restored, _ = load_checkpoint(checkpoint)
            actual_masks, actual_heads, _ = predict_fold_probabilities(restored, sample, config, tta='d4')
            np.testing.assert_array_equal(actual_masks, masks)
            np.testing.assert_array_equal(actual_heads, heads)


    def test_normalization_mean_and_payload(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = CoverageFoldEnsemble().eval()
        stats = [{'bands': ['R', 'G', 'B', 'NIR'], 'mean': [i * .03] * 4,
                  'std': [.1 + i * .02] * 4} for i in range(1, 4)]
        config = {**INFERENCE_VALUES, **index_configuration('NDVI'), 'model': model.model_id,
                  'roi_size': 96, 'origin_policy': 'shift_inside',
                  'member_normalization_statistics': stats}
        raw = np.random.default_rng(42).integers(1, 4000, (64, 64, 4), dtype=np.uint16)
        raw[10:15, 10:15, 0] = 0
        prior = [[[0, 0], [8, 0], [8, 8], [0, 8], [0, 0]]]
        sample = prepare_sample(raw, prior, input_index='NDVI', origin_policy='shift_inside')
        for tta in ('none', 'd4'):
            masks, heads, _ = predict_fold_probabilities(model, sample, config, tta=tta)
            expected = []
            for member, normalization in zip(model.members, stats, strict=True):
                direct = prepare_sample(raw, prior, input_index='NDVI', origin_policy='shift_inside',
                                        normalization='zscore', normalization_statistics=normalization)
                expected.append(predict_probabilities(member, torch.from_numpy(direct['input'])[None], tta=tta))
            np.testing.assert_array_equal(masks, np.mean([r[0] for r in expected], axis=0))
            np.testing.assert_array_equal(heads, np.mean([r[1] for r in expected], axis=0))
        with self.assertRaises(ValueError):
            validate_inference_config({**config, 'member_normalization_statistics': stats[:2]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'ensemble.pt'
            torch.save({'model': model.state_dict(), 'config': config, 'seed': 42, 'updates': 0}, checkpoint)
            from models.checkpoint import load_checkpoint
            restored, _ = load_checkpoint(checkpoint)
            actual_masks, actual_heads, _ = predict_fold_probabilities(restored, sample, config, tta='d4')
            np.testing.assert_array_equal(actual_masks, masks)
            np.testing.assert_array_equal(actual_heads, heads)


    def test_s030_assembly_and_cli_in_isolated_process(self):
        project = Path(__file__).resolve().parents[2]
        public = json.loads((project / 'configs/inference_s030.json').read_text())
        from models.unet import SmallUNet
        from inference.unet import run_inference
        from tests.integration.test_unet_inference import _fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {'segmentation': [], 'presence': []}
            for role, experiment, updates in (('segmentation','E090',[4750,4500,1500]),
                                               ('presence','E086',[2000,1750,2500])):
                for i, update in enumerate(updates):
                    torch.manual_seed(42)
                    model = SmallUNet(input_channels=6)
                    config = {k:v for k,v in public.items() if k not in ('member_normalization_statistics','checkpoint','tta','postprocessing')}
                    config.update(model='small_unet_v1', normalization='zscore',
                                  normalization_statistics=public['member_normalization_statistics'][i], experiment_id=experiment)
                    path = root / f'{role}{i}.pt'
                    torch.save({'model':model.state_dict(),'config':config,'seed':42,'updates':update},path)
                    paths[role].append(str(path))
            checkpoint = root / 'ensemble.pt'
            command = [sys.executable,'-I','-B',str(project/'scripts/prepare_coverage_submission.py'),
                       '--segmentation',*paths['segmentation'],'--presence',*paths['presence'],'--output',str(checkpoint)]
            env = {k:v for k,v in os.environ.items() if k != 'PYTHONPATH'}
            result = subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr)
            result = subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=90)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('FileExistsError',result.stderr)
            inputs = _fixture(root/'input')
            run_inference(inputs,root/'direct.csv',checkpoint,config=public,tta='d4')
            result = subprocess.run([sys.executable,'-I','-B',str(project/'scripts/run_inference.py'),
                       '--input',str(inputs),'--output',str(root/'cli.csv'),'--checkpoint',str(checkpoint)],
                       cwd=root,env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((root/'direct.csv').read_bytes(),(root/'cli.csv').read_bytes())


if __name__ == '__main__':
    unittest.main()
