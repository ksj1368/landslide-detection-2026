"""Synthetic loader -> frozen split -> training -> postprocessing -> inference."""
import json
from pathlib import Path
import tempfile
import unittest

from inference.unet import run_inference
from training.holdout_data import load_holdout_data, _count_samples
from training.public_comparison import run_comparison
from tests.integration.test_holdout_data import _fixture


class PublicComparisonTests(unittest.TestCase):
    def test_selected_recipes_train_and_export(self):
        project = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            _fixture(data)
            samples = load_holdout_data(data, expected_invalid_gt=0)['samples']
            train = [s for s in samples if s['key'][0] != 'location4']
            validation = [s for s in samples if s['key'][0] == 'location4']
            split = {'split_id': 'synthetic', 'seed': 42,
                     'train_locations': [f'location{i}' for i in range(4)],
                     'validation_locations': ['location4'],
                     'connected_groups': {f'location{i}': [f'location{i}'] for i in range(5)},
                     'counts': {k: _count_samples(v) for k,v in
                                (('all', samples), ('train', train), ('validation', validation))}}
            split_path = root / 'split.json'
            split_path.write_text(json.dumps(split))
            for recipe in sorted((project / 'configs/experiments').glob('*.json')):
                with self.subTest(recipe=recipe.stem):
                    config = json.loads(recipe.read_text())
                    config.update(split_id='synthetic', updates=1, validation_every=1,
                                  batch_size=3, expected_invalid_gt=0)
                    config_path = root / recipe.name
                    config_path.write_text(json.dumps(config))
                    output = root / recipe.stem
                    prepared = run_comparison(data, config_path, split_path, output)
                    self.assertEqual(prepared['status'], 'prepared')
                    result = run_comparison(data, config_path, split_path, output, execute=True)
                    self.assertEqual(result['training']['updates'], 1)
                    self.assertEqual(set(result['postprocessing_d4_02_04']), {'baseline', 'probability_contour'})
                    inference = json.loads((output / 'inference.json').read_text())
                    run_inference(data, output / 'predictions.csv', output / 'training/best.pt',
                                  config=inference, tta='d4')
                    self.assertEqual(len((output / 'predictions.csv').read_text().splitlines()), 16)
                    if config['normalization'] == 'zscore':
                        statistics = json.loads((output / 'prepared.json').read_text())['statistics']
                        self.assertEqual(statistics['fit_samples'], 8)
                        self.assertTrue(all(k.startswith(tuple(split['train_locations'])) for k in statistics['image_keys']))
            # Keep the real loader's duplicate-image partition check in the public runner.
            source = data / 'location0/bands/20220101.npy'
            (data / 'location4/bands/20220101.npy').write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, 'duplicate crosses'):
                run_comparison(data, config_path, split_path, root / 'duplicate')


if __name__ == '__main__':
    unittest.main()
