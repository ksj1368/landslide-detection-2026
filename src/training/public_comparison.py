"""Run selected early recipes on the original holdout without old run directories."""
import json
from pathlib import Path

import numpy as np
import torch

from analysis.data import file_snapshot
from analysis.report import write_json
from evaluation.preprocessing_boundary_diagnostics import score_condition
from inference.unet import predict_probabilities
from models.checkpoint import inference_config, load_checkpoint
from preprocessing.model_input import append_input_index, normalize_sample, normalization_configuration
from training.checkpoint_selection_experiment import train_trajectory
from training.holdout_data import load_holdout_data
from training.preprocessing_experiment import fit_frozen_statistics
from training.roi_experiment import frozen_partition
from training.unet import _check_config


def prepare_comparison(samples, manifest, config):
    samples = [append_input_index(s, config.get('input_index', 'none')) for s in samples]
    train, validation = frozen_partition(samples, manifest)
    statistics = None
    if config['normalization'] != 'none':
        statistics = fit_frozen_statistics(train, manifest)
        config = {**config, **normalization_configuration({**config, 'normalization_statistics': statistics})}
        train, validation = [[normalize_sample(s, config['normalization'], statistics) for s in part]
                             for part in (train, validation)]
    _check_config(config)
    return train, validation, config, statistics


def run_comparison(data_root, config_path, split_path, output, *, device='cpu', execute=False):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if (output.is_relative_to(data_root) or data_root.is_relative_to(output)
            or Path(config_path).resolve().is_relative_to(output)
            or Path(split_path).resolve().is_relative_to(output)):
        raise ValueError('Output must be separate from inputs and configurations')
    config = {**json.loads(Path(config_path).read_text()), 'device': device}
    manifest = json.loads(Path(split_path).read_text())
    if config['split_id'] != manifest['split_id']:
        raise ValueError('Recipe and frozen split differ')
    base = load_holdout_data(data_root, expected_invalid_gt=config['expected_invalid_gt'],
                             roi_size=config['roi_size'], origin_policy=config['origin_policy'])
    for duplicate in base['duplicate_groups']:
        locations = {key.split('/')[0] for key in duplicate}
        if locations.intersection(manifest['train_locations']) and locations.intersection(manifest['validation_locations']):
            raise ValueError('Known exact duplicate crosses a partition')
    train, validation, config, statistics = prepare_comparison(base['samples'], manifest, config)
    prepared = {'config': config, 'split': manifest, 'statistics': statistics, 'input_snapshot': base['snapshot']}
    if output.exists():
        if json.loads((output / 'prepared.json').read_text()) != prepared:
            raise ValueError('Existing preparation differs')
    else:
        output.mkdir(parents=True)
        write_json(output / 'prepared.json', prepared)
    if not execute:
        return {'status': 'prepared', 'config': config}
    summary = train_trajectory(train, validation, config, output / 'training')
    model, saved = load_checkpoint(output / 'training/best.pt', device)
    # Shared diagnostic for postprocessing: identical checkpoint/probabilities,
    # D4 and .2/.4 for both methods; historical .5/.5 scores remain in fit_summary.
    masks, heads = [], []
    for sample in validation:
        mask, head, _ = predict_probabilities(model, torch.from_numpy(sample['input'])[None].to(device), tta='d4')
        masks.append(mask[0]); heads.append(head[0])
    conditions = {}
    for method in ('baseline', 'probability_contour'):
        result = score_condition(validation, np.asarray(masks), np.asarray(heads),
                                 condition_id=config['experiment_id'], method=method)
        write_json(output / f'{method}_d4.json', result)
        conditions[method] = result['summary']
    write_json(output / 'inference.json', {**inference_config(saved['config']), 'tta': 'd4',
               'postprocessing': {'presence_threshold': .2, 'mask_threshold': .4, 'method': 'probability_contour'}})
    if file_snapshot(base['input_files'], data_root) != base['snapshot']:
        raise ValueError('Input files changed')
    result = {'training': summary, 'postprocessing_d4_02_04': conditions}
    write_json(output / 'summary.json', result)
    return result
