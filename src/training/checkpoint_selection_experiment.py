from itertools import combinations
import json
from pathlib import Path
import time
import numpy as np
import torch
from analysis.report import write_json
from evaluation.contracts import validate_partition
from evaluation.preprocessing_boundary_diagnostics import score_condition
from inference.unet import predict_probabilities
from training.holdout import fit_holdout
from training.holdout_data import _count_samples
from training.roi_experiment import frozen_partition
from training.unet import load_checkpoint, evaluate_loss
from training.d4_selection import evaluate_selection, selected_row


POLICIES = {'E081': 'last', 'E082': 'best'}


PROTOCOL = {
    'id': 'd100-inner-selection-v1', 'seed': 42,
    'outer_split': 'd95-inner-group3-v1', 'inner_train_locations': 6,
    'selection_locations': 2, 'updates': 5000, 'validation_every': 250,
    'selection': 'First minimum ordinary segmentation BCE + Presence BCE on inner selection only',
    'allocation': 'Among connected-group-safe pairs, minimize sum((pair_count/outer_train_count-.25)^2) for labeled, positive, negative, small_positive, shape_evaluable; lexical tie. Require all five counts in both partitions >0. No model scores.',
    'policies': POLICIES, 'normalization': 'Inner supervised train only; no refit',
    'outer_evaluation': 'After all three trajectories finish; D4/Presence .2/contour .4',
    'historical_limitation': 'Outer locations have been used in earlier development; not an untouched test.',
}


D103_POLICIES = {'last': 'last', 'bce_best': 'best', 'd4_best': 'best_d4'}


D103_PROTOCOL = {**PROTOCOL, 'id': 'd103-d4-selection-v1', 'experiment_id': 'E086',
    'policies': D103_POLICIES,
    'selection': 'First maximum inner D4 reference_total=.6Presence+.4Shape_diagnostic; retain first BCE minimum and last controls',
    'allocation': 'Frozen D100 train6/selection2/outer4; no reallocation',
    'primary_comparison': 'd4_best-bce_best', 'secondary_comparison': 'd4_best-last'}


FROZEN_SELECTION = {1: ('01_w03', '01_w06'), 2: ('01_w08', '01_w10'), 3: ('01_w05', '01_w08')}


INNER_COUNTS = {1: (224, 223, 96, 127, 59, 4, 0),
                2: (188, 173, 63, 110, 47, 5, 0),
                3: (224, 209, 97, 112, 77, 6, 0)}


def read(path):
    return json.loads(Path(path).read_text())


def allocation_counts(samples):
    counts = _count_samples(samples)
    counts['small_positive'] = sum(s['state'] == 'positive' and s['gt_geometry'].is_valid
                                   and 0 < s['gt_geometry'].area < 10 for s in samples)
    return counts


def inner_partition(fold, *, frozen_pair=None):
    """Fix a label-count-balanced pair without looking at predictions or outer labels."""
    samples, outer = fold['train'], fold['validation']
    manifest = fold['manifest']
    fields = ('labeled', 'positive', 'negative', 'small_positive', 'shape_evaluable')
    totals = allocation_counts(samples)
    choices = []
    pairs = [frozen_pair] if frozen_pair is not None else combinations(sorted(manifest['train_locations']), 2)
    for pair in pairs:
        train = [s for s in samples if s['key'][0] not in pair]
        selection = [s for s in samples if s['key'][0] in pair]
        try:
            validate_partition(train, selection)
        except ValueError:
            continue
        a, b = allocation_counts(train), allocation_counts(selection)
        if any(a[k] == 0 or b[k] == 0 for k in fields):
            continue
        cost = sum((b[k] / totals[k] - .25) ** 2 for k in fields)
        choices.append((cost, pair))
    if not choices:
        raise ValueError('No eligible connected-group-safe inner pair')
    cost, pair = min(choices)
    train = [s for s in samples if s['key'][0] not in pair]
    selection = [s for s in samples if s['key'][0] in pair]
    for a, b in ((train, selection), (train, outer), (selection, outer)):
        validate_partition(a, b)
    inner = {**manifest, 'split_id': f"d100-inner-selection-v1-fold{fold['fold_id']}",
             'train_locations': sorted(set(manifest['train_locations']) - set(pair)),
             'validation_locations': list(pair), 'outer_locations': manifest['validation_locations'],
             'selection': PROTOCOL['allocation'], 'allocation_cost': cost,
             'counts': {k: _count_samples(v) for k, v in
                        [('all', samples), ('train', train), ('validation', selection)]}}
    train, selection = frozen_partition(samples, inner)
    return inner, train, selection


def train_trajectory(train, selection, config, run):
    """No outer samples enter training, checkpoint selection, or ordinary validation."""
    run = Path(run)
    if run.exists():
        if read(run / 'config.json') != config:
            raise ValueError('Existing config differs')
        if not (run / 'fit_summary.json').exists():
            raise ValueError('Partial trajectory exists; refusing automatic restart')
        return read(run / 'fit_summary.json')
    run.mkdir(parents=True)
    write_json(run / 'config.json', config)
    result = fit_holdout(train, selection, config, run)
    write_json(run / 'fit_summary.json', result['summary'])
    return result['summary']


def evaluate_policy(run, selection, outer, config, policy, *, policies=None):
    """Restore the predeclared checkpoint; reproduce inner loss then evaluate outer once."""
    checkpoint_name = (POLICIES if policies is None else policies)[policy]
    destination = run / f'{policy}_outer_d4.json'
    if destination.exists():
        return read(destination)
    model, checkpoint = load_checkpoint(run / f'{checkpoint_name}.pt', config['device'])
    history = read(run / 'validation_history.json')
    expected = selected_row(history, checkpoint_name)
    if checkpoint['config'] != config or checkpoint['updates'] != expected['update']:
        raise ValueError('Checkpoint is not the declared inner selection policy')
    reproduced = evaluate_loss(model, selection, config['device'], config['batch_size'])
    if not np.isclose(reproduced['loss'], expected['loss'], rtol=1e-4, atol=1e-5):
        raise ValueError('Inner-selection BCE reproduction failed')
    inner_d4 = None
    if config.get('checkpoint_selection') == 'bce_and_d4_reference':
        inner_d4, inner_predictions = evaluate_selection(model, selection, config['device'])
        if inner_d4 != expected['selection_d4']:
            raise ValueError('Inner D4 checkpoint score reproduction failed')
        if checkpoint_name == 'best_d4':
            saved = read(run / 'best_d4_predictions.json')
            if [r['polygons'] for r in inner_predictions] != [r['polygons'] for r in saved]:
                raise ValueError('Restored inner D4 polygons differ')
    masks, heads = [], []
    started = time.perf_counter()
    for sample in outer:
        tensor = torch.from_numpy(sample['input']).unsqueeze(0).to(config['device'])
        mask, head, _ = predict_probabilities(model, tensor, tta='d4')
        masks.append(mask[0]); heads.append(head[0])
    masks, heads = np.asarray(masks), np.asarray(heads)
    np.savez_compressed(run / f'{policy}_outer_d4.npz', masks=masks, presence=heads,
                        keys=np.asarray([s['key'] for s in outer]))
    condition = score_condition(outer, masks, heads, condition_id=policy, method='probability_contour')
    condition['checkpoint'] = {'file': checkpoint_name + '.pt', 'update': checkpoint['updates'],
                               'inner_loss': reproduced['loss'], 'seconds': time.perf_counter()-started}
    if inner_d4 is not None:
        condition['checkpoint']['inner_d4'] = inner_d4
    write_json(destination, condition)
    del model, checkpoint
    if config['device'] == 'mps':
        torch.mps.empty_cache()
    return condition
