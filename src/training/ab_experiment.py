"""D106: independent occupancy-BCE and site-balanced Dice experiments."""
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from analysis.data import file_snapshot
from analysis.report import write_json
from evaluation.contracts import validate_oof
from evaluation.group_cv_checkpoint_diagnostics import verify_pooled
from evaluation.prediction_diagnostics import _enrich_summary
from evaluation.preprocessing_boundary_diagnostics import named_group_diagnostics, named_detailed_shapes
from evaluation.scorer import score_predictions
from preprocessing.model_input import normalize_sample
from training.checkpoint_selection_experiment import (
    read, FROZEN_SELECTION, INNER_COUNTS, inner_partition,
    train_trajectory, evaluate_policy, allocation_counts, D103_PROTOCOL)
from training.group_cv_experiment import make_folds, forward_check
from training.holdout_data import load_holdout_data
from training.index_experiment import canonical_sampling_trace
from training.preprocessing_experiment import fit_frozen_statistics
from training.two_stage_data import prepare_raw_samples
from training.d4_selection import selection_population
from training.unet import (_check_config, set_seed, make_batch, supervised_loss,
                           BCE_DICE, COVERAGE_BCE_DICE, SITE_BCE_DICE)
from models.unet import SmallUNet
from training.ab_supervision import attach_coverage, attach_site_weights

VARIANTS = {'baseline': ('E086', BCE_DICE), 'A': ('E090', COVERAGE_BCE_DICE),
            'B': ('E091', SITE_BCE_DICE)}
AB_PROTOCOL = {**D103_PROTOCOL, 'id': 'd106-ab-independent-v1',
    'experiment_id': ['E086_new_run', 'E090', 'E091'], 'variants': VARIANTS,
    'policies': {'best_d4': 'best_d4', 'last': 'last'},
    'primary_comparison': ['A_best_d4-baseline_best_d4', 'B_best_d4-baseline_best_d4'],
    'secondary_comparison': ['A_last-baseline_last', 'B_last-baseline_last'],
    'budget': {'trajectories': 9, 'updates': 45000, 'ordinary_validations': 180, 'd4_validations': 180},
    'order': 'fold1,2,3; baseline,A,B; all training before outer evaluation',
    'coverage': 'Exact original GT/unit pixel intersection; BCE only; binary Dice retained',
    'site_weight': 'N_positive/(K_positive*n_site), train valid-positive ROI only; batch-positive mean, no renormalization',
    'outer_evaluation': 'After all nine trajectories; best_d4 primary, last secondary; D4/.2/.4'}
# JSON round-trip gives protocol values the same types when resuming preparation.
AB_PROTOCOL = json.loads(json.dumps(AB_PROTOCOL))

def run_experiment(data_root, config_dir, output, *, execute=False, device='mps', include_bce_control=False):
    active = {**AB_PROTOCOL, 'device': device}
    if include_bce_control:
        active['policies'] = {**active['policies'], 'bce': 'best'}
    policies = active['policies']
    data_root, config_dir, output = Path(data_root).resolve(), Path(config_dir).resolve(), Path(output).resolve()
    torch.set_num_threads(2)
    if device not in ('cpu', 'mps'):
        raise ValueError('Supported training devices: cpu, mps')
    if execute and device == 'mps' and (not torch.backends.mps.is_available()
                                      or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0'):
        raise RuntimeError('MPS training requires MPS FP32 fallback=0')
    if output.is_relative_to(data_root) or data_root.is_relative_to(output) or output.is_relative_to(config_dir):
        raise ValueError('Output must be separate from data and frozen configurations')
    if output.exists() and read(output / 'protocol.json') != active:
        raise ValueError('Existing protocol differs')
    if (output / 'summary.json').exists():
        return read(output / 'summary.json')
    output.mkdir(parents=True, exist_ok=True)
    if not (output / 'protocol.json').exists():
        write_json(output / 'protocol.json', active)
    started = time.perf_counter()
    base = load_holdout_data(data_root, expected_invalid_gt=3, roi_size=96, origin_policy='shift_inside')
    if (output / 'input_snapshot.json').exists() and read(output / 'input_snapshot.json') != base['snapshot']:
        raise ValueError('Input files changed since preparation')
    parent = read(config_dir / 'splits/fixed-group-holdout-v1-6f2d58d1a68c.json')
    folds = make_folds(prepare_raw_samples(base['samples']), parent,
                       read(config_dir / 'splits/d95-inner-group3-v1.json'))
    prepared = []
    for fold in folds:
        pair = tuple('ls_scene_'+v for v in FROZEN_SELECTION[fold['fold_id']])
        manifest, train, selection = inner_partition(fold, frozen_pair=pair)
        population = selection_population(selection)
        if tuple(population.values()) != INNER_COUNTS[fold['fold_id']]:
            raise ValueError('Frozen D103 inner denominators differ')
        original_prepared = read(config_dir / f"training/fold{fold['fold_id']}.json")
        if manifest != original_prepared['split']:
            raise ValueError('D100 partition changed')
        outer = fold['validation']
        partitions = (train, selection, outer)
        for duplicate in base['duplicate_groups']:
            locations = {str(k).split('/')[0] for k in duplicate}
            if sum(bool(locations & {s['key'][0] for s in part}) for part in partitions) > 1:
                raise ValueError('Known exact duplicate crosses a partition')
        stats = fit_frozen_statistics(train, manifest)
        stats['application'] = 'D100 inner supervised train only; selection and outer excluded'
        if set(stats['image_keys']) != {'/'.join(s['key']) for s in train if s['presence_valid']}:
            raise ValueError('Normalization fitting population differs')
        config = {**original_prepared['config'], 'device': device}
        if {k:v for k,v in stats.items() if k != 'image_keys'} != original_prepared['statistics']:
            raise ValueError('D100 train-only normalization changed')
        config.update(experiment_id='E086', checkpoint_selection='bce_and_d4_reference',
                      primary_evaluation_checkpoint='best_d4.pt; same-trajectory best.pt and last.pt controls')
        directory = output / f"fold{fold['fold_id']}"

        frozen = {'split': manifest, 'statistics': stats, 'config': config}
        if (directory / 'prepared.json').exists() and read(directory / 'prepared.json') != frozen:
            raise ValueError('Frozen preparation changed')
        write_json(directory / 'prepared.json', frozen)
        normalized = [[normalize_sample(s, 'zscore', stats) for s in part] for part in partitions]
        variants = {'baseline': normalized[0]}
        coverage_path = directory / 'coverage_targets.npz'
        if coverage_path.exists():
            with np.load(coverage_path) as cache:
                if cache['keys'].tolist() != [list(s['key']) for s in normalized[0]]:
                    raise ValueError('Coverage cache order differs')
                variants['A'] = [dict(s, coverage_target=c) for s,c in zip(normalized[0], cache['coverage'], strict=True)]
            oracle = read(directory / 'coverage_oracle.json')
        else:
            variants['A'], oracle = attach_coverage(normalized[0])
            np.savez_compressed(coverage_path, keys=np.asarray([s['key'] for s in normalized[0]]),
                                coverage=np.stack([s['coverage_target'] for s in variants['A']]))
        variants['B'], weights = attach_site_weights(normalized[0])
        configs = {name: {**config, 'experiment_id': experiment, 'loss': loss,
                   'primary_evaluation_checkpoint': 'best_d4.pt; last.pt secondary'}
                   for name, (experiment, loss) in VARIANTS.items()}
        for candidate in configs.values():
            _check_config(candidate)
        write_json(directory / 'site_weights.json', weights)
        write_json(directory / 'coverage_oracle.json', oracle)
        write_json(directory / 'variant_configs.json', configs)
        fold.update(train=normalized[0], selection=normalized[1], validation=normalized[2],
                    config=config, variants=variants, configs=configs)
        trace = canonical_sampling_trace(fold['train'], config)
        fold['trace'] = trace
        record = {'fold': fold['fold_id'], 'train_locations': manifest['train_locations'],
                  'selection_locations': manifest['validation_locations'], 'outer_locations': manifest['outer_locations'],
                  'counts': {k: allocation_counts(v) for k, v in zip(('train','selection','outer'),partitions)},
                  'sampling': trace}
        prepared.append(record)
    write_json(output / 'preflight.json', prepared)
    write_json(output / 'input_snapshot.json', base['snapshot'])
    write_json(output / ('execution_preparation_timing.json' if execute else 'preparation_timing.json'),
               {'seconds': time.perf_counter()-started})
    if not execute:
        return {'complete': False, 'status': 'prepared', 'folds': prepared}
    runs, conditions = {}, {}
    # Freeze all folds and auxiliary supervision before the first optimizer step.
    smoke = []
    for fold in folds:
        for name in VARIANTS:
            data, config = fold['variants'][name], fold['configs'][name]
            if canonical_sampling_trace(data, config) != fold['trace']:
                raise ValueError('Variant sample/D4 order differs')
            checked = forward_check(data, device)
            set_seed()
            model = SmallUNet(96, 6).to(device).eval()
            batch = make_batch([s for s in data if s['presence_valid']][:32], device)
            with torch.inference_mode():
                losses = supervised_loss(model(batch['input']), batch, config)
            if not all(torch.isfinite(v).all() for v in losses):
                raise ValueError('Real batch loss smoke failed')
            smoke.append({'fold': fold['fold_id'], 'variant': name, **checked})
            del model, batch
    write_json(output / 'real_batch_smoke.json', smoke)
    for fold in folds:
        number = str(fold['fold_id'])
        for name in VARIANTS:
            print(json.dumps({'stage': 'training', 'fold': number, 'variant': name}), flush=True)
            fit = train_trajectory(fold['variants'][name], fold['selection'], fold['configs'][name],
                                   output / f'fold{number}/{name}')
            if (fit['updates'] != 5000 or fit['validation_count'] != 20 or fit['d4_validation_count'] != 20
                    or fit['sampling_d4_trace_sha256'] != fold['trace']['sha256']
                    or fit['samples_seen'] != fold['trace']['samples_seen']):
                raise ValueError('Training budget or sampling differs')
            runs[f'{number}/{name}'] = fit
            write_json(output / 'progress.json', {'completed_training': list(runs), 'planned': 9})
            gc.collect()
    write_json(output / 'all_training_complete.json', {'runs': list(runs), 'time': time.time()})
    for fold in folds:
        number = str(fold['fold_id'])
        for name in VARIANTS:
            for policy in policies:
                print(json.dumps({'stage': 'outer_evaluation', 'fold': number, 'variant': name,
                                  'policy': policy}), flush=True)
                conditions[f'{number}/{name}/{policy}'] = evaluate_policy(output / f'fold{number}/{name}',
                    fold['selection'], fold['validation'], fold['configs'][name], policy, policies=policies)
    samples = sorted([s for f in folds for s in f['validation']], key=lambda s:s['key'])
    pooled = []
    for name in VARIANTS:
        for policy in policies:
            predictions, rows = [], []
            for fold in folds:
                c = conditions[f"{fold['fold_id']}/{name}/{policy}"]
                validate_oof(c['predictions'], [s['key'] for s in fold['validation']])
                predictions.extend(c['predictions']); rows.extend(c['rows'])
            predictions.sort(key=lambda r:tuple(r['key'])); rows.sort(key=lambda r:tuple(r['key']))
            validate_oof(predictions, [s['key'] for s in samples])
            score = _enrich_summary(score_predictions(samples,predictions,presence_threshold=.2)['summary'],rows)
            score['reference_total'] = .6*score['presence']+.4*score['shape_diagnostic']
            c = {'id':name+'_'+policy,'summary':score,'rows':rows,'predictions':predictions}
            verify_pooled(c)
            write_json(output / f'{name}_{policy}_pooled_oof.json',c)
            pooled.append(c)
    details = []
    for policy in policies:
        selected = [c for c in pooled if c['id'].endswith('_'+policy)]
        details.extend(named_detailed_shapes(samples, selected))
    write_json(output / 'detailed_shapes.json', details)
    groups, sensitivity = named_group_diagnostics(pooled)
    write_json(output / 'groups.json',groups)
    write_json(output / 'sensitivity.json',sensitivity)
    if file_snapshot(base['input_files'],data_root) != base['snapshot']:
        raise ValueError('Input files changed')
    summary = {'complete':True,'protocol':active,'runs':runs,
               'pooled':{c['id']:c['summary'] for c in pooled},
               'fold_scores':{k:v['summary'] for k,v in conditions.items()},
               'checkpoints':{k:v['checkpoint'] for k,v in conditions.items()},
               'updates':45000,'validations':180,'total_seconds':time.perf_counter()-started,
               'device':device,'dtype':'float32','fallback':0,'input_files_unchanged':True}
    summary['d4_validations'] = sum(r['d4_validation_count'] for r in runs.values())
    write_json(output / 'summary.json',summary)
    return summary
