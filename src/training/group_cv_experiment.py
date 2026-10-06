import torch
from evaluation.contracts import validate_oof
from models.unet import SmallUNet
from training.holdout_data import _count_samples
from training.roi_experiment import frozen_partition
from training.unet import set_seed, make_batch


def make_folds(raw_samples, parent, protocol):
    """Apply the frozen allocation and check supervision and connected groups."""
    locations = set(parent['train_locations'])
    assignments = protocol['validation_locations']
    if (protocol['seed'] != 42 or protocol['parent_split_id'] != parent['split_id']
            or len(assignments) != 3 or any(len(f) != 4 for f in assignments)
            or len({loc for f in assignments for loc in f}) != 12
            or set(loc for f in assignments for loc in f) != locations
            or locations & set(parent['validation_locations'])):
        raise ValueError('D95 fold allocation differs from the frozen parent population')
    samples = sorted((s for s in raw_samples if s['key'][0] in locations), key=lambda s: s['key'])
    if _count_samples(samples) != parent['counts']['train']:
        raise ValueError('Original train population changed')
    groups = {g: members for g, members in parent['connected_groups'].items()
              if set(members) & locations}
    folds = []
    for number, held in enumerate(assignments, 1):
        train, valid = ([s for s in samples if (s['key'][0] in held) == selected]
                        for selected in (False, True))
        manifest = {'split_id': f"{protocol['split_id']}-fold{number}", 'seed': 42,
                    'fold_id': number, 'parent_split_id': parent['split_id'],
                    'train_locations': sorted(locations - set(held)), 'validation_locations': held,
                    'connected_groups': groups, 'selection': protocol['selection'],
                    'counts': {k: _count_samples(v) for k, v in
                               [('all', samples), ('train', train), ('validation', valid)]}}
        train, valid = frozen_partition(samples, manifest)
        eligible = [s for s in valid if s['state'] == 'positive' and s['gt_geometry'].is_valid
                    and not s['gt_geometry'].is_empty]
        counts = manifest['counts']['validation']
        actual = [counts[k] for k in ('samples', 'labeled', 'positive', 'negative')]
        actual += [sum(s['gt_geometry'].area < 10 for s in eligible), counts['shape_evaluable'],
                   len({(s['key'][0], s['key'][2]) for s in eligible if s['gt_geometry'].area >= 10})]
        if actual != protocol['expected_validation_counts'][number - 1]:
            raise ValueError('D95 validation denominators changed')
        folds.append({'fold_id': number, 'manifest': manifest, 'train': train, 'validation': valid})
    validate_oof([{'key': s['key']} for f in folds for s in f['validation']], [s['key'] for s in samples])
    return folds


def forward_check(train, device):
    """Verify paired scratch initialization and a real FP32 batch of 32."""
    set_seed()
    first = SmallUNet(96, 6)
    rng = torch.get_rng_state()
    set_seed()
    second = SmallUNet(96, 6)
    if (not torch.equal(rng, torch.get_rng_state()) or
            any(not torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())):
        raise ValueError('Paired initial weights/RNG differ')
    model = first.to(device).eval()
    batch = make_batch([s for s in train if s['presence_valid']][:32], device)
    with torch.inference_mode():
        output = model(batch['input'])
    shapes = {k: list(v.shape) for k, v in output.items()}
    if (list(batch['input'].shape) != [32, 6, 96, 96]
            or shapes != {'segmentation': [32, 1, 96, 96], 'presence': [32]}
            or any(not torch.isfinite(v).all() for v in output.values())):
        raise ValueError('Real batch forward failed')
    return {'device': device, 'input': list(batch['input'].shape), 'outputs': shapes,
            'parameters': sum(p.numel() for p in model.parameters()), 'paired_initialization_equal': True}
