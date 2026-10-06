"""Assemble six role-specific members; use --retrained for newly trained weights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from models.checkpoint import load_checkpoint, inference_config
from models.fold_ensemble import CoverageFoldEnsemble
from training.d4_selection import selected_row


def assemble(segmentation, presence, output, config_path, *, retrained=False):
    if len(segmentation) != 3 or len(presence) != 3:
        raise ValueError('Supply fold1, fold2, fold3 for each role')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config_output = output.with_suffix('.json')
    if retrained and (config_output == output or config_output.exists()):
        raise FileExistsError(config_output)
    public = json.loads(Path(config_path).read_text())
    model = CoverageFoldEnsemble(baseline_presence=True).eval()
    configs, updates, presence_updates = [], [], []
    def check_retrained(saved, source, fold):
        if not retrained:
            return
        history = json.loads(Path(source).with_name('validation_history.json').read_text())
        selected = selected_row(history, 'best_d4')
        if (saved['config'].get('fold_id') != fold
                or saved['config'].get('checkpoint_selection') != 'bce_and_d4_reference'
                or saved['updates'] != selected['update']
                or saved.get('metadata', {}).get('selection_d4') != selected['selection_d4']):
            raise ValueError('Retrained members require ordered folds and their selected best_d4 checkpoint')

    for index, (member, source) in enumerate(zip(model.members, segmentation, strict=True)):
        _, saved = load_checkpoint(source)
        check_retrained(saved, source, index + 1)
        if saved['config'].get('experiment_id') != 'E090':
            raise ValueError('Segmentation requires E090/A')
        member.candidate_model.load_state_dict(saved['model'], strict=True)
        configs.append(inference_config(saved['config']))
        updates.append(saved['updates'])
    for index, source in enumerate(presence):
        _, saved = load_checkpoint(source)
        check_retrained(saved, source, index + 1)
        if saved['config'].get('experiment_id') != 'E086' or inference_config(saved['config']) != configs[index]:
            raise ValueError('Presence member/config does not match the recorded E086 fold')
        model.members[index].baseline_model.load_state_dict(saved['model'], strict=True)
        presence_updates.append(saved['updates'])
    if not retrained and (updates != [4750, 4500, 1500] or presence_updates != [2000, 1750, 2500]):
        raise ValueError('Selected updates differ from S030; do not label retrained weights as the original')
    for config in configs[1:]:
        if {k:v for k,v in config.items() if k != 'normalization_statistics'} != {k:v for k,v in configs[0].items() if k != 'normalization_statistics'}:
            raise ValueError('Fold input/model contracts differ')
    config = {**configs[0], 'model': model.model_id, 'experiment_id': 'E107',
              'normalization': 'none', 'member_normalization_statistics': [c['normalization_statistics'] for c in configs]}
    config.pop('normalization_statistics')
    if not retrained and inference_config(config) != inference_config(public):
        raise ValueError('Member statistics differ from the published S030 configuration')
    if retrained:
        config['experiment_id'] = 'retrained_role_ensemble'
    inference_config(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        torch.save({'model': model.state_dict(), 'config': config, 'seed': 42, 'updates': 0,
                    'metadata': {'retrained': retrained, 'segmentation_updates': updates,
                                 'presence_updates': presence_updates}}, stream)
    result = {'checkpoint': str(output), 'segmentation_updates': updates, 'presence_updates': presence_updates}
    if retrained:
        with config_output.open('x') as stream:
            json.dump({**inference_config(config), 'tta': public['tta'],
                       'postprocessing': public['postprocessing']}, stream, indent=2, allow_nan=False)
            stream.write('\n')
        result['inference_config'] = str(config_output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segmentation', nargs=3, required=True, type=Path, metavar=('FOLD1', 'FOLD2', 'FOLD3'))
    parser.add_argument('--presence', nargs=3, required=True, type=Path, metavar=('FOLD1', 'FOLD2', 'FOLD3'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/inference_s030.json')
    parser.add_argument('--retrained', action='store_true', help='Use new best_d4 selections and write a matching inference JSON')
    args = parser.parse_args()
    print(json.dumps(assemble(args.segmentation, args.presence, args.output, args.config, retrained=args.retrained), indent=2))


if __name__ == '__main__':
    main()
