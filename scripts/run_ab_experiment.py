"""Prepare the frozen D106 comparison; --execute runs all nine trajectories."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from training.ab_experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'configs')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--include-bce-control', action='store_true', help='Also evaluate best.pt on each outer fold; training is unchanged')
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.data_root, args.config_dir, args.output,
                     execute=args.execute, device=args.device,
                     include_bce_control=args.include_bce_control), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
