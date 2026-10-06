"""Run a selected U-Net input/loss recipe and compare two postprocessing methods."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from training.public_comparison import run_comparison


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--split', type=Path, default=root / 'configs/splits/fixed-group-holdout-v1-6f2d58d1a68c.json')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_comparison(args.data_root, args.config, args.split, args.output,
                                   device=args.device, execute=args.execute), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
