"""Run local S030 inference from explicit inputs and a supplied checkpoint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from inference.unet import run_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/inference_s030.json')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    result = run_inference(args.input, args.output, args.checkpoint, device=args.device,
                           config=config, tta=config['tta'])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
