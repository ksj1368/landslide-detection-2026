"""Create one synthetic RGBNIR scene and prior polygon; no real assets needed."""
import argparse
import json
from pathlib import Path

import numpy as np


def make_inputs(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    scene = root / 'synthetic_region'
    (scene / 'bands').mkdir(parents=True)
    raw = np.random.default_rng(42).integers(400, 4000, (96, 96, 4), dtype=np.uint16)
    np.save(scene / 'bands/20200101.npy', raw, allow_pickle=False)
    sites = [{'site_id': '001', 'prior_polygon': [[[24,24],[72,24],[72,72],[24,72],[24,24]]]}]
    (scene / 'sites.json').write_text(json.dumps(sites, indent=2) + '\n', encoding='utf-8')
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    print(make_inputs(parser.parse_args().output))
