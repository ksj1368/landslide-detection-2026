import numpy as np
from preprocessing.model_input import append_input_index


def prepare_raw_samples(samples):
    """Append NDVI to unnormalized A96 loader samples without modifying their arrays."""
    prepared, seen = [], set()
    for sample in samples:
        key = tuple(sample['key'])
        if len(key) != 3 or any(not isinstance(part, str) for part in key) or key in seen:
            raise ValueError('Expected unique string location/date/site keys')
        seen.add(key)
        if (sample.get('normalization', 'none') != 'none'
                or sample.get('normalization_statistics') is not None
                or sample.get('roi_size') != 96 or sample.get('origin_policy') != 'shift_inside'
                or sample['input'].shape != (5, 96, 96)
                or sample['input'].dtype != np.float32 or not np.isfinite(sample['input']).all()):
            raise ValueError('Expected unnormalized FP32 five-channel A96 samples')
        prepared.append({**append_input_index(sample, 'NDVI'), 'key': key})
    if not prepared:
        raise ValueError('No raw samples supplied')
    return sorted(prepared, key=lambda sample: sample['key'])
