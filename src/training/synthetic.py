import numpy as np
from preprocessing.model_input import prepare_sample


def fixture_samples():
    """A fresh generated RGBNIR image and independent synthetic location groups."""
    rng = np.random.default_rng(42)
    raw = rng.integers(100, 9000, (256, 256, 4), dtype=np.uint16)
    prior = [[112, 112], [144, 112], [144, 144], [112, 144], [112, 112]]
    positive = {"visible": 1, "polygon": [[120, 124], [135, 124], [135, 137], [120, 137], [120, 124]]}
    samples = []
    for location in ("synthetic_train", "synthetic_validation"):
        for index, label in enumerate((positive, {"visible": 0}, None, {"visible": 1, "ignore": True})):
            sample = prepare_sample(raw, prior, label)
            sample.update(key=(location, "20220101", f"s{index}"), group=location)
            samples.append(sample)
    return samples[:4], samples[4:]
