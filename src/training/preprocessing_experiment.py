import numpy as np
from preprocessing.transforms import fit_statistics


def usable(sample):
    return sample['inside'] & sample['official_valid'] & sample['numeric_valid']


def fit_frozen_statistics(train, manifest):
    """동결 split을 확인한 뒤 감독 train만으로 새 통계를 정확히 적합한다."""
    locations = set(manifest['train_locations'])
    if manifest['seed'] != 42 or locations & set(manifest['validation_locations']):
        raise ValueError('Invalid frozen partition')
    keys = [tuple(s['key']) for s in train]
    if len(keys) != len(set(keys)) or any(key[0] not in locations for key in keys):
        raise ValueError('Duplicate or non-train sample in fitting population')
    if {key[0] for key in keys} != locations:
        raise ValueError('Missing frozen train location')
    members = [s for s in train if s['presence_valid']]
    stats = fit_statistics((np.where(usable(s)[..., None], s['input'][:4].transpose(1, 2, 0), np.nan)
                            for s in members), ['/'.join(s['key']) for s in members])
    stats.update(scope='training_fit', fit_partition='frozen_train', split_id=manifest['split_id'],
                 seed=42, fit_samples=len(members), validity='inside & official_valid & numeric_valid',
                 population='supervised train ROI; overlapping ROIs counted repeatedly',
                 reflectance_divisor=10000, application='D61 E015/E016 only',
                 degenerate_scale_replacement=None)
    return stats
