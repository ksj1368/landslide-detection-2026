"""D106: exact pixel occupancy and fixed train-only positive-site Dice weights."""
from collections import Counter

import numpy as np
from shapely.geometry import box

from postprocessing.boundary_processing import _contour_geometry


def exact_coverage(geometry, origin, shape):
    """Intersection with unit cells in original-image coordinates; no raster approximation."""
    result = np.zeros(shape, dtype=np.float32)
    if geometry.is_empty:
        return result
    if not geometry.is_valid:
        raise ValueError('Coverage requires valid original GT; no repair')
    x0, y0 = origin
    height, width = shape
    left, top, right, bottom = geometry.bounds
    for y in range(max(0, int(np.floor(top-y0))), min(height, int(np.ceil(bottom-y0)))):
        for x in range(max(0, int(np.floor(left-x0))), min(width, int(np.ceil(right-x0)))):
            result[y, x] = geometry.intersection(box(x0+x, y0+y, x0+x+1, y0+y+1)).area
    expected = geometry.intersection(box(x0, y0, x0+width, y0+height)).area
    if not np.isclose(result.sum(dtype=np.float64), expected, rtol=2e-7, atol=1e-6):
        raise ValueError('Pixel coverage area is not conserved')
    return result


def dice_positive(sample):
    valid = sample['segmentation_valid']
    return bool(np.where(valid, sample['target'], 0).sum() > 0)


def attach_site_weights(train):
    counts = Counter((s['key'][0], s['key'][2]) for s in train if s['presence_valid'] and dice_positive(s))
    total, sites = sum(counts.values()), len(counts)
    if not sites:
        raise ValueError('No positive Dice sites in training')
    weights = {key: total/(sites*count) for key, count in counts.items()}
    rows = [{'location': key[0], 'site': key[1], 'positive_rois': count,
             'weight': weights[key]} for key, count in sorted(counts.items())]
    if not np.isclose(sum(counts[k]*weights[k] for k in counts)/total, 1):
        raise ValueError('Population mean site weight differs from one')
    samples = [dict(s, dice_site_weight=np.float32(weights[(s['key'][0], s['key'][2])]
               if s['presence_valid'] and dice_positive(s) else 1)) for s in train]
    return samples, rows


def attach_coverage(train):
    samples, rows = [], []
    for sample in train:
        coverage = np.zeros_like(sample['target'], dtype=np.float32)
        valid = sample['segmentation_valid'][0]
        if sample['state'] == 'positive' and valid.any():
            geometry = sample['gt_geometry']
            coverage[0] = exact_coverage(geometry, sample['origin'], valid.shape)
            usable = sample['inside'] & sample['official_valid'] & sample['numeric_valid']
            row = {'location': sample['key'][0], 'date': sample['key'][1], 'site': sample['key'][2],
                   'gt_area': geometry.area, 'small': geometry.area < 10,
                   'coverage_area': float(coverage.sum(dtype=np.float64)),
                   'supervised_coverage_area': float(coverage[0][valid].sum(dtype=np.float64))}
            for name, values in (('binary', sample['target'][0]), ('coverage', coverage[0])):
                prediction, _ = _contour_geometry(values, usable, sample['origin'], sample['original_shape'], .4)
                row[name+'_empty'] = prediction.is_empty
                row[name+'_iou'] = prediction.intersection(geometry).area/prediction.union(geometry).area
            rows.append(row)
        samples.append(dict(sample, coverage_target=coverage))
    return samples, rows
