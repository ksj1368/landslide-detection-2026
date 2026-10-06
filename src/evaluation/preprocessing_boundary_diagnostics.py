import numpy as np
from evaluation.prediction_diagnostics import _condition, _mask_diagnostics
from evaluation.scorer import _prediction_geometry
from evaluation.threshold_extension import aggregate, compare_rows, detailed_shapes
from postprocessing.boundary_processing import boundary_prediction


def score_condition(samples, masks, heads, *, condition_id, method='baseline', tolerance=None):
    """Score the fixed .2/.4 policy with existing scorer and exact decoded geometry."""
    masks, heads = np.asarray(masks), np.asarray(heads)
    if (not samples or masks.ndim != 3 or len(masks) != len(samples) or heads.shape != (len(samples),)
            or not np.isfinite(masks).all() or not np.isfinite(heads).all()
            or np.any((masks < 0) | (masks > 1)) or np.any((heads < 0) | (heads > 1))):
        raise ValueError('Condition probabilities must match the fixed sample population')
    if method == 'baseline' and tolerance is not None:
        raise ValueError('Baseline does not use simplification tolerance')
    cached = []
    for sample, mask in zip(samples, masks, strict=True):
        polygons, values = _mask_diagnostics(sample, mask, .4)
        if method != 'baseline':
            decoded = boundary_prediction(sample, mask, 1., method=method, tolerance=tolerance)
            polygons = decoded['polygons']
            prediction = _prediction_geometry(polygons)
            gt = sample.get('gt_geometry')
            overlap = float(gt.intersection(prediction).area) if values['gt_valid'] else None
            iou = (float(overlap / gt.union(prediction).area) if values['gt_valid'] else None)
            values.update(head_bypass_present=not prediction.is_empty, head_bypass_iou=iou,
                          head_bypass_overlap_area=overlap,
                          polygon_conversion_empty=bool(values['usable_mask_pixels'] and prediction.is_empty),
                          head_bypass_contour_degenerate_ring_count=decoded['contour_degenerate_ring_count'])
        cached.append((polygons, values))
    # Keep the reference's historical mode for exact reproduction; new records
    # identify their own training/postprocessing experiment explicitly.
    mode = 'E015-d4' if condition_id == 'S010' else condition_id + '-d4'
    result = _condition(samples, masks, heads, cached, mode, .2, .4)
    result.update(id=condition_id, method=method, tolerance=tolerance)
    if method != 'baseline':
        for row in result['rows']:
            row.pop('gt_pixel_recall')
            row.pop('final_gt_pixel_recall')
            row['gt_pixel_reason'] = 'not_computed_for_boundary_geometry'
        result['summary']['head_bypass_contour_degenerate_ring_count'] = sum(
            row['head_bypass_contour_degenerate_ring_count'] for row in result['rows'])
        result['summary']['head_pass_contour_degenerate_ring_count'] = sum(
            row['head_bypass_contour_degenerate_ring_count'] for row in result['rows'] if row['head_positive'])
        for summary in [result['summary'], *result['regions'], *result['sizes']]:
            summary['raster_diagnostics'] = 'threshold_input_only_no_final_polygon_raster_recall'
    return result


def named_group_diagnostics(conditions):
    """Region/site regrouping and leave-one-group sensitivity, never CV."""
    base = conditions[0]['rows']
    groups, sensitivity = [], []
    for condition in conditions:
        rows = condition['rows']
        overall = compare_rows(rows, base)
        for kind in ('location', 'site'):
            labels = [(r['key'][0], r['key'][2]) if kind == 'site' else r['key'][0] for r in base]
            for label in sorted(set(labels)):
                indices = [i for i, value in enumerate(labels) if value == label]
                selected, before = [rows[i] for i in indices], [base[i] for i in indices]
                context = {'condition': condition['id'], 'group_type': kind, 'group': label}
                groups.append({**context, **aggregate(selected), **compare_rows(selected, before)})
                remaining = [i for i, value in enumerate(labels) if value != label]
                after, prior = [rows[i] for i in remaining], [base[i] for i in remaining]
                comparison = compare_rows(after, prior)
                sensitivity.append({**context, **aggregate(after), **comparison,
                    'diagnostic_only_not_cv': True,
                    'sign_reversal': any(overall[k] is not None and overall[k] >= 0
                                         and comparison[k] is not None and comparison[k] < -1e-12
                                         for k in ('presence_delta', 'shape_diagnostic_delta'))})
    return groups, sensitivity


def named_detailed_shapes(samples, conditions):
    """Fixed-population deficits and distinct common-detection shape/boundary changes."""
    indexed = {condition['id']: condition for condition in conditions}
    if len(indexed) != len(conditions):
        raise ValueError('Named conditions must have unique IDs')
    return detailed_shapes(samples, indexed, reference_key=conditions[0]['id'], id_formatter=str)
