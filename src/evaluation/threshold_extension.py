REFERENCE = (0.2, 0.4)

from collections import Counter, defaultdict
import math
from evaluation.scorer import _prediction_geometry
from evaluation.shape_diagnostics import geometry_metrics, geometry_change


def condition_id(value):
    p, m = value
    return f'p{round(p * 100):02d}-m{round(m * 100):02d}'


def confusion_score(confusion):
    tp, tn, fp, fn = (confusion[k] for k in ('tp', 'tn', 'fp', 'fn'))
    if not tp + fn or not tn + fp:
        return None
    return .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))


def aggregate(rows):
    """Diagnostic regrouping of scorer rows, keeping single-class Presence NA."""
    confusion = {k: sum(r['presence_outcome'] == k for r in rows) for k in ('tp', 'tn', 'fp', 'fn')}
    head_confusion = dict.fromkeys(confusion, 0)
    for row in rows:
        if row['state'] in ('positive', 'negative'):
            outcome = ('tp' if row['head_positive'] else 'fn') if row['state'] == 'positive' else ('fp' if row['head_positive'] else 'tn')
            head_confusion[outcome] += 1
    sites = defaultdict(list)
    for row in rows:
        if row['shape_status'] == 'eligible':
            sites[(row['key'][0], row['key'][2])].append(row['iou'])
    means = [sum(v) / len(v) for v in sites.values()]
    labeled = [r for r in rows if r['state'] in ('positive', 'negative')]
    return {**confusion, 'input_count': len(rows), 'labeled_count': len(labeled),
            'positive_count': confusion['tp'] + confusion['fn'], 'negative_count': confusion['tn'] + confusion['fp'],
            'presence': confusion_score(confusion),
            'presence_reason': None if confusion_score(confusion) is not None else 'single_or_missing_gt_class',
            'shape_diagnostic': sum(means) / len(means) if means else None,
            'shape_eligible_count': sum(map(len, sites.values())), 'shape_site_count': len(sites),
            'shape_reason': None if sites else 'no_shape_eligible_samples',
            'head_presence': confusion_score(head_confusion),
            **{'head_' + k: v for k, v in head_confusion.items()},
            'head_positive_count': sum(r['head_positive'] for r in rows),
            'final_positive_count': sum(r['predicted_present'] for r in rows),
            'head_positive_empty_count': sum(r['head_positive_without_polygon'] for r in rows),
            'labeled_head_positive_empty_count': sum(r['head_positive_without_polygon'] for r in labeled),
            'polygon_conversion_empty_count': sum(r['polygon_conversion_empty'] for r in rows),
            'small_positive_fn_count': sum(r['shape_status'] == 'small_positive' and r['presence_outcome'] == 'fn' for r in rows)}


def compare_rows(rows, before):
    """Partition fixed-site-weight Shape delta; common-detection means are separate."""
    if ([tuple(r['key']) for r in rows] != [tuple(r['key']) for r in before] or
            [(r['state'], r['shape_status']) for r in rows] != [(r['state'], r['shape_status']) for r in before]):
        raise ValueError('Paired comparison population differs')
    site_counts = Counter((r['key'][0], r['key'][2]) for r in before if r['shape_status'] == 'eligible')
    terms = defaultdict(list)
    counts = Counter()
    common_mass = []
    for new, old in zip(rows, before):
        transition = old['presence_outcome'], new['presence_outcome']
        for name, expected in (('fn_recovered', ('fn', 'tp')), ('new_fn', ('tp', 'fn')),
                               ('fp_resolved', ('fp', 'tn')), ('new_fp', ('tn', 'fp'))):
            counts[name] += transition == expected
        if transition == ('fn', 'tp'):
            overlap = new['gt_overlap_area']
            counts['fn_recovered_overlap_na' if overlap is None else
                   'fn_recovered_overlap_positive' if overlap > 0 else 'fn_recovered_overlap_zero'] += 1
        if old['shape_status'] != 'eligible':
            continue
        present = bool(old['predicted_present']), bool(new['predicted_present'])
        kind = {(False, True): 'new_detection', (True, False): 'lost_detection',
                (True, True): 'common_detection', (False, False): 'both_empty'}[present]
        counts[kind] += 1
        weight = 1 / (len(site_counts) * site_counts[(old['key'][0], old['key'][2])])
        terms[kind].append(weight * (new['iou'] - old['iou']))
        if kind == 'new_detection':
            counts['new_detection_overlap_positive' if new['gt_overlap_area'] > 0 else 'new_detection_overlap_zero'] += 1
        if kind == 'common_detection':
            common_mass.append(weight)
    result = {name: counts[name] for name in ('fn_recovered', 'new_fn', 'fp_resolved', 'new_fp',
        'fn_recovered_overlap_na', 'fn_recovered_overlap_positive', 'fn_recovered_overlap_zero',
        'new_detection_overlap_positive', 'new_detection_overlap_zero')}
    for kind in ('new_detection', 'lost_detection', 'common_detection', 'both_empty'):
        result[kind + '_count'] = counts[kind]
        result[kind + '_shape_delta'] = math.fsum(terms[kind]) if site_counts else None
    new_metrics, old_metrics = aggregate(rows), aggregate(before)
    for field in ('presence', 'shape_diagnostic'):
        result[field + '_delta'] = (new_metrics[field] - old_metrics[field]
                                   if new_metrics[field] is not None and old_metrics[field] is not None else None)
    mass = math.fsum(common_mass)
    result['common_detection_shape_weight'] = mass
    result['common_detection_mean_iou_delta'] = math.fsum(terms['common_detection']) / mass if mass else None
    if site_counts and not math.isclose(math.fsum(math.fsum(v) for v in terms.values()),
                                       result['shape_diagnostic_delta'], abs_tol=1e-12, rel_tol=0):
        raise ValueError('Shape transition contributions do not sum to fixed-weight delta')
    return result


def detailed_shapes(samples, conditions, *, reference_key=REFERENCE, id_formatter=condition_id):
    """Selected policies vs S010 only, without rerunning GT raster roundtrips."""
    indexed_samples = {tuple(s['key']): s for s in samples}
    reference = conditions[reference_key]
    base_rows = reference['rows']
    site_counts = Counter((r['key'][0], r['key'][2]) for r in base_rows if r['shape_status'] == 'eligible')
    metrics, geometries = {}, {}
    summaries = []
    for key, condition in conditions.items():
        metrics[key], geometries[key] = {}, {}
        predictions = {tuple(r['key']): r for r in condition['predictions']}
        for row in condition['rows']:
            if row['shape_status'] != 'eligible':
                continue
            sample_key = tuple(row['key'])
            pred = _prediction_geometry(predictions[sample_key]['polygons'])
            geometry = geometry_metrics(indexed_samples[sample_key]['gt_geometry'], pred)
            if not math.isclose(geometry['iou'], row['iou'], abs_tol=1e-12, rel_tol=0):
                raise ValueError('Detailed shape does not reproduce selected IoU')
            metrics[key][sample_key], geometries[key][sample_key] = geometry, pred
    for key, condition in conditions.items():
        if key == reference_key:
            continue
        rows = condition['rows']
        record = {'condition': id_formatter(key), 'reference': id_formatter(reference_key),
                  'calculation_origin': 'new_detailed_diagnostic', **compare_rows(rows, base_rows)}
        common, common_sites, weights = [], set(), []
        area = Counter()
        deficits = defaultdict(list)
        common_deficits = defaultdict(list)
        common_area_weights = []
        old_na = new_na = 0
        for row in rows:
            if row['shape_status'] != 'eligible':
                continue
            sample_key = tuple(row['key'])
            old, new = metrics[reference_key][sample_key], metrics[key][sample_key]
            weight = 1 / (len(site_counts) * site_counts[(sample_key[0], sample_key[2])])
            old_na += old['boundary_mean_px'] is None
            new_na += new['boundary_mean_px'] is None
            for label, value in (('reference', old), ('candidate', new)):
                deficits[label + '_missing'].append(weight if not value['pred_area'] else 0.)
                deficits[label + '_under'].append(weight * value['under_fraction'] if value['pred_area'] else 0.)
                deficits[label + '_over'].append(weight * value['over_fraction'] if value['pred_area'] else 0.)
            if old['pred_area'] and new['pred_area']:
                common_area_weights.append(weight)
                for label, value in (('reference', old), ('candidate', new)):
                    for field in ('under', 'over'):
                        common_deficits[label + '_' + field].append(weight * value[field + '_fraction'])
            change = geometry_change(indexed_samples[sample_key]['gt_geometry'], geometries[reference_key][sample_key], geometries[key][sample_key])
            for field in ('added_inside_gt_area', 'added_outside_gt_area', 'removed_inside_gt_area', 'removed_outside_gt_area'):
                area[field] += change[field]
            if change['boundary_mean_delta_px'] is not None:
                common.append(change)
                weights.append(weight)
                common_sites.add((sample_key[0], sample_key[2]))
        record.update({name + '_deficit': math.fsum(values) for name, values in deficits.items()})
        common_mass = math.fsum(common_area_weights)
        for field in ('under', 'over'):
            for label in ('reference', 'candidate'):
                value = math.fsum(common_deficits[label + '_' + field])
                record['common_' + label + '_' + field + '_contribution'] = value
                record['common_' + label + '_' + field + '_mean'] = value / common_mass if common_mass else None
            delta = (record['common_candidate_' + field + '_contribution']
                     - record['common_reference_' + field + '_contribution'])
            record['common_' + field + '_delta_contribution'] = delta
            record['common_' + field + '_mean_delta'] = delta / common_mass if common_mass else None
        record.update(area)
        record.update(reference_boundary_na=old_na, candidate_boundary_na=new_na,
                      common_boundary_count=len(common), common_boundary_site_count=len(common_sites),
                      common_boundary_sites=sorted(common_sites), common_boundary_weight=math.fsum(weights),
                      area_sum_unit='site_date_px2_not_unique_land_area')
        for field in ('boundary_mean_delta_px', 'boundary_p95_delta_px'):
            record[field] = math.fsum(w * r[field] for w, r in zip(weights, common)) / math.fsum(weights) if weights else None
        summaries.append(record)
    return summaries
