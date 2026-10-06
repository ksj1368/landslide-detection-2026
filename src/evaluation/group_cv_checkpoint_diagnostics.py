from collections import defaultdict
import numpy as np


def verify_pooled(condition):
    """Independent confusion/site aggregation protects the diagnostic denominator."""
    rows, score = condition['rows'], condition['summary']
    counts = {name: sum(r['presence_outcome'] == name for r in rows) for name in ('tp', 'tn', 'fp', 'fn')}
    sites = defaultdict(list)
    for row in rows:
        if row['shape_status'] == 'eligible':
            sites[row['key'][0], row['key'][2]].append(row['iou'])
    p = .5 * (2 * counts['tp'] / (2 * counts['tp'] + counts['fp'] + counts['fn'])
              + 2 * counts['tn'] / (2 * counts['tn'] + counts['fp'] + counts['fn']))
    shape = sum(sum(values) / len(values) for values in sites.values()) / len(sites)
    if (counts != score['confusion'] or not np.isclose(p, score['presence'], rtol=0, atol=1e-14)
            or not np.isclose(shape, score['shape_diagnostic'], rtol=0, atol=1e-14)
            or score['shape'] is not None or score['total'] is not None
            or [len(rows), sum(counts.values()), sum(map(len, sites.values())), len(sites)] != [1348,1282,381,33]):
        raise ValueError('Pooled formula, denominator or NA policy differs')
