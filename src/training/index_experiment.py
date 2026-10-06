import hashlib
import json
import numpy as np


def canonical_sampling_trace(train, config):
    """E005의 독립 seed42 셔플·D4·마지막 작은 배치 알고리즘을 재현한다."""
    train=[s for s in train if s['presence_valid']]
    shuffle=np.random.default_rng(42); d4=np.random.default_rng(42)
    order=[];position=0;digest=hashlib.sha256();seen=0
    for _ in range(config['updates']):
        if position>=len(order):order=shuffle.permutation(len(train));position=0
        selected=order[position:position+config['batch_size']];position+=len(selected)
        for index in selected:
            transform=int(d4.integers(0,8))
            digest.update(json.dumps([list(train[index]['key']),transform],separators=(',',':')).encode())
        seen+=len(selected)
    return {'sha256':digest.hexdigest(),'samples_seen':seen,
            'basis':'E005 source algorithm replay; historical E005 has no per-key trace'}
