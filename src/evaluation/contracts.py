"""학습·검증 분할의 독립성과 예측 키의 누락·중복을 확인하는 공통 검사.

분할 자체는 training.holdout_data에서 정한다. 여기서는 전달된 분할을 검사하며
새 그룹을 고르거나 교차 검증 학습을 실행하지 않는다.
"""


def _keys(records):
    """각 표본의 (location, date, site) 키가 세 항목이며 중복 없는지 확인한다."""
    keys = [tuple(row["key"]) for row in records]
    if any(len(key) != 3 for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Sample keys must be unique location/date/site triples")
    return set(keys)


def validate_partition(train, validation):
    """학습·검증 사이에 표본, location, 연결 그룹이 겹치면 오류를 낸다.

    서로 다른 날짜라도 같은 location이면 분리하지 않는다. 연결 그룹 배정은
    호출자가 제공하며, 여기서는 학습·검증의 group 값이 겹치는지만 확인한다.
    """
    train_keys, validation_keys = _keys(train), _keys(validation)
    if not train_keys or not validation_keys:
        raise ValueError("Both partitions must be nonempty")
    if train_keys & validation_keys:
        raise ValueError("Sample leakage")
    if {k[0] for k in train_keys} & {k[0] for k in validation_keys}:
        raise ValueError("Location leakage")
    if {s["group"] for s in train} & {s["group"] for s in validation}:
        raise ValueError("Connected-group leakage")


def validate_oof(predictions, expected_keys):
    """OOF(각 표본을 학습하지 않은 모델의 검증 예측)에 필요한 키 집합을 검사한다.

    이 함수가 확인하는 것은 키의 중복·누락·추가뿐이다. 해당 모델이 실제로
    그 표본을 학습하지 않았는지는 별도의 분할·실행 기록으로 확인해야 한다.
    """
    expected = [tuple(key) for key in expected_keys]
    if len(set(expected)) != len(expected):
        raise ValueError("Duplicate expected OOF key")
    if _keys(predictions) != set(expected):
        raise ValueError("Missing or unexpected OOF keys")
