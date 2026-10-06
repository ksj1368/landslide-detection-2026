from evaluation.contracts import validate_partition
from training.holdout_data import _count_samples


def frozen_partition(samples, manifest):
    """기존 분할 명세의 location 배정을 적용해 (학습, 검증) 표본을 반환한다.

    새 분할을 선정·저장하지 않는다. location 구성과 연결 그룹의 독립성,
    라벨 상태별 수를 기존 명세와 대조하고 표본은 key 순서로 정렬한다.
    """
    if manifest["seed"] != 42:
        raise ValueError("Frozen split seed differs")
    train_locations = set(manifest["train_locations"])
    valid_locations = set(manifest["validation_locations"])
    if train_locations & valid_locations:
        raise ValueError("Frozen location overlap")
    if {s["key"][0] for s in samples} != train_locations | valid_locations:
        raise ValueError("Frozen location membership differs")
    groups = {loc: group for group, locations in manifest["connected_groups"].items() for loc in locations}
    train, validation = [], []
    for sample in sorted(samples, key=lambda s: tuple(s["key"])):
        target = train if sample["key"][0] in train_locations else validation
        target.append({**sample, "group": groups[sample["key"][0]]})
    validate_partition(train, validation)
    for name, records in (("all", samples), ("train", train), ("validation", validation)):
        if _count_samples(records) != manifest["counts"][name]:
            raise ValueError("Frozen sample counts or supervision states differ")
    return train, validation
