"""학습과 추론에서 함께 쓰는 관심지별 원해상도 U-Net 입력을 만든다.

ROI는 관심지 주변에서 잘라낸 입력 영역이며, prior는 정답과 별개로 주어진
관심지 도형이다. 도형을 마스크로 바꿀 때는 기존 화소 중심 포함 규약을 쓴다.
이 규약이 아직 확보하지 못한 공식 scorer와 호환된다는 뜻은 아니다.
"""

import math

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from analysis.data import geometry, label_state, pixel_mask
from preprocessing.transforms import normalize, transform_aligned_arrays
from preprocessing.spectral import spectral_indices


ROI_SIZE = 96


def _polygon(value):
    """이미 만든 Shapely 도형은 재사용하고, 좌표 목록이면 도형으로 변환한다."""
    return value if isinstance(value, (Polygon, MultiPolygon)) else geometry(value)


def roi_parameters(config):
    """고정된 ROI 조합을 확인하고, 과거 checkpoint의 누락값은 E001로 해석한다.

    E001은 96px 중심 crop, E002는 96px 안쪽 이동, E003은 128px 중심 crop이다.
    E006은 128px 안쪽 이동으로 A와 B를 함께 적용한다.
    origin_policy가 없는 과거 설정도 중심 crop을 유지한다.
    """
    size = config.get("roi_size", 96)
    policy = config.get("origin_policy", "centroid_floor")
    if type(size) is not int or (size, policy) not in (
        (96, "centroid_floor"), (96, "shift_inside"),
        (128, "centroid_floor"), (128, "shift_inside")
    ):
        raise ValueError("Unsupported ROI size/origin policy")
    return {"roi_size": size, "origin_policy": policy}


def _crop(array, origin, roi_size=ROI_SIZE):
    """ROI를 새 배열로 복사하고 영상 밖은 0으로 채운다(zero padding).

    origin은 원영상의 (x, y)이지만 NumPy 배열은 [행(y), 열(x)] 순서다.
    반환 배열은 원본과 저장 공간을 공유하지 않아 이후 변환이 원본을 바꾸지 않는다.
    """
    x0, y0 = origin
    height, width = array.shape[:2]
    output = np.zeros((roi_size, roi_size, *array.shape[2:]), dtype=array.dtype)
    left, top = max(x0, 0), max(y0, 0)
    right, bottom = min(x0 + roi_size, width), min(y0 + roi_size, height)
    if left < right and top < bottom:
        output[top - y0:bottom - y0, left - x0:right - x0] = array[top:bottom, left:right]
    return output


def prepare_sample(raw, prior_polygon, label=None, *, rotation=0, flip=False,
                   segmentation_supervision=True, roi_size=ROI_SIZE, origin_policy="centroid_floor",
                   input_index="none", normalization="none", normalization_statistics=None):
    """공통 입력·정답·유효성 마스크를 같은 ROI 좌표에 맞춰 반환한다.

    raw는 (높이, 너비, 4)의 R/G/B/NIR 원값이며 여기서 10000으로 나눈다.
    출력 input은 FP32 (5·6·7, S, S), target·segmentation_valid는 (1, S, S),
    presence·presence_valid는 각각 표본 하나의 값이다. S는 ROI 한 변 길이다.

    label의 visible/ignore/polygon으로 감독 여부를 정한다. 미라벨·ignore는
    손실에서 제외하며, 명시적 음성은 해당 관심지에 대한 음성만 뜻한다.
    segmentation_supervision=False이면 양성 존재 감독을 유지하고 분할만 제외한다.
    제외할 표본의 판단·기록은 호출자가 담당한다.
    origin은 항상 증강 전 ROI의 원점이다. 추론에서는 회전·반전을 적용하지 않는다.
    """
    roi_parameters({"roi_size": roi_size, "origin_policy": origin_policy})
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[2] != 4 or min(raw.shape[:2]) < 1:
        raise ValueError("Expected nonempty H x W x 4 input")
    if raw.dtype.kind not in "uif":
        raise ValueError("Expected real numeric bands")
    prior = _polygon(prior_polygon)
    if prior.is_empty or not prior.is_valid:
        raise ValueError("Prior must be a nonempty valid polygon")
    cx, cy = prior.centroid.coords[0]
    # 정답 GT를 보지 않고 prior의 무게중심으로 crop 원점을 정한다.
    origin = (math.floor(cx - roi_size / 2), math.floor(cy - roi_size / 2))
    if origin_policy == "shift_inside":
        # 영상 크기가 ROI 이상인 축만 안쪽으로 이동한다. 작은 축은 중심 crop·padding을 유지한다.
        origin = tuple(max(0, min(value, extent - roi_size)) if extent >= roi_size else value
                       for value, extent in zip(origin, (raw.shape[1], raw.shape[0])))
    shape = raw.shape[:2]
    # 공식 유효화소 조건(R·NIR > 0)과 수치 유효성(모든 밴드가 유한함)은 별개다.
    # 둘 다 만족한 화소만 입력에 쓴다. 추가 정규화는 지수 계산 뒤 네 밴드에만 적용한다.
    official_valid = (raw[..., 0] > 0) & (raw[..., 3] > 0)
    numeric_valid = np.isfinite(raw).all(axis=-1)
    usable = official_valid & numeric_valid
    image = (np.where(usable[..., None], raw, 0) / 10000).astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Scaled input cannot be represented as finite FP32")
    state = label_state(label)
    gt_mask = np.zeros(shape, dtype=bool)
    if state in ("positive", "negative"):
        gt = _polygon(label.get("polygon", label.get("polygons", [])))
        if state == "positive" and segmentation_supervision:
            if gt.is_empty or not gt.is_valid:
                raise ValueError("Positive GT must be a nonempty valid polygon")
            gt_mask = pixel_mask(gt, shape)
        elif state == "negative" and not gt.is_empty:
            raise ValueError("Negative label contradicts a nonempty GT polygon")

    # 영상·prior·GT·유효성 마스크를 함께 자르고 D4(회전·반전) 변환한다.
    # 도형을 다시 래스터화하지 않고 배열을 같이 옮겨 화소별 정렬을 유지한다.
    # inside는 원영상 안쪽 여부다. ROI 안에서 원영상 밖을 채운 0 padding과 실제 화소를 구분한다.
    arrays = transform_aligned_arrays({
        name: _crop(array, origin, roi_size) for name, array in {
            "image": image,
            "prior": pixel_mask(prior, shape),
            "gt": gt_mask,
            "official_valid": official_valid,
            "numeric_valid": numeric_valid,
            "inside": np.ones(shape, dtype=bool),
        }.items()
    }, rotation, flip)
    usable_roi = arrays["official_valid"] & arrays["numeric_valid"] & arrays["inside"]
    if state == "positive" and segmentation_supervision and not (arrays["gt"] & usable_roi).any():
        raise ValueError("Positive GT has no supervised raster pixel in this ROI")
    supervised = state in ("positive", "negative")
    # prior는 정답 채널이 아니라 대상 관심지를 알려주는 다섯 번째 입력 채널이다.
    model_input = np.concatenate((arrays["image"], arrays["prior"][..., None]), axis=-1)
    # HWC → CHW: PyTorch 합성곱이 받는 채널 우선 형태로 바꾸고 연속 메모리에 둔다.
    # 감독 여부는 별도 mask로 전달하므로 미라벨의 presence=0을 음성 정답으로 쓰지 않는다.
    sample = {
        "input": np.ascontiguousarray(model_input.transpose(2, 0, 1), dtype=np.float32),
        "target": arrays["gt"][None].astype(np.float32),
        "segmentation_valid": (usable_roi & supervised & segmentation_supervision)[None],
        "presence": np.float32(state == "positive"),
        "presence_valid": supervised,
        "state": state,
        "origin": origin,
        "original_shape": shape,
        "roi_size": roi_size, "origin_policy": origin_policy,
        **{name: value for name, value in arrays.items() if name != "image"},
    }

    return normalize_sample(append_input_index(sample, input_index), normalization,
                            normalization_statistics)


INDEX_CALCULATION = {"version": "reflectance_fp32_to_float64_v1", "denominator_epsilon": 1e-8,
                     "invalid_policy": "error_on_usable", "output_dtype": "float32", "clipping": False}
BASE_INPUT_BANDS = ["R", "G", "B", "NIR", "prior"]


def index_configuration(name="none"):
    """지수 종류·채널 순서·계산 계약을 한 곳에서 정의한다."""
    if name not in ("none", "NDVI", "NDWI", "MSAVI2", "NDVI+MSAVI2", "NDVI+NDWI"):
        raise ValueError("Unsupported model input index")
    return {"input_index": name, "input_bands": BASE_INPUT_BANDS + ([] if name == "none" else name.split("+")),
            **({"index_calculation": dict(INDEX_CALCULATION)} if name != "none" else {})}


def input_parameters(config):
    """과거 checkpoint는 5채널을 유지하며 ROI와 지수 설정을 전달한다."""
    name = config.get("input_index", "none")
    index_configuration(name)
    return {**roi_parameters(config), "input_index": name, **normalization_configuration(config)}


def normalization_configuration(config):
    """추론에 필요한 네 밴드 통계만 복사한다. 0 scale을 자동 대체하지 않는다."""
    method = config.get("normalization", "none")
    stats = config.get("normalization_statistics")
    if method == "none":
        if stats is not None:
            raise ValueError("Unnormalized input must not carry normalization statistics")
        return {"normalization": "none"}
    if method not in ("zscore", "robust"):
        raise ValueError("Unsupported model input normalization")
    if not isinstance(stats, dict) or stats.get("bands") != BASE_INPUT_BANDS[:4]:
        raise ValueError("Normalization statistics must use R/G/B/NIR band order")
    center, scale = ("mean", "std") if method == "zscore" else ("median", "iqr")
    selected = {"bands": BASE_INPUT_BANDS[:4]}
    for name in (center, scale):
        values = np.asarray(stats.get(name), dtype=np.float64)
        if (values.shape != (4,) or not np.isfinite(values).all()
                or (name == scale and (values <= 0).any())):
            raise ValueError("Normalization requires four finite centers and positive scales")
        selected[name] = values.tolist()
    return {"normalization": method, "normalization_statistics": selected}


def normalize_sample(sample, normalization="none", normalization_statistics=None):
    """이미 계산한 prior·지수를 보존하고 RGBNIR만 새 배열에서 정규화한다."""
    config = normalization_configuration({"normalization": normalization,
                                          "normalization_statistics": normalization_statistics})
    if sample.get("normalization", "none") != "none":
        raise ValueError("Sample normalization must be applied exactly once")
    if normalization == "none":
        return {**sample, **config}
    eligible = sample["inside"] & sample["official_valid"] & sample["numeric_valid"]
    with np.errstate(over="ignore", invalid="ignore"):
        image = normalize(sample["input"][:4].transpose(1, 2, 0),
                          config["normalization_statistics"], normalization)
    image[~eligible] = 0
    if not np.isfinite(image).all():
        raise ValueError("Normalized input cannot be represented as finite FP32")
    model_input = sample["input"].copy()
    model_input[:4] = image.transpose(2, 0, 1)
    return {**sample, "input": model_input, **config}


class InvalidIndexError(ValueError):
    def __init__(self, name, issues):
        self.index, self.issues = name, dict(issues)
        super().__init__(f"{name}: unusable derived values in otherwise usable pixels")


def append_input_index(sample, name="none", *, strict=True):
    """기존 5채널·감독을 보존하고 지수만 추가한다. strict=False는 전수 오류 집계용이다."""
    index_configuration(name)
    if sample.get("normalization", "none") != "none":
        raise ValueError("Input index must be calculated before normalization")
    if sample["input"].shape[0] != 5:
        raise ValueError("Index must be appended exactly once to the five original channels")
    if name == "none":
        return {**sample, "input_index": "none"}
    eligible = sample["inside"] & sample["official_valid"] & sample["numeric_valid"]
    names = index_configuration(name)["input_bands"][5:]
    derived = spectral_indices(sample["input"][:4].transpose(1, 2, 0), eligible, names)
    channels, valid_by_name, issues_by_name = [], {}, {}
    for index in names:
        item = derived[index]
        with np.errstate(over="ignore", invalid="ignore"):
            values = item["values"].astype(np.float32)
        overflow = item["valid"] & ~np.isfinite(values)
        issues = dict(item["issues"])
        issues["nonfinite_result"] += int(overflow.sum())
        valid = item["valid"] & ~overflow
        values[~valid] = 0
        if strict and (eligible & ~valid).any():
            raise InvalidIndexError(index, issues)
        channels.append(values[None])
        valid_by_name[index], issues_by_name[index] = valid, issues
    aggregate = {key: sum(issues[key] for issues in issues_by_name.values())
                 for key in next(iter(issues_by_name.values()))}
    return {**sample, "input": np.concatenate([sample["input"], *channels], axis=0),
            "input_index": name, "index_valid": np.logical_and.reduce(list(valid_by_name.values())),
            "index_issues": aggregate, "index_valid_by_name": valid_by_name,
            "index_issues_by_name": issues_by_name}
