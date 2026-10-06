"""진단·학습·추론이 공유하는 순수 NumPy 반사율 지수 계산."""

import numpy as np

INDEX_NAMES = ("NDVI", "NDWI", "EVI", "MSAVI2", "VARI")
DENOMINATOR_EPS = 1e-8


def spectral_indices(image, usable_mask, names=INDEX_NAMES):
    """반사율에서 지수를 계산하고 계산 불가를 0 값·False 마스크로 분리한다.

    issues는 unusable을 제외하면 usable 모집단 안의 서로 겹치지 않는 제외 사유다.
    계산 배열은 clipping하지 않는다. 음수 제곱근 인자도 자동으로 0에 맞추지 않는다.
    """
    if not names or any(name not in INDEX_NAMES for name in names):
        raise ValueError("Unsupported spectral index")
    source = np.asarray(image, dtype=np.float64)
    eligible = np.asarray(usable_mask, dtype=bool)
    if source.ndim != 3 or source.shape[-1] != 4 or source.shape[:2] != eligible.shape:
        raise ValueError("Expected H x W x 4 reflectance and aligned usable mask")
    red, green, blue, nir = np.moveaxis(source, -1, 0)
    with np.errstate(over="ignore", invalid="ignore"):
        numerators = {"NDVI": nir - red, "NDWI": green - nir,
                      "EVI": 2.5 * (nir - red), "VARI": green - red}
        denominators = {"NDVI": nir + red, "NDWI": green + nir,
                        "EVI": nir + 6 * red - 7.5 * blue + 1,
                        "VARI": green + red - blue}
        radicand = (2 * nir + 1) ** 2 - 8 * (nir - red)
    participants = {"NDVI": (0, 3), "NDWI": (1, 3), "EVI": (0, 2, 3),
                    "MSAVI2": (0, 3), "VARI": (0, 1, 2)}
    results = {}
    for name in names:
        finite_source = np.isfinite(source[..., participants[name]]).all(axis=-1)
        good = eligible & finite_source
        issues = {"unusable": int((~eligible).sum()),
                  "nonfinite_input": int((eligible & ~finite_source).sum()),
                  "near_zero_denominator": 0, "negative_radicand": 0,
                  "nonfinite_intermediate": 0, "nonfinite_result": 0}
        finite_intermediate = np.isfinite(radicand if name == "MSAVI2" else denominators[name])
        issues["nonfinite_intermediate"] = int((good & ~finite_intermediate).sum())
        good &= finite_intermediate
        values = np.zeros(eligible.shape, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if name == "MSAVI2":
                negative = good & (radicand < 0)
                issues["negative_radicand"] = int(negative.sum())
                good &= ~negative
                values[good] = (2 * nir[good] + 1 - np.sqrt(radicand[good])) / 2
            else:
                small = good & (np.abs(denominators[name]) <= DENOMINATOR_EPS)
                issues["near_zero_denominator"] = int(small.sum())
                good &= ~small
                values[good] = numerators[name][good] / denominators[name][good]
        finite_result = np.isfinite(values)
        issues["nonfinite_result"] = int((good & ~finite_result).sum())
        good &= finite_result
        values[~good] = 0
        results[name] = {"values": values, "valid": good, "issues": issues}
    return results
