"""Reversible four-band diagnostics; inputs already use the agreed physical scale."""

import numpy as np
from shapely.affinity import affine_transform


def _image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError("Expected H x W x 4 input")
    return image


def fit_statistics(images, image_keys=None):
    """Fit exact original-pixel statistics on only the images supplied by the caller.

    All four channels must be finite for a pixel to enter the common population.
    These full-EDA statistics are diagnostic only, not validation-safe training
    statistics. A future training caller must supply its training subset alone.
    """
    BANDS = ["R", "G", "B", "NIR"]

    chunks, image_count = [], 0
    for image in images:
        image = _image(image)
        chunks.append(image[np.isfinite(image).all(axis=-1)])
        image_count += 1
    keys = list(image_keys) if image_keys is not None else None
    if keys is not None and len(keys) != image_count:
        raise ValueError("image_keys must match the supplied image count")
    if not chunks or not any(len(chunk) for chunk in chunks):
        raise ValueError("Cannot fit statistics without finite four-band pixels")
    values = np.concatenate(chunks)
    quantiles = np.percentile(values, [2, 25, 50, 75, 98], axis=0)
    std = values.std(axis=0, dtype=np.float64)
    iqr = quantiles[3] - quantiles[1]
    return {
        "scope": "diagnostic_only",
        "bands": list(BANDS),
        "image_count": image_count,
        "image_keys": keys,
        "pixel_count": int(len(values)),
        "validity": "all_four_bands_finite",
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": std.tolist(),
        "median": quantiles[2].tolist(),
        "iqr": iqr.tolist(),
        "display_limits": quantiles[[0, 4]].tolist(),
        "degenerate_std": (std == 0).tolist(),
        "degenerate_iqr": (iqr == 0).tolist(),
        "degenerate_scale_replacement": 1.0,
    }


def _parameters(stats, method):
    if method == "baseline":
        return np.zeros(4), np.ones(4)
    if method not in ("zscore", "robust"):
        raise ValueError("method must be baseline, zscore, or robust")
    center, scale = ("mean", "std") if method == "zscore" else ("median", "iqr")
    center, scale = np.asarray(stats[center]), np.asarray(stats[scale])
    if center.shape != (4,) or scale.shape != (4,):
        raise ValueError("Normalization statistics must have four bands")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("Normalization statistics must be finite with nonnegative scales")
    return center, np.where(scale == 0, 1.0, scale)


def normalize(image, stats, method="baseline"):
    """Return unclipped FP32 values; missing pixels remain missing."""
    center, scale = _parameters(stats, method)
    return ((_image(image) - center) / scale).astype(np.float32)


def denormalize(image, stats, method="baseline"):
    center, scale = _parameters(stats, method)
    return (_image(image) * scale + center).astype(np.float32)


def quality_masks(image):
    """Diagnostic proxies, never a cloud classifier or supervision mask."""
    image = _image(image)
    valid = np.isfinite(image).all(axis=-1)
    # Replacing nonfinite values only prevents warnings in excluded pixels.
    rgb = np.where(valid[..., None], image[..., :3], 0)
    brightness = rgb.mean(axis=-1)
    return {
        "numeric_valid": valid,
        "bright": valid & (brightness > 0.35) & (rgb.std(axis=-1) < 0.05),
        "dark": valid & (brightness < 0.03),
    }


def transform_array(array, rotation=0, flip=False):
    """D4: counterclockwise quarter turns, then a left-right reflection."""
    if rotation not in range(4):
        raise ValueError("rotation must be 0, 1, 2, or 3 quarter turns")
    transformed = np.rot90(array, k=rotation, axes=(0, 1))
    return (np.flip(transformed, axis=1) if flip else transformed).copy()


def transform_aligned_arrays(arrays, rotation=0, flip=False):
    """Apply one D4 operation to named arrays sharing their first two axes.

    Rasterize polygons before calling this function. Arrays may have different
    channel counts and dtypes; every output is independent of the input storage.
    """
    if not arrays:
        raise ValueError("Expected at least one aligned array")
    values = {name: np.asarray(array) for name, array in arrays.items()}
    shapes = {array.shape[:2] for array in values.values()}
    if any(array.ndim not in (2, 3) for array in values.values()) or len(shapes) != 1:
        raise ValueError("Aligned arrays must be 2D or H x W x C with the same H/W")
    return {name: transform_array(array, rotation, flip) for name, array in values.items()}


def transform_geometry(geom, shape, rotation=0, flip=False):
    """Match transform_array using continuous pixel edges, without geometry repair.

    The input frame spans x=[0,width], y=[0,height]. A pixel center therefore
    maps to the center of the corresponding transformed pixel, including when
    rectangular frames swap dimensions. None, empty and invalid states survive.
    """
    if rotation not in range(4):
        raise ValueError("rotation must be 0, 1, 2, or 3 quarter turns")
    if geom is None or geom.is_empty:
        return geom
    height, width = shape[:2]
    matrices = (
        [1, 0, 0, 1, 0, 0],
        [0, 1, -1, 0, 0, width],
        [-1, 0, 0, -1, width, height],
        [0, -1, 1, 0, height, 0],
    )
    matrix = matrices[rotation]
    if flip:
        output_width = height if rotation % 2 else width
        matrix[0], matrix[1], matrix[4] = -matrix[0], -matrix[1], output_width - matrix[4]
    return affine_transform(geom, matrix)
