"""실험 전용 확률 등위선·polygon 단순화. 기존 화소 합집합 경로는 보존한다."""

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from skimage.measure import find_contours

from inference.unet import prediction_from_probabilities
from postprocessing.polygons import masks_to_polygons


def _geometry(polygons):
    geometry = MultiPolygon([Polygon(rings[0], rings[1:]) for rings in polygons])
    if not geometry.is_valid or not np.isfinite(geometry.area):
        raise ValueError("Invalid boundary geometry; no repair is applied")
    return geometry


def _parts(geometry):
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        return [part for member in geometry.geoms for part in _parts(member)]
    # Clipping can leave a point or line at a valid-area boundary, with zero area.
    return []


def _coordinates(geometry):
    if not geometry.is_valid or not np.isfinite(geometry.area):
        raise ValueError("Invalid boundary geometry; no repair is applied")
    result = []
    for polygon in _parts(geometry):
        polygon = orient(polygon, sign=1.0)
        result.append([[[float(x), float(y)] for x, y in ring.coords]
                       for ring in (polygon.exterior, *polygon.interiors)])
    # A collection can be valid although its polygon members overlap each other.
    _geometry(result)
    return result


def _contour_geometry(probabilities, usable, origin, original_shape, threshold):
    padded = np.pad(np.where(usable, probabilities, 0), 1)
    contours = find_contours(padded, level=threshold, fully_connected="low")
    rings, degenerate = [], 0
    for contour in contours:
        if not np.isfinite(contour).all():
            raise ValueError("Nonfinite probability contour")
        if len(contour) < 2:
            degenerate += 1
            continue
        if not np.array_equal(contour[0], contour[-1]):
            raise ValueError("Open probability contour after background padding")
        if len(contour) < 4:
            degenerate += 1
            continue
        # find_contours positions are padded pixel centers in (row, column).
        points = contour[:, ::-1] + np.asarray(origin) + .5 - 1
        ring = Polygon(points)
        if ring.area == 0 and np.linalg.matrix_rank(points - points[0]) <= 1:
            degenerate += 1
            continue
        if not ring.is_valid:
            raise ValueError("Invalid probability contour ring; no repair is applied")
        rings.append(ring)
    # Nesting depth distinguishes shells, holes, and islands inside holes without
    # depending on contour winding or using a geometric repair operation.
    parents = []
    for index, ring in enumerate(rings):
        containers = [other for other, candidate in enumerate(rings)
                      if other != index and candidate.area > ring.area
                      and candidate.covers(ring)]
        parents.append(min(containers, key=lambda other: rings[other].area)
                       if containers else None)
    depths = []
    for index in range(len(rings)):
        depth, parent = 0, parents[index]
        while parent is not None:
            depth += 1
            parent = parents[parent]
        depths.append(depth)
    polygons = [Polygon(ring.exterior.coords,
                        [hole.exterior.coords for child, hole in enumerate(rings)
                         if parents[child] == index])
                for index, ring in enumerate(rings) if depths[index] % 2 == 0]
    geometry = MultiPolygon(polygons)
    if not geometry.is_valid:
        raise ValueError("Invalid nested probability contours; no repair is applied")
    valid_area = _geometry(masks_to_polygons(usable, origin, original_shape))
    return geometry.intersection(valid_area), degenerate


def boundary_prediction(sample, mask_probability, head_probability, *, method,
                        tolerance=None, presence_threshold=.2, mask_threshold=.4):
    """저장 확률로 실험 polygon을 반환한다. 배열·좌표를 파일이나 로그에 쓰지 않는다.

    method는 probability_contour 또는 simplify다. simplify는 기존 화소 합집합을
    topology 보존으로 단순화하며 재클리핑이나 별도 영역 제거를 적용하지 않는다.
    등위선만 화소 중심 보간 후 원영상 안의 유효 화소 영역으로 제한한다.
    기존 추론의 네 필드와 contour_degenerate_ring_count 정수를 반환한다.
    임계값에 정확히 닿는 등위선이 면적을 만들지 못하면 산사태 없음으로 판정한다.
    """
    if method not in {"probability_contour", "simplify"}:
        raise ValueError("Unknown boundary processing method")
    if method == "simplify":
        if tolerance is None or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Simplification tolerance must be finite and positive")
    elif tolerance is not None:
        raise ValueError("Probability contours do not take a simplification tolerance")
    if any(not np.isfinite(value) or not 0 <= value <= 1
           for value in (presence_threshold, mask_threshold, head_probability)):
        raise ValueError("Probabilities and thresholds must be finite and within [0, 1]")
    probabilities = np.asarray(mask_probability)
    valid = [np.asarray(sample[field]) for field in ("inside", "official_valid", "numeric_valid")]
    if (probabilities.ndim != 2 or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or any(mask.shape != probabilities.shape or mask.dtype != bool for mask in valid)):
        raise ValueError("Invalid boundary probability array or validity mask")
    origin, original_shape = sample["origin"], sample["original_shape"]
    if (len(origin) != 2 or any(not np.isfinite(x) or int(x) != x for x in origin)
            or len(original_shape) != 2
            or any(not np.isfinite(x) or int(x) != x or x <= 0 for x in original_shape)):
        raise ValueError("Invalid ROI origin or original image shape")
    if method == "simplify":
        result = prediction_from_probabilities(sample, probabilities, head_probability,
                                               presence_threshold=presence_threshold,
                                               mask_threshold=mask_threshold)
        result["polygons"] = _coordinates(_geometry(result["polygons"]).simplify(
            tolerance, preserve_topology=True))
        result["positive_without_polygon"] = result["presence_positive"] and not result["polygons"]
        result["contour_degenerate_ring_count"] = 0
        return result
    is_present = float(head_probability) >= presence_threshold
    polygons, degenerate = [], 0
    if is_present:
        geometry, degenerate = _contour_geometry(probabilities, valid[0] & valid[1] & valid[2],
                                                 origin, original_shape, mask_threshold)
        polygons = _coordinates(geometry)
    return {"polygons": polygons, "presence_probability": float(head_probability),
            "presence_positive": is_present,
            "positive_without_polygon": is_present and not polygons,
            "contour_degenerate_ring_count": degenerate}
