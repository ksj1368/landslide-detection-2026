"""이진 분할 마스크를 원영상 좌표의 연속 MultiPolygon 도형으로 변환한다.

inference.unet의 공통 후처리가 호출한다. 화소를 면적 1의 사각형으로 다뤄
연속 polygon 평가에 전달할 외곽선과 내부 구멍의 좌표를 만든다.
"""

import numpy as np
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


def masks_to_polygons(mask, origin, original_shape):
    """양성 화소를 합치고 ROI 원점을 더해 원영상 경계 안의 좌표를 반환한다.

    mask는 (높이, 너비) 이진 배열, origin은 원영상에서의 (x, y),
    original_shape는 원영상의 (높이, 너비)다. 반환값은 JSON에 넣을 수 있는
    [polygon][외곽선/구멍][점][x/y] 중첩 목록이며 산사태 없음은 []로 나타낸다.
    구멍·분리된 영역·한 화소 예측을 유지하고 prior로 대체하지 않는다.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, (0, 1)).all():
        raise ValueError("Expected a two-dimensional binary mask")
    if len(origin) != 2 or any(int(value) != value for value in origin):
        raise ValueError("Expected an integer (x, y) ROI origin")
    if len(original_shape) != 2 or any(int(value) != value or value <= 0 for value in original_shape):
        raise ValueError("Expected a positive (height, width) original shape")
    x0, y0 = origin
    height, width = original_shape
    cells = []
    for row, values in enumerate(mask):
        y = y0 + row
        if not 0 <= y < height:
            continue
        # 한 행에서 0↔1로 바뀌는 위치를 찾아 연속 양성 구간마다 사각형 하나를 만든다.
        # 각 화소를 따로 만드는 것과 같은 영역이며, 원영상 밖 구간은 잘라낸다.
        edges = np.flatnonzero(np.diff(np.pad(values.astype(np.int8), (1, 1))))
        for start, end in edges.reshape(-1, 2):
            left, right = max(0, x0 + int(start)), min(width, x0 + int(end))
            if left < right:
                cells.append(box(left, y, right, y + 1))
    if not cells:
        return []
    # 인접 구간을 합쳐 내부 구멍과 서로 떨어진 영역을 포함하는 도형을 만든다.
    merged = unary_union(cells)
    if merged.is_empty or not merged.is_valid:
        return []
    polygons = [merged] if isinstance(merged, Polygon) else list(merged.geoms)
    result = []
    for polygon in polygons:
        # 외곽선은 반시계, 내부 구멍은 시계 방향으로 좌표 순서를 맞춘다.
        polygon = orient(polygon, sign=1.0)
        result.append([
            [[float(x), float(y)] for x, y in ring.coords]
            for ring in (polygon.exterior, *polygon.interiors)
        ])
    return result
