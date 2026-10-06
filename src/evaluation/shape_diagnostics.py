import math
import numpy as np
from shapely import distance, line_interpolate_point
from shapely.geometry import MultiPolygon, Polygon


BOUNDARY_STEP_PX = .25


def _geometry_reason(geometry, name):
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        return f"{name}_nonpolygon"
    if geometry.is_empty:
        return f"{name}_empty"
    if not geometry.is_valid or not math.isfinite(geometry.area):
        return f"{name}_invalid"
    return None


def _parts(geometry):
    if _geometry_reason(geometry, "geometry"):
        return []
    return [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)


def directed_boundary_samples(reference, target):
    """Return the existing <=.25px boundary samples with one-based component IDs.

    Arrays stay local. Exterior and hole rings retain their original order and
    arc-length weights, so extracting them does not alter existing metrics.
    """
    distances, weights, components = [], [], []
    for component, polygon in enumerate(_parts(reference), 1):
        for ring in (polygon.exterior, *polygon.interiors):
            length = float(ring.length)
            if not length:
                continue
            count = math.ceil(length / BOUNDARY_STEP_PX)
            step = length / count
            # Midpoints avoid counting a closed ring's first vertex twice.
            points = line_interpolate_point(ring, (np.arange(count) + .5) * step)
            distances.extend(distance(points, target.boundary).tolist())
            weights.extend([step] * count)
            components.extend([component] * count)
    values, mass = np.asarray(distances), np.asarray(weights)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("A valid polygon must have finite, nonempty boundary distances")
    return values, mass, np.asarray(components, dtype=np.int64)


def _directed_boundary(reference, target):
    values, mass, _ = directed_boundary_samples(reference, target)
    order = np.argsort(values, kind="stable")
    position = np.searchsorted(np.cumsum(mass[order]), .95 * mass.sum(), side="left")
    p95 = float(values[order[min(position, len(values) - 1)]])
    return float(np.dot(values, mass)), float(mass.sum()), p95, len(values)


def boundary_metrics(reference, prediction):
    """Sample all exterior/hole rings at <=.25px and measure to exact boundaries.

    Mean pools both directed distance samples with their represented arc length.
    P95 is the larger of the two directed length-weighted inverse-ECDF P95s.
    Empty/invalid boundaries are NA, including a missing prediction's boundary.
    """
    reason = _geometry_reason(reference, "gt") or _geometry_reason(prediction, "pred")
    result = {"boundary_mean_px": None, "boundary_p95_px": None,
              "boundary_gt_to_pred_p95_px": None, "boundary_pred_to_gt_p95_px": None,
              "boundary_gt_samples": 0, "boundary_pred_samples": 0,
              "boundary_reason": reason}
    if reason:
        return result
    forward, backward = _directed_boundary(reference, prediction), _directed_boundary(prediction, reference)
    result.update(boundary_mean_px=(forward[0] + backward[0]) / (forward[1] + backward[1]),
                  boundary_p95_px=max(forward[2], backward[2]),
                  boundary_gt_to_pred_p95_px=forward[2], boundary_pred_to_gt_p95_px=backward[2],
                  boundary_gt_samples=forward[3], boundary_pred_samples=backward[3])
    return result


def geometry_metrics(gt, prediction):
    """Scalar continuous-area, topology and boundary metrics; no coordinate rows."""
    gt_reason, pred_reason = _geometry_reason(gt, "gt"), _geometry_reason(prediction, "pred")
    pred_empty = isinstance(prediction, (Polygon, MultiPolygon)) and prediction.is_empty
    gt_parts, pred_parts = _parts(gt), _parts(prediction)
    result = {"gt_area": float(gt.area) if not gt_reason else None,
              "pred_area": float(prediction.area) if not pred_reason or pred_empty else None,
              "gt_components": len(gt_parts) if not gt_reason else None,
              "gt_holes": sum(len(p.interiors) for p in gt_parts) if not gt_reason else None,
              "pred_components": len(pred_parts) if not pred_reason or pred_empty else None,
              "pred_holes": sum(len(p.interiors) for p in pred_parts) if not pred_reason or pred_empty else None,
              "intersection_area": None, "union_area": None, "iou": None,
              "under_fraction": None, "over_fraction": None,
              "coverage": None, "precision": None, "area_ratio": None,
              "geometry_reason": gt_reason or (None if pred_empty else pred_reason),
              **boundary_metrics(gt, prediction)}
    if result["geometry_reason"]:
        return result
    intersection = float(gt.intersection(prediction).area)
    union = float(gt.union(prediction).area)
    gt_area, pred_area = result["gt_area"], result["pred_area"]
    result.update(intersection_area=intersection, union_area=union, iou=intersection / union,
                  under_fraction=(gt_area - intersection) / union,
                  over_fraction=(pred_area - intersection) / union,
                  coverage=intersection / gt_area,
                  precision=intersection / pred_area if pred_area else None,
                  area_ratio=pred_area / gt_area)
    if not math.isclose(result["iou"] + result["under_fraction"] + result["over_fraction"],
                        1., abs_tol=1e-12, rel_tol=0):
        raise ValueError("Continuous IoU area partition does not sum to one")
    return result


def geometry_change(gt, reference, candidate):
    """Partition additions/removals inside/outside GT; empty predictions are valid."""
    old, new = geometry_metrics(gt, reference), geometry_metrics(gt, candidate)
    reason = old["geometry_reason"] or new["geometry_reason"]
    result = {"added_area": None, "removed_area": None,
              "added_inside_gt_area": None, "added_outside_gt_area": None,
              "removed_inside_gt_area": None, "removed_outside_gt_area": None,
              "iou_delta": None, "boundary_mean_delta_px": None, "boundary_p95_delta_px": None,
              "reference_boundary_reason": old["boundary_reason"],
              "candidate_boundary_reason": new["boundary_reason"], "change_reason": reason}
    if reason:
        return result
    added, removed = candidate.difference(reference), reference.difference(candidate)
    result.update(added_area=float(added.area), removed_area=float(removed.area),
                  added_inside_gt_area=float(added.intersection(gt).area),
                  added_outside_gt_area=float(added.difference(gt).area),
                  removed_inside_gt_area=float(removed.intersection(gt).area),
                  removed_outside_gt_area=float(removed.difference(gt).area),
                  iou_delta=new["iou"] - old["iou"])
    for field in ("boundary_mean", "boundary_p95"):
        if old[f"{field}_px"] is not None and new[f"{field}_px"] is not None:
            result[f"{field}_delta_px"] = new[f"{field}_px"] - old[f"{field}_px"]
    for name in ("added", "removed"):
        if not math.isclose(result[f"{name}_inside_gt_area"] + result[f"{name}_outside_gt_area"],
                            result[f"{name}_area"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Prediction-change area partition is inconsistent")
    return result
