from collections import Counter, defaultdict
import json
import numpy as np
from evaluation.scorer import score_predictions
from inference.unet import prediction_from_probabilities


_STATES = ("positive", "negative", "missing", "ignore")


_OUTCOMES = ("tp", "tn", "fp", "fn")


_FN_REASONS = ("FN_head_and_mask", "FN_head_only", "FN_mask_only")


_MEDIAN_FIELDS = (
    "presence_probability", "max_mask_probability", "gt_pixel_probability_mean",
    "gt_pixel_recall", "prior_gt_pixel_coverage", "usable_fraction", "gt_area",
)


def _binary_array(sample, field, shape):
    values = np.asarray(sample[field])
    if values.shape != shape or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{field} must have the expected shape and binary values")
    return values.astype(bool, copy=False)


def _validate_predictions(samples, predictions, masks, heads):
    if len(samples) != len(predictions):
        raise ValueError("Sample and prediction counts differ")
    keys = [tuple(sample["key"]) for sample in samples]
    predicted_keys = [tuple(prediction["key"]) for prediction in predictions]
    if (any(len(key) != 3 for key in keys + predicted_keys)
            or len(set(keys)) != len(keys)
            or len(set(predicted_keys)) != len(predicted_keys)
            or keys != predicted_keys):
        raise ValueError("Sample and prediction keys must be unique, complete and in identical order")
    masks = np.asarray(masks)
    heads = np.asarray(heads)
    if masks.shape != (len(samples), *samples[0]["inside"].shape) or heads.shape != (len(samples),):
        raise ValueError("Prediction arrays must match sample ROI shape and count")
    for values in (masks, heads):
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError("Prediction probabilities must be finite and within [0, 1]")
    return masks, heads


def _area_bin(row):
    if row["state"] != "positive":
        return "not_positive"
    area = row["gt_area"]
    if area is None:
        return "invalid"
    return "<10" if area < 10 else "10-30" if area < 30 else "30-100" if area < 100 else ">=100"


def analyze_predictions(samples, predictions, masks, heads, *,
                        presence_threshold=0.5, mask_threshold=0.5, polygon_method="pixel_union"):
    """Diagnose stored probabilities; defaults preserve the .5/.5 decode policy.

    GT pixel probability/recall use GT intersected with segmentation_valid.
    Prior coverage uses all rasterized GT pixels, before validity filtering.
    Both are diagnostic center-raster quantities, not the official polygon IoU.
    Location groups are for local reports; callers must sanitize external logs.
    """
    samples, predictions = list(samples), list(predictions)
    masks, heads = _validate_predictions(samples, predictions, masks, heads)
    result = score_predictions(samples, predictions, presence_threshold=presence_threshold)
    fn_reasons = Counter()
    by_area, by_location = defaultdict(Counter), defaultdict(Counter)
    for sample, prediction, mask, head, row in zip(samples, predictions, masks, heads, result["rows"]):
        decoded = prediction_from_probabilities(sample, mask, head,
                                                presence_threshold=presence_threshold,
                                                mask_threshold=mask_threshold,
                                                method=polygon_method)
        if json.dumps(decoded["polygons"], separators=(",", ":")) != json.dumps(prediction["polygons"], separators=(",", ":")):
            raise ValueError("Stored polygons differ from the shared baseline decode policy")
        recorded_head = prediction.get("presence_probability")
        if recorded_head is not None and float(recorded_head) != float(head):
            raise ValueError("Stored presence probability differs from the supplied head")
        usable = (
            _binary_array(sample, "official_valid", sample["inside"].shape)
            & _binary_array(sample, "numeric_valid", sample["inside"].shape)
            & _binary_array(sample, "inside", sample["inside"].shape)
        )
        prior = _binary_array(sample, "prior", sample["inside"].shape)
        gt = _binary_array(sample, "target", (1, *sample["inside"].shape))[0]
        segmentation_valid = _binary_array(sample, "segmentation_valid", (1, *sample["inside"].shape))[0]
        raw_mask = mask >= mask_threshold
        usable_mask = raw_mask & usable
        head_positive = bool(head >= presence_threshold)
        raw_count, usable_count = int(raw_mask.sum()), int(usable_mask.sum())
        fn_reason = None
        if row["presence_outcome"] == "fn":
            if not head_positive:
                fn_reason = "FN_head_only" if usable_count else "FN_head_and_mask"
            elif not usable_count:
                fn_reason = "FN_mask_only"
            else:
                raise ValueError("Usable positive prediction did not produce a polygon")
            fn_reasons[fn_reason] += 1
        gt_valid = row["state"] == "positive" and row["gt_area"] is not None
        gt_count = int(gt.sum()) if gt_valid else None
        supervised_gt = gt & segmentation_valid
        supervised_gt_count = int(supervised_gt.sum()) if gt_valid else None
        row.update({
            "fn_reason": fn_reason, "head_positive": head_positive,
            "raw_mask_pixels": raw_count, "usable_mask_pixels": usable_count,
            "lost_to_validity": bool(raw_count and not usable_count),
            "area_bin": _area_bin(row), "gt_valid": gt_valid,
            "gt_pixel_count": gt_count, "supervised_gt_pixel_count": supervised_gt_count,
            "gt_pixel_probability_mean": float(mask[supervised_gt].mean()) if supervised_gt_count else None,
            "gt_pixel_recall": float(usable_mask[supervised_gt].mean()) if supervised_gt_count else None,
            "prior_gt_pixel_coverage": float(prior[gt].mean()) if gt_count else None,
            "usable_fraction": float(usable.mean()),
            "presence_probability": float(head), "max_mask_probability": float(mask.max()),
        })
        for group in (by_area[row["area_bin"]], by_location[row["key"][0]]):
            group["count"] += 1
            if row["presence_outcome"] is not None:
                group[row["presence_outcome"]] += 1
    rows = result["rows"]
    medians = {}
    for outcome in _OUTCOMES:
        selected = [row for row in rows if row["presence_outcome"] == outcome]
        medians[outcome] = {}
        for field in _MEDIAN_FIELDS:
            values = [row[field] for row in selected if row[field] is not None]
            medians[outcome][field] = float(np.median(values)) if values else None
    result["summary"].update({
        "fn_reasons": {reason: fn_reasons[reason] for reason in _FN_REASONS},
        "raw_empty_mask_count": sum(row["raw_mask_pixels"] == 0 for row in rows),
        "usable_empty_mask_count": sum(row["usable_mask_pixels"] == 0 for row in rows),
        "lost_to_validity_count": sum(row["lost_to_validity"] for row in rows),
        "fn_raw_empty_mask_count": sum(row["presence_outcome"] == "fn" and row["raw_mask_pixels"] == 0 for row in rows),
        "fn_lost_to_validity_count": sum(row["presence_outcome"] == "fn" and row["lost_to_validity"] for row in rows),
        "by_area": {group: {field: counts[field] for field in ("count", *_OUTCOMES)} for group, counts in by_area.items()},
        "by_location": {group: {field: counts[field] for field in ("count", *_OUTCOMES)} for group, counts in by_location.items()},
        "medians_by_outcome": medians,
        "prediction_policy_verified": True,
        "raster_diagnostics": "pixel_center_convention_not_official_scorer",
        "gt_probability_denominator": "gt_intersect_segmentation_valid",
        "prior_coverage_denominator": "all_rasterized_gt_pixels",
    })
    return result
