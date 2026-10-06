import math
from evaluation.error_analysis import _area_bin, _binary_array
from evaluation.scorer import _prediction_geometry, score_predictions
from inference.unet import prediction_from_probabilities


FN_REASONS = ("head", "mask_threshold", "validity", "polygon_conversion")


def _mask_diagnostics(sample, mask, mask_threshold):
    """Decode once without the head gate; cache scalar mask/geometry diagnostics."""
    shape = sample["inside"].shape
    inside, official, numeric = (_binary_array(sample, field, shape)
                                 for field in ("inside", "official_valid", "numeric_valid"))
    usable = inside & official & numeric
    raw = mask >= mask_threshold
    kept = raw & usable
    bypass = prediction_from_probabilities(sample, mask, 1.0, mask_threshold=mask_threshold)
    predicted = _prediction_geometry(bypass["polygons"])
    gt = sample.get("gt_geometry")
    valid_gt = (sample["state"] == "positive" and gt is not None
                and gt.geom_type in ("Polygon", "MultiPolygon") and not gt.is_empty
                and gt.is_valid and math.isfinite(gt.area))
    target = _binary_array(sample, "target", (1, *shape))[0]
    segmentation = _binary_array(sample, "segmentation_valid", (1, *shape))[0]
    supervised_gt = target & segmentation
    gt_count = int(target.sum()) if valid_gt else None
    supervised_count = int(supervised_gt.sum()) if valid_gt else None
    overlap = float(gt.intersection(predicted).area) if valid_gt else None
    iou = (float(overlap / gt.union(predicted).area) if valid_gt and not predicted.is_empty
           else 0.0 if valid_gt else None)
    raw_count, kept_count = int(raw.sum()), int(kept.sum())
    values = {
        "raw_mask_pixels": raw_count, "usable_mask_pixels": kept_count,
        "mask_threshold_empty": raw_count == 0, "usable_mask_empty": kept_count == 0,
        "lost_to_validity": bool(raw_count and not kept_count),
        "polygon_conversion_empty": bool(kept_count and predicted.is_empty),
        "head_bypass_present": not predicted.is_empty,
        "head_bypass_iou": iou, "head_bypass_overlap_area": overlap,
        "excluded_official_pixels": int((raw & ~official).sum()),
        "excluded_numeric_pixels": int((raw & ~numeric).sum()),
        "excluded_padding_pixels": int((raw & ~inside).sum()),
        "excluded_union_pixels": int((raw & ~usable).sum()),
        "sequential_official_removed": int((raw & ~official).sum()),
        "sequential_numeric_removed": int((raw & official & ~numeric).sum()),
        "sequential_padding_removed": int((raw & official & numeric & ~inside).sum()),
        "gt_valid": bool(valid_gt), "gt_pixel_count": gt_count,
        "supervised_gt_pixel_count": supervised_count,
        "gt_pixel_probability_mean": float(mask[supervised_gt].mean()) if supervised_count else None,
        "gt_pixel_recall": float(kept[supervised_gt].mean()) if supervised_count else None,
        "gt_pixel_reason": None if supervised_count else (
            "no_supervised_gt_pixels" if valid_gt else "not_positive_or_invalid_gt"),
        "gt_overlap_reason": None if valid_gt else "not_positive_or_invalid_gt",
        "max_mask_probability": float(mask.max()), "usable_fraction": float(usable.mean()),
    }
    return bypass["polygons"], values


def _enrich_summary(summary, rows):
    summary = dict(summary)
    positives = summary["positive_count"]
    summary.update({
        "positive_recall": summary["confusion"]["tp"] / positives if positives else None,
        "small_positive_fn_count": sum(r["presence_outcome"] == "fn" and r["area_bin"] == "<10" for r in rows),
        "tp_zero_overlap_count": sum(r["tp_zero_overlap"] for r in rows),
        "fn_reasons": {reason: sum(r["fn_reason"] == reason for r in rows) for reason in FN_REASONS},
        "polygon_conversion_empty_count": sum(r["polygon_conversion_empty"] for r in rows),
        "raw_empty_mask_count": sum(r["mask_threshold_empty"] for r in rows),
        "usable_empty_mask_count": sum(r["usable_mask_empty"] for r in rows),
        "lost_to_validity_count": sum(r["lost_to_validity"] for r in rows),
        "site_count": len({(r["key"][0], r["key"][2]) for r in rows}),
        "raster_diagnostics": "gt_intersect_segmentation_valid_pixel_center_not_official_shape",
    })
    if sum(summary["fn_reasons"].values()) != summary["confusion"]["fn"]:
        raise ValueError("Mutually exclusive FN categories do not sum to FN")
    return summary


def _condition(samples, masks, heads, mask_cache, mode, presence_threshold, mask_threshold):
    predictions = []
    for sample, head, (polygons, _) in zip(samples, heads, mask_cache):
        head_positive = bool(float(head) >= presence_threshold)
        final = polygons if head_positive else []
        predictions.append({"key": sample["key"], "polygons": final,
                            "presence_probability": float(head), "presence_positive": head_positive,
                            "positive_without_polygon": bool(head_positive and not final)})
    scored = score_predictions(samples, predictions, presence_threshold=presence_threshold)
    rows = scored["rows"]
    for row, prediction, (_, cached) in zip(rows, predictions, mask_cache):
        head_positive = prediction["presence_positive"]
        fn_reason = None
        if row["presence_outcome"] == "fn":
            fn_reason = ("head" if not head_positive else "mask_threshold" if cached["mask_threshold_empty"]
                         else "validity" if cached["usable_mask_empty"] else "polygon_conversion")
        valid_gt = cached["gt_valid"]
        overlap = cached["head_bypass_overlap_area"] if head_positive else (0.0 if valid_gt else None)
        iou = cached["head_bypass_iou"] if head_positive else (0.0 if valid_gt else None)
        row.update(cached)
        row.update({
            "mode": mode, "presence_threshold": presence_threshold, "mask_threshold": mask_threshold,
            "area_bin": _area_bin(row), "fn_reason": fn_reason,
            "head_positive": head_positive, "head_failed": not head_positive,
            "head_mask_joint_failure": bool(not head_positive and cached["usable_mask_empty"]),
            "head_positive_without_polygon": bool(prediction["positive_without_polygon"]),
            "continuous_iou": iou, "gt_overlap_area": overlap,
            "tp_zero_overlap": bool(row["presence_outcome"] == "tp" and valid_gt and overlap == 0),
            "gt_overlapped": overlap > 0 if valid_gt else None,
            "final_gt_pixel_recall": cached["gt_pixel_recall"] if head_positive else (0.0 if cached["supervised_gt_pixel_count"] else None),
        })
    summary = _enrich_summary(scored["summary"], rows)
    context = {"mode": mode, "presence_threshold": presence_threshold, "mask_threshold": mask_threshold}
    summary.update(context)
    groups = {}
    for field, labels in (("regions", [r["key"][0] for r in rows]),
                          ("sizes", [r["area_bin"] for r in rows])):
        group_rows = []
        for label in sorted(set(labels)):
            indices = [i for i, value in enumerate(labels) if value == label]
            group = score_predictions([samples[i] for i in indices], [predictions[i] for i in indices],
                                      presence_threshold=presence_threshold)
            group_summary = _enrich_summary(group["summary"], [rows[i] for i in indices])
            # Regional or area scores are separate diagnostics, never pooled substitutes.
            group_summary.update(context, group=label)
            group_rows.append(group_summary)
        groups[field] = group_rows
    return {**context, "summary": summary, "rows": rows, "predictions": predictions, **groups}
