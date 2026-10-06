"""문서에 근거해 연속 polygon의 Presence·Shape를 계산하는 내부 scorer.

Holdout·ROI 비교에서 쓰지만 공식 구현과의 호환성은 미확인이다.
Presence·Shape의 산출 조건을 충족하지 못하면 사유와 함께 None(보고서의 NA)을
반환한다. 잘못된 예측·키는 오류를 낸다.
"""

from collections import Counter, defaultdict
import math

from shapely.geometry import MultiPolygon, Polygon, shape


SCORER_ID = "scorer-doc-v1"


def _index(records, description):
    """표본·예측을 유일한 (location, date, site) 키로 연결할 색인을 만든다."""
    indexed = {}
    for record in records:
        key = tuple(record["key"])
        if len(key) != 3 or key in indexed:
            raise ValueError(f"{description} keys must be unique location/date/site triples")
        indexed[key] = record
    return indexed


def _prediction_geometry(polygons):
    """직렬화된 예측을 Shapely 도형으로 읽되 잘못된 도형을 자동 복구하지 않는다."""
    if not isinstance(polygons, (list, tuple)):
        raise ValueError("Prediction polygons must be MultiPolygon coordinates or []")
    if not polygons:
        return MultiPolygon()
    try:
        geometry = shape({"type": "MultiPolygon", "coordinates": polygons})
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("Malformed prediction polygon coordinates") from exc
    if geometry.is_empty or not geometry.is_valid or not math.isfinite(geometry.area):
        raise ValueError("Prediction polygons must be nonempty and valid; no repair is applied")
    return geometry


def score_predictions(samples, predictions, *, presence_threshold=0.5):
    """같은 key를 가진 정답·예측을 평가해 summary와 표본별 rows를 반환한다.

    samples는 state와 원본 Shapely gt_geometry를, predictions는 MultiPolygon
    좌표 목록 polygons와 선택적인 presence_probability를 제공한다.
    미라벨·ignore도 예측은 하나씩 있어야 하지만 정답 기반 점수에는 넣지 않는다.
    반환 rows에는 좌표를 담지 않는다. 잘못된 양성 GT도 Presence의 양성을 유지하며
    전체 Shape에서 조용히 제외하지 않고 산출 불가 사유로 남긴다.
    """
    if not math.isfinite(presence_threshold) or not 0 <= presence_threshold <= 1:
        raise ValueError("Presence threshold must be finite and within [0, 1]")
    samples = _index(samples, "Sample")
    predictions = _index(predictions, "Prediction")
    if samples.keys() != predictions.keys():
        raise ValueError("Missing or unexpected prediction keys")
    counts = Counter()
    confusion = dict(tp=0, tn=0, fp=0, fn=0)
    site_ious = defaultdict(list)
    rows = []
    for key, sample in samples.items():
        state = sample["state"]
        if state not in {"positive", "negative", "missing", "ignore"}:
            raise ValueError("Unknown sample label state")
        counts[state] += 1
        prediction = predictions[key]
        predicted_geometry = _prediction_geometry(prediction["polygons"])
        # 최종 존재 판정은 head 확률이 아니라 제출 형태의 polygon이 비었는지로 정한다.
        predicted_present = not predicted_geometry.is_empty
        counts["predicted_present"] += predicted_present
        probability = prediction.get("presence_probability")
        if probability is not None:
            probability = float(probability)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Presence probability must be finite and between zero and one")
            # 이 인자는 진단 건수에만 관여하며 Presence/Shape 산식은 바꾸지 않는다.
            counts["head_positive_without_polygon"] += probability >= presence_threshold and not predicted_present
        gt_present = state == "positive" if state in {"positive", "negative"} else None
        outcome = None
        if gt_present is not None:
            outcome = ("tp" if predicted_present else "fn") if gt_present else ("fp" if predicted_present else "tn")
            confusion[outcome] += 1
        iou = gt_area = None
        shape_status = "not_positive" if state == "negative" else "not_supervised"
        if state == "positive":
            gt = sample.get("gt_geometry")
            if not isinstance(gt, (Polygon, MultiPolygon)) or gt.is_empty:
                shape_status = "empty_or_nonpolygon_gt"
                counts["invalid_gt"] += 1
                counts["empty_or_nonpolygon_gt"] += 1
            elif not gt.is_valid or not math.isfinite(gt.area):
                shape_status = "invalid_gt"
                counts["invalid_gt"] += 1
            else:
                gt_area = float(gt.area)
                if gt_area < 10:
                    # 작은 양성은 Shape 대상에서만 제외한다. 위의 Presence 양성은 유지된다.
                    shape_status = "small_positive"
                    counts["small_positive"] += 1
                else:
                    shape_status = "eligible"
                    # 래스터 화소 수가 아닌 원좌표 도형의 교집합/합집합 면적으로 IoU를 계산한다.
                    iou = 0.0 if not predicted_present else float(gt.intersection(predicted_geometry).area / gt.union(predicted_geometry).area)
                    if not math.isfinite(iou) or not 0 <= iou <= 1:
                        raise ValueError("Continuous polygon IoU is not finite or outside [0, 1]")
                    site_ious[(key[0], key[2])].append(iou)
                    counts["shape_eligible"] += 1
        rows.append({
            "key": key, "state": state, "gt_present": gt_present,
            "predicted_present": predicted_present, "presence_probability": probability,
            "presence_outcome": outcome, "gt_area": gt_area, "iou": iou,
            "shape_status": shape_status,
        })

    presence = None
    presence_reason = None
    if not counts["positive"] or not counts["negative"]:
        presence_reason = "no_labeled_samples" if not (counts["positive"] + counts["negative"]) else "single_gt_class"
    else:
        tp, tn, fp, fn = (confusion[name] for name in ("tp", "tn", "fp", "fn"))
        # 양성 F1과 음성 F1을 동등하게 평균한 macro-F1이다.
        presence = .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))
    # 같은 site의 평가 가능한 날짜들을 먼저 평균하고, 다시 site별로 동등하게 평균한다.
    # 관측 날짜가 많은 site가 전체 Shape를 지배하지 않도록 하는 집계 순서다.
    site_means = [sum(values) / len(values) for values in site_ious.values()]
    shape_diagnostic = sum(site_means) / len(site_means) if site_means else None
    shape_reason = "invalid_positive_gt" if counts["invalid_gt"] else ("no_shape_eligible_samples" if not site_means else None)
    # 유효한 부분집합의 진단값이 있어도 무효 양성 GT가 있으면 전체 Shape는 None이다.
    # Total도 Presence·Shape가 모두 정의된 경우에만 계산하며 공식 점수 인증은 아니다.
    shape_score = shape_diagnostic if shape_reason is None else None
    total = .6 * presence + .4 * shape_score if presence is not None and shape_score is not None else None
    summary = {
        "scorer_id": SCORER_ID, "official_compatibility": "not_verified",
        "status": "complete" if total is not None else "diagnostic_incomplete",
        "presence": presence, "shape": shape_score, "total": total,
        "shape_diagnostic": shape_diagnostic,
        "presence_reason": presence_reason, "shape_reason": shape_reason,
        "shape_diagnostic_reason": None if site_means else "no_shape_eligible_samples",
        "total_reason": [reason for reason in (presence_reason, shape_reason) if reason] or None,
        "input_count": len(samples), "labeled_count": counts["positive"] + counts["negative"],
        "positive_count": counts["positive"], "negative_count": counts["negative"],
        "missing_count": counts["missing"], "ignored_count": counts["ignore"],
        "invalid_gt_count": counts["invalid_gt"],
        "empty_or_nonpolygon_gt_count": counts["empty_or_nonpolygon_gt"],
        "small_positive_count": counts["small_positive"],
        "shape_eligible_count": counts["shape_eligible"], "shape_site_count": len(site_ious),
        "predicted_present_count": counts["predicted_present"],
        "head_positive_without_polygon_count": counts["head_positive_without_polygon"],
        "confusion": confusion,
    }
    return {"summary": summary, "rows": rows}
