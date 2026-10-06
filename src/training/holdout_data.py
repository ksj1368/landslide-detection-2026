from collections import Counter
from pathlib import Path
import numpy as np
from shapely.geometry import box
from analysis.data import (
    discover, duplicate_images, file_snapshot, geometry, label_state,
)


from preprocessing.model_input import ROI_SIZE, prepare_sample


def _count_samples(samples):
    """라벨 상태·관심지·location·장면 수와 Shape 평가 가능한 양성 수를 집계한다.

    shape_evaluable은 유효한 양성 도형의 연속 면적이 10px² 이상인 경우다.
    작은 양성의 학습·존재 감독을 제외하는 기준이 아니며, site는 날짜별 중복을 뺀다.
    """
    states = Counter(sample["state"] for sample in samples)
    shape_count = sum(
        sample["state"] == "positive" and sample["gt_geometry"].is_valid
        and not sample["gt_geometry"].is_empty and sample["gt_geometry"].area >= 10
        for sample in samples
    )
    return {
        "samples": len(samples), "positive": states["positive"],
        "negative": states["negative"], "missing": states["missing"],
        "ignore": states["ignore"],
        "labeled": states["positive"] + states["negative"],
        "shape_evaluable": int(shape_count),
        "sites": len({(sample["key"][0], sample["key"][2]) for sample in samples}),
        "locations": len({sample["key"][0] for sample in samples}),
        "scenes": len({sample["key"][:2] for sample in samples}),
    }


def load_holdout_data(data_root, *, expected_invalid_gt=3, roi_size=ROI_SIZE,
                      origin_policy="centroid_floor"):
    """원본을 읽기 전용으로 점검하고 전처리 표본·제외 기록·집계·분할 근거를 반환한다.

    비어 있지 않지만 무효한 양성 도형은 존재 감독만 유지하며, 그 총수가
    고정된 expected_invalid_gt와 같아야 한다. 다른 감독 입력 결함은 오류로 알린다.
    원본 파일을 수정하지 않으며 읽기 전후 크기·수정시각을 비교한다.
    반환 snapshot은 전체 실행 후 같은 메타데이터를 다시 확인할 때도 사용한다.
    """
    if type(expected_invalid_gt) is not int or expected_invalid_gt < 0:
        raise ValueError("Expected invalid GT count must be a nonnegative integer")
    root = Path(data_root).resolve()
    candidates = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in (".json", ".npy")
    )
    if any(path.is_symlink() or not path.resolve().is_relative_to(root) for path in candidates):
        raise ValueError("Symlink or escaped input is not supported")
    before = file_snapshot(candidates, root)
    try:
        scenes, issues, files = discover(root)
    except Exception:
        raise ValueError("Input discovery failed; inspect schema locally") from None
    if any(issue["issue"] == "annotation_without_image" for issue in issues):
        raise ValueError("Annotation without matching image found")
    samples, exclusions, keys = [], [], set()
    roi_fractions, frame_fractions = [], []
    masks = Counter()
    for scene in scenes:
        try:
            raw = np.load(scene.image_path, allow_pickle=False, mmap_mode="r")
        except Exception:
            raise ValueError("Input array could not be loaded read-only") from None
        lookup = {str(record["site_id"]): record for record in scene.labels}
        for site in scene.sites:
            key = (scene.location, scene.date, str(site["site_id"]))
            if key in keys:
                raise ValueError("Duplicate site-date input key")
            keys.add(key)
            label = lookup.get(key[2])
            try:
                state = label_state(label)
                prior = geometry(site.get("prior_polygon", []))
                gt = geometry(label.get("polygon", label.get("polygons", []))) if label else geometry([])
            except Exception:
                raise ValueError("Unsupported label or polygon schema") from None
            exclude_segmentation = state == "positive" and not gt.is_empty and not gt.is_valid
            # 무효 양성 GT를 복구하거나 음성으로 바꾸지 않는다. 대상 키는 아래에 기록한다.
            try:
                sample = prepare_sample(
                    raw, prior, label, segmentation_supervision=not exclude_segmentation,
                    roi_size=roi_size, origin_policy=origin_policy,
                )
            except Exception:
                raise ValueError("Input preflight failed: invalid geometry, bands, or supervised raster") from None
            if state == "positive" and not exclude_segmentation:
                # 정상 양성 GT는 원영상과 실제 정수 crop 영역 안에 전부 있어야 한다.
                # GT를 맞추려고 crop을 이동시키지 않고, 포함되지 않으면 점검을 중단한다.
                x0, y0 = sample["origin"]
                roi = box(x0, y0, x0 + roi_size, y0 + roi_size)
                frame = box(0, 0, raw.shape[1], raw.shape[0])
                roi_fraction = float(gt.intersection(roi).area / gt.area)
                frame_fraction = float(gt.intersection(frame).area / gt.area)
                roi_fractions.append(roi_fraction)
                frame_fractions.append(frame_fraction)
                if not roi.covers(gt) or not frame.covers(gt):
                    raise ValueError("Positive GT extends outside the integer ROI or image frame")
            else:
                roi_fraction = frame_fraction = None
            if exclude_segmentation:
                exclusions.append({
                    "location_id": key[0], "date": key[1], "site_id": key[2],
                    "reason": "invalid_nonempty_positive_gt", "presence_supervised": True,
                    "segmentation_supervised": False,
                })
            # 관심지별 ROI의 합계라 겹치는 원영상 화소가 여러 번 집계될 수 있다.
            masks.update({
                "roi_pixels": roi_size * roi_size,
                "inside_pixels": int(sample["inside"].sum()),
                "padding_pixels": int((~sample["inside"]).sum()),
                "official_valid_pixels": int((sample["official_valid"] & sample["inside"]).sum()),
                "numeric_valid_pixels": int((sample["numeric_valid"] & sample["inside"]).sum()),
                "usable_pixels": int((sample["official_valid"] & sample["numeric_valid"] & sample["inside"]).sum()),
                "segmentation_supervised_pixels": int(sample["segmentation_valid"].sum()),
            })
            sample.update({
                "key": key, "group": scene.location, "gt_geometry": gt,
                "prior_geometry": prior, "image_path": scene.image_path,
                "segmentation_excluded": exclude_segmentation,
                "roi_gt_fraction": roi_fraction, "frame_gt_fraction": frame_fraction,
            })
            samples.append(sample)
        del raw
    if len(exclusions) != expected_invalid_gt:
        raise ValueError(
            f"Invalid positive GT count differs from frozen count: "
            f"expected {expected_invalid_gt}, observed {len(exclusions)}"
        )
    duplicates = duplicate_images(scenes)
    after = file_snapshot(candidates, root)
    if before != after:
        raise RuntimeError("Source file size or modification time changed during preflight")
    used = {str(path.relative_to(root)) for path in files}
    snapshot = [row for row in before if row["path"] in used]
    if len(snapshot) != len(files):
        raise RuntimeError("Discovered input was not included in the preflight snapshot")
    summary = _count_samples(samples)
    summary.update({
        "invalid_positive_gt": len(exclusions), "input_files": len(files),
        "duplicate_image_groups": len(duplicates), "discovery_issues": issues,
        "roi_mask_counts": dict(masks),
        "roi_count_note": "ROI totals can overlap and are not whole-scene pixel counts",
        "minimum_roi_gt_fraction": min(roi_fractions, default=None),
        "minimum_frame_gt_fraction": min(frame_fractions, default=None),
        "roi_geometry_note": "integer crop bounds; invalid GT coverage is NA",
        "source_unchanged_at_preflight": True,
    })
    return {
        "samples": samples, "input_files": files, "snapshot": snapshot,
        "summary": summary, "exclusions": exclusions, "duplicate_groups": duplicates,
    }
