"""Read-only competition input and explicit site-date supervision states."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from shapely import contains_xy
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.validation import explain_validity


@dataclass
class Scene:
    location: str
    date: str
    image_path: Path
    labels: list
    sites: list

    @property
    def key(self):
        return f"{self.location}/{self.date}"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def discover(data_root):
    """Only the specified input root is traversed."""
    root = Path(data_root).resolve()
    scenes, issues, input_files = [], [], set()
    site_files = sorted(p for p in root.rglob("sites.json") if not p.is_symlink())
    if not site_files:
        raise ValueError("No site definitions found inside configured input root")
    locations = set()
    for site_path in site_files:
        folder = site_path.parent
        if not folder.resolve().is_relative_to(root):
            raise ValueError("Input escapes configured root")
        location = folder.name
        if location in locations:
            raise ValueError("Duplicate location directory identifiers")
        locations.add(location)
        sites = read_json(site_path)
        if not isinstance(sites, list):
            raise ValueError("Site definition must be a list")
        ids = [str(s["site_id"]) for s in sites]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate site identifiers within a location")
        input_files.add(site_path)
        label_paths = {}
        for path in sorted(folder.rglob("*.json")):
            if path == site_path or path.is_symlink():
                continue
            if path.stem in label_paths:
                raise ValueError("Multiple annotation files for the same date")
            label_paths[path.stem] = path
        dates = set()
        for path in sorted(folder.rglob("*.npy")):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Symlink or escaped array path is not supported")
            datetime.strptime(path.stem, "%Y%m%d")
            if path.stem in dates:
                raise ValueError("Duplicate image date within a location")
            dates.add(path.stem)
            label_path = label_paths.get(path.stem)
            labels = read_json(label_path) if label_path else []
            if not isinstance(labels, list):
                raise ValueError("Annotations must be a list")
            label_ids = [str(r["site_id"]) for r in labels]
            if len(label_ids) != len(set(label_ids)):
                raise ValueError("Duplicate annotation keys")
            if not set(label_ids).issubset(ids):
                raise ValueError("Annotation references an unknown site")
            input_files.add(path)
            if label_path:
                input_files.add(label_path)
            else:
                issues.append({"location_id": location, "date": path.stem,
                               "issue": "missing_annotation_file"})
            scenes.append(Scene(location, path.stem, path, labels, sites))
        for date in set(label_paths) - dates:
            issues.append({"location_id": location, "date": date,
                           "issue": "annotation_without_image"})
    if not scenes:
        raise ValueError("No input arrays found")
    return scenes, issues, sorted(input_files)


def load_image(scene, divisor):
    raw = np.load(scene.image_path, allow_pickle=False, mmap_mode="r")
    if raw.ndim != 3 or raw.shape[2] != 4:
        raise ValueError("Expected H x W x 4 input")
    if raw.dtype.kind not in "uif":
        raise ValueError("Expected numeric input")
    return np.asarray(raw, dtype=np.float32) / divisor


def geometry(value):
    """Accept a ring, Polygon rings, or MultiPolygon nesting without repair."""
    if value is None or len(value) == 0:
        return MultiPolygon([])
    item = value
    depth = 0
    while isinstance(item, (list, tuple)) and item:
        depth += 1
        item = item[0]
    if depth == 2:
        return Polygon(value)
    if depth == 3:
        return Polygon(value[0], value[1:])
    if depth == 4:
        return MultiPolygon([Polygon(rings[0], rings[1:]) for rings in value if rings])
    raise ValueError("Unsupported polygon nesting")


def label_state(record):
    if record is None:
        return "missing"
    if record.get("ignore", False):
        return "ignore"
    visible = record.get("visible")
    if visible == 1:
        return "positive"
    if visible == 0:
        return "negative"
    raise ValueError("Unsupported visibility state")


def roi_geometry(prior, size):
    if prior.is_empty or not prior.is_valid:
        return None
    x, y = prior.centroid.coords[0]
    return box(x - size / 2, y - size / 2, x + size / 2, y + size / 2)


def pixel_mask(geom, shape):
    """EDA-only pixel-center convention, not the official polygon scorer."""
    if geom is None or geom.is_empty or not geom.is_valid:
        return np.zeros(shape, dtype=bool)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return contains_xy(geom, xx + 0.5, yy + 0.5)


def site_records(scene, image_shape, roi_size):
    lookup = {str(r["site_id"]): r for r in scene.labels}
    records = []
    for site in scene.sites:
        site_id = str(site["site_id"])
        label = lookup.get(site_id)
        state = label_state(label)
        prior = geometry(site.get("prior_polygon", []))
        gt = geometry(label.get("polygon", label.get("polygons", []))) if label else geometry([])
        roi = roi_geometry(prior, roi_size)
        area = float(gt.area) if gt.is_valid and not gt.is_empty else None
        record = {
            "location_id": scene.location, "date": scene.date, "site_id": site_id,
            "label_state": state, "presence_supervised": state in ("positive", "negative"),
            "shape_supervised": state == "positive", "gt_area_px2": area,
            "shape_evaluable": state == "positive" and area is not None and area >= 10,
            "small_positive": state == "positive" and area is not None and area < 10,
            "gt_valid": bool(gt.is_valid), "gt_empty": bool(gt.is_empty),
            "geometry_issue": explain_validity(gt) if not gt.is_valid else "",
            "prior_valid": bool(prior.is_valid and not prior.is_empty),
            "prior_area_px2": float(prior.area) if prior.is_valid else None,
            "roi_contains_fraction": None, "prior_gt_iou_diagnostic": None,
            "roi_image_fraction": None,
            "coordinate_status": "x_y_assumed_unverified; EDA pixel-center convention",
            "gt_frame_fraction": None, "prior_frame_fraction": None,
        }
        frame = box(0, 0, image_shape[1], image_shape[0])
        for name, poly in (("gt", gt), ("prior", prior)):
            if poly.is_valid and not poly.is_empty and poly.area > 0:
                record[f"{name}_frame_fraction"] = float(poly.intersection(frame).area / poly.area)
        if roi is not None:
            frame = box(0, 0, image_shape[1], image_shape[0])
            record["roi_image_fraction"] = float(roi.intersection(frame).area / roi.area)
            if area and state == "positive":
                record["roi_contains_fraction"] = float(gt.intersection(roi).area / area)
        if prior.is_valid and area and state == "positive":
            union = prior.union(gt).area
            record["prior_gt_iou_diagnostic"] = float(prior.intersection(gt).area / union) if union else None
        records.append(record)
    return records


def site_overlaps(scene):
    """Within-scene polygon overlap only; no inference of world georeferencing."""
    lookup = {str(r["site_id"]):r for r in scene.labels}
    rows = []
    for field in ("prior_polygon", "polygon"):
        polygons = []
        for site in scene.sites:
            source = site if field == "prior_polygon" else lookup.get(str(site["site_id"]), {})
            polygon = geometry(source.get(field, []))
            if polygon.is_valid and not polygon.is_empty:
                polygons.append((str(site["site_id"]),polygon))
        for i, (left_id,left) in enumerate(polygons):
            for right_id,right in polygons[i+1:]:
                overlap = left.intersection(right).area
                if overlap > 0:
                    rows.append({"location_id":scene.location,"date":scene.date,
                        "kind":field,"site_left":left_id,"site_right":right_id,
                        "intersection_px2":float(overlap),"coordinate_status":"x_y_assumed_unverified"})
    return rows


def duplicate_images(scenes):
    """Content digests are used specifically to detect exact duplicates."""
    groups = {}
    for scene in scenes:
        digest = hashlib.blake2b(digest_size=16)
        with scene.image_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        groups.setdefault(digest.hexdigest(), []).append(scene.key)
    return [members for members in groups.values() if len(members) > 1]


def file_snapshot(paths, root):
    return [{"path": str(p.relative_to(root)), "size": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in paths]


def supervision_summary(records):
    return dict(Counter(row["label_state"] for row in records))
