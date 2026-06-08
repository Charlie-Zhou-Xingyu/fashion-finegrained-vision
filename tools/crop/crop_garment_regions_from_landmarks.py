from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


CATEGORY_TO_REGIONS: Dict[str, List[str]] = {
    # Upper garments: now include waist.
    "short sleeve top": ["collar", "sleeve", "hem", "waist"],
    "long sleeve top": ["collar", "sleeve", "hem", "waist"],
    "short sleeve outwear": ["collar", "sleeve", "hem", "waist"],
    "long sleeve outwear": ["collar", "sleeve", "hem", "waist"],

    # Sleeveless / sling upper garments: now include waist.
    "vest": ["collar", "hem", "waist"],
    "sling": ["collar", "hem", "waist"],

    # Bottom garments.
    "shorts": ["waist", "pant_leg"],
    "trousers": ["waist", "pant_leg"],
    "skirt": ["waist", "hem"],

    # Dresses: now include waist.
    "short sleeve dress": ["collar", "sleeve", "hem", "waist"],
    "long sleeve dress": ["collar", "sleeve", "hem", "waist"],
    "vest dress": ["collar", "hem", "waist"],
    "sling dress": ["collar", "hem", "waist"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop garment local regions from landmark results."
    )
    parser.add_argument(
        "--landmarks-json",
        type=str,
        required=True,
        help="Path to landmarks_results.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for region crops.",
    )
    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=[
            "collar",
            "sleeve",
            "hem",
            "waist",
            "pant_leg",
        ],
        help=(
            "Candidate regions to crop. If --use-category-regions is enabled, "
            "these are further intersected with CATEGORY_TO_REGIONS[class_name]."
        ),
    )
    parser.add_argument(
        "--use-category-regions",
        action="store_true",
        help="Only crop regions supported by each garment category.",
    )
    parser.add_argument(
        "--max-outside-distance",
        type=float,
        default=5.0,
        help="Max distance for outside_mask landmark to be considered reliable.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help=(
            "Minimum reliable landmarks for multi-point landmark bbox crop. "
            "If 1 reliable point exists, a fixed-size point crop is used."
        ),
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.35,
        help=(
            "Base padding ratio around landmark bbox. "
            "Some regions use semantic expansion and ignore/override this."
        ),
    )
    parser.add_argument(
        "--single-point-box-ratio",
        type=float,
        default=0.18,
        help="Box size ratio for one-point region crop.",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Enable bbox fallback when reliable landmarks are insufficient.",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Reserved flag. Debug visualization is handled by visualize_region_crops_debug.py.",
    )
    return parser.parse_args()


def normalize_category_name(value: Any) -> str:
    if value is None:
        return "unknown"
    name = str(value).strip().lower()
    name = name.replace("_", " ")
    name = " ".join(name.split())
    return name if name else "unknown"


def get_regions_for_instance(
    class_name: str,
    candidate_regions: List[str],
    use_category_regions: bool,
) -> List[str]:
    candidate = [str(r).strip().lower() for r in candidate_regions]

    if not use_category_regions:
        return candidate

    normalized_class = normalize_category_name(class_name)
    supported = CATEGORY_TO_REGIONS.get(normalized_class)

    if supported is None:
        # Unknown category: safe fallback to user-specified candidate regions.
        return candidate

    supported_set = set(supported)
    return [region for region in candidate if region in supported_set]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_image_bgr(path: str) -> Optional[np.ndarray]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def normalize_bbox_xyxy(raw_bbox: Any) -> Optional[List[float]]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    except Exception:
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def clip_bbox(
    bbox: List[float],
    image_width: int,
    image_height: int,
) -> Optional[List[int]]:
    x1, y1, x2, y2 = bbox

    x1 = int(round(max(0, min(image_width - 1, x1))))
    y1 = int(round(max(0, min(image_height - 1, y1))))
    x2 = int(round(max(0, min(image_width, x2))))
    y2 = int(round(max(0, min(image_height, y2))))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def expand_bbox(
    bbox: List[float],
    pad_ratio: float,
) -> List[float]:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    pad = max(w, h) * float(pad_ratio)

    return [
        x1 - pad,
        y1 - pad,
        x2 + pad,
        y2 + pad,
    ]


def is_reliable_landmark(
    landmark: Dict[str, Any],
    max_outside_distance: float,
) -> bool:
    if landmark.get("valid_for_class", True) is False:
        return False

    quality = str(landmark.get("quality", ""))

    if quality == "ok":
        return True

    if quality == "refined_by_mask":
        return True

    if quality == "outside_mask":
        distance = landmark.get("distance_to_mask", None)
        if isinstance(distance, (int, float)):
            return float(distance) <= float(max_outside_distance)

    return False


def points_from_landmarks(
    landmarks: List[Dict[str, Any]],
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []

    for lm in landmarks:
        try:
            x = float(lm["x"])
            y = float(lm["y"])
        except Exception:
            continue
        points.append((x, y))

    return points


def landmark_bbox(
    landmarks: List[Dict[str, Any]],
    instance_bbox: List[float],
    min_points: int,
    pad_ratio: float,
    single_point_box_ratio: float,
) -> Optional[List[float]]:
    points = points_from_landmarks(landmarks)

    if len(points) <= 0:
        return None

    inst_x1, inst_y1, inst_x2, inst_y2 = instance_bbox
    inst_w = max(1.0, inst_x2 - inst_x1)
    inst_h = max(1.0, inst_y2 - inst_y1)

    if len(points) < int(min_points):
        x, y = points[0]
        size = max(inst_w, inst_h) * float(single_point_box_ratio)
        return [
            x - size * 0.5,
            y - size * 0.5,
            x + size * 0.5,
            y + size * 0.5,
        ]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    box = [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]

    return expand_bbox(box, pad_ratio=pad_ratio)


def split_sleeve_landmarks(
    sleeve_landmarks: List[Dict[str, Any]],
    instance_bbox: List[float],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Split sleeve landmarks into left_sleeve and right_sleeve.

    Priority:
        1. Use semantic name, e.g. left_sleeve_01 / right_sleeve_01.
        2. Fallback to x-position relative to instance bbox center.

    Note:
        The returned component names are used for crop filenames:
            left_sleeve / right_sleeve
        The logical region remains:
            sleeve
    """
    left: List[Dict[str, Any]] = []
    right: List[Dict[str, Any]] = []
    unknown: List[Dict[str, Any]] = []

    for lm in sleeve_landmarks:
        name = str(lm.get("name", "")).strip().lower()

        if "left_sleeve" in name or "left_cuff" in name:
            left.append(lm)
        elif "right_sleeve" in name or "right_cuff" in name:
            right.append(lm)
        else:
            unknown.append(lm)

    if unknown:
        x1, _, x2, _ = instance_bbox
        cx = 0.5 * (x1 + x2)

        for lm in unknown:
            try:
                x = float(lm["x"])
            except Exception:
                continue

            if x <= cx:
                left.append(lm)
            else:
                right.append(lm)

    components: List[Tuple[str, List[Dict[str, Any]]]] = []

    if left:
        components.append(("left_sleeve", left))
    if right:
        components.append(("right_sleeve", right))

    if not components and sleeve_landmarks:
        components.append(("sleeve", sleeve_landmarks))

    return components


def fallback_bbox_for_region(
    instance_bbox: List[float],
    region: str,
) -> Optional[List[float]]:
    x1, y1, x2, y2 = instance_bbox
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return None

    region = str(region).lower()

    if region == "collar":
        return [
            x1 + 0.20 * w,
            y1 + 0.00 * h,
            x1 + 0.80 * w,
            y1 + 0.30 * h,
        ]

    if region == "sleeve":
        return [
            x1 + 0.00 * w,
            y1 + 0.05 * h,
            x1 + 1.00 * w,
            y1 + 0.55 * h,
        ]

    if region == "left_sleeve":
        return [
            x1 + 0.00 * w,
            y1 + 0.05 * h,
            x1 + 0.48 * w,
            y1 + 0.58 * h,
        ]

    if region == "right_sleeve":
        return [
            x1 + 0.52 * w,
            y1 + 0.05 * h,
            x1 + 1.00 * w,
            y1 + 0.58 * h,
        ]

    if region == "hem":
        return [
            x1 + 0.04 * w,
            y1 + 0.70 * h,
            x1 + 0.96 * w,
            y1 + 1.00 * h,
        ]

    if region == "waist":
        return [
            x1 + 0.08 * w,
            y1 + 0.32 * h,
            x1 + 0.92 * w,
            y1 + 0.58 * h,
        ]

    if region == "pant_leg":
        return [
            x1 + 0.05 * w,
            y1 + 0.45 * h,
            x1 + 0.95 * w,
            y1 + 1.00 * h,
        ]

    return None


def expand_region_bbox_semantic(
    bbox: List[float],
    region: str,
    instance_bbox: List[float],
) -> List[float]:
    """
    Region-specific bbox expansion.

    Key behavior:
        - hem:
            Use landmark y-position but widen x to most of garment bbox.
            This prevents tiny square hem crops.
        - waist:
            Use landmark y-position but widen x to most of garment bbox.
        - left_sleeve/right_sleeve:
            Keep local crop around each side sleeve landmarks.
        - collar:
            Keep compact.
    """
    x1, y1, x2, y2 = bbox
    gx1, gy1, gx2, gy2 = instance_bbox

    gw = max(1.0, gx2 - gx1)
    gh = max(1.0, gy2 - gy1)

    region = str(region).lower()

    if region == "hem":
        cy = 0.5 * (y1 + y2)

        # Use most of the garment bbox width.
        new_x1 = gx1 + 0.04 * gw
        new_x2 = gx2 - 0.04 * gw

        # Hem landmarks are usually near the lower edge.
        # Expand more upward than downward.
        up_h = max(0.15 * gh, 45.0)
        down_h = max(0.15 * gh, 25.0)

        return [
            new_x1,
            cy - up_h,
            new_x2,
            cy + down_h,
        ]


    if region == "waist":
        cy = 0.5 * (y1 + y2)

        new_x1 = gx1 + 0.08 * gw
        new_x2 = gx2 - 0.08 * gw

        half_h = max(0.070 * gh, 22.0)

        return [
            new_x1,
            cy - half_h,
            new_x2,
            cy + half_h,
        ]

    if region == "collar":
        return expand_bbox(bbox, pad_ratio=0.15)

    if region in {"sleeve", "left_sleeve", "right_sleeve"}:
        return expand_bbox(bbox, pad_ratio=0.15)

    if region == "pant_leg":
        return expand_bbox(bbox, pad_ratio=0.45)

    return bbox


def crop_and_save(
    image_bgr: np.ndarray,
    bbox_xyxy: List[int],
    output_path: Path,
) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    crop = image_bgr[y1:y2, x1:x2]

    if crop.size <= 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), crop))


def make_crop_filename(
    image_path: str,
    det_id: Any,
    class_name: str,
    region: str,
) -> str:
    stem = Path(image_path).stem
    safe_class = str(class_name).replace(" ", "_").replace("/", "_")
    safe_region = str(region).replace(" ", "_").replace("/", "_")

    if det_id is None:
        det_text = "detNA"
    else:
        try:
            det_text = f"det{int(det_id):03d}"
        except Exception:
            det_text = f"det{det_id}"

    return f"{stem}_{det_text}_{safe_class}_{safe_region}.jpg"


def make_region_components(
    region: str,
    region_landmarks: List[Dict[str, Any]],
    instance_bbox: List[float],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    region = str(region).lower()

    if region == "sleeve":
        return split_sleeve_landmarks(
            sleeve_landmarks=region_landmarks,
            instance_bbox=instance_bbox,
        )

    return [(region, region_landmarks)]


def process_instance(
    image_bgr: np.ndarray,
    image_path: str,
    instance: Dict[str, Any],
    candidate_regions: List[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    image_h, image_w = image_bgr.shape[:2]

    instance_bbox = normalize_bbox_xyxy(instance.get("bbox_xyxy", instance.get("bbox")))
    if instance_bbox is None:
        return []

    landmarks = instance.get("landmarks", [])
    if not isinstance(landmarks, list):
        return []

    class_name = str(instance.get("class_name", instance.get("category_name", "unknown")))
    det_id = instance.get("det_id")

    regions = get_regions_for_instance(
        class_name=class_name,
        candidate_regions=candidate_regions,
        use_category_regions=bool(args.use_category_regions),
    )

    records: List[Dict[str, Any]] = []

    for region in regions:
        region = str(region).lower()

        region_landmarks = [
            lm for lm in landmarks
            if str(lm.get("region", "")).lower() == region
        ]

        components = make_region_components(
            region=region,
            region_landmarks=region_landmarks,
            instance_bbox=instance_bbox,
        )

        for component_name, component_landmarks in components:
            component_name = str(component_name).lower()

            reliable_landmarks = [
                lm for lm in component_landmarks
                if is_reliable_landmark(
                    lm,
                    max_outside_distance=float(args.max_outside_distance),
                )
            ]

            crop_source = "landmark_region"
            fallback_used = False

            # Region-specific base padding before semantic expansion.
            # Avoid making crops too large.
            if component_name in {"left_sleeve", "right_sleeve", "sleeve"}:
                local_pad_ratio = 0.16
            elif component_name == "collar":
                local_pad_ratio = 0.18
            elif component_name in {"hem", "waist"}:
                # Hem/waist will be expanded semantically later.
                local_pad_ratio = 0.05
            elif component_name == "pant_leg":
                local_pad_ratio = 0.30
            else:
                local_pad_ratio = float(args.pad_ratio)

            box = landmark_bbox(
                landmarks=reliable_landmarks,
                instance_bbox=instance_bbox,
                min_points=int(args.min_points),
                pad_ratio=local_pad_ratio,
                single_point_box_ratio=float(args.single_point_box_ratio),
            )

            if box is not None:
                # Critical semantic expansion:
                # hem/waist become wide bands; sleeves stay side-local.
                box = expand_region_bbox_semantic(
                    bbox=box,
                    region=component_name,
                    instance_bbox=instance_bbox,
                )

            if box is None and bool(args.fallback):
                box = fallback_bbox_for_region(
                    instance_bbox=instance_bbox,
                    region=component_name,
                )
                crop_source = "bbox_fallback"
                fallback_used = True

            if box is None:
                records.append(
                    {
                        "image_path": image_path,
                        "class_name": class_name,
                        "det_id": det_id,
                        "region": region,
                        "component": component_name,
                        "crop_path": None,
                        "source": "none",
                        "fallback": False,
                        "success": False,
                        "reason": "no_reliable_landmarks",
                        "bbox_xyxy": None,
                        "num_landmarks": len(component_landmarks),
                        "num_reliable_landmarks": len(reliable_landmarks),
                        "landmark_indices": [
                            int(lm.get("index", -1)) for lm in component_landmarks
                        ],
                        "reliable_landmark_indices": [
                            int(lm.get("index", -1)) for lm in reliable_landmarks
                        ],
                    }
                )
                continue

            clipped = clip_bbox(
                box,
                image_width=image_w,
                image_height=image_h,
            )

            if clipped is None:
                records.append(
                    {
                        "image_path": image_path,
                        "class_name": class_name,
                        "det_id": det_id,
                        "region": region,
                        "component": component_name,
                        "crop_path": None,
                        "source": crop_source,
                        "fallback": fallback_used,
                        "success": False,
                        "reason": "invalid_crop_bbox",
                        "bbox_xyxy": None,
                        "num_landmarks": len(component_landmarks),
                        "num_reliable_landmarks": len(reliable_landmarks),
                        "landmark_indices": [
                            int(lm.get("index", -1)) for lm in component_landmarks
                        ],
                        "reliable_landmark_indices": [
                            int(lm.get("index", -1)) for lm in reliable_landmarks
                        ],
                    }
                )
                continue

            filename = make_crop_filename(
                image_path=image_path,
                det_id=det_id,
                class_name=class_name,
                region=component_name,
            )

            # Keep directory grouped by logical region.
            # Example:
            #   crops/sleeve/xxx_left_sleeve.jpg
            #   crops/sleeve/xxx_right_sleeve.jpg
            crop_path = output_dir / "crops" / region / filename

            ok = crop_and_save(
                image_bgr=image_bgr,
                bbox_xyxy=clipped,
                output_path=crop_path,
            )

            records.append(
                {
                    "image_path": image_path,
                    "class_name": class_name,
                    "det_id": det_id,
                    "region": region,
                    "component": component_name,
                    "crop_path": str(crop_path),
                    "source": crop_source,
                    "fallback": fallback_used,
                    "success": bool(ok),
                    "bbox_xyxy": clipped,
                    "num_landmarks": len(component_landmarks),
                    "num_reliable_landmarks": len(reliable_landmarks),
                    "landmark_indices": [
                        int(lm.get("index", -1)) for lm in component_landmarks
                    ],
                    "reliable_landmark_indices": [
                        int(lm.get("index", -1)) for lm in reliable_landmarks
                    ],
                }
            )

    return records


def main() -> None:
    args = parse_args()

    landmarks_json = Path(args.landmarks_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(landmarks_json)

    crop_records: List[Dict[str, Any]] = []

    num_images = 0
    num_images_failed = 0
    num_instances = 0

    for image_record in data.get("images", []):
        image_path = image_record.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            continue

        image_bgr = read_image_bgr(image_path)
        if image_bgr is None:
            print(f"[WARN] Failed to read image: {image_path}")
            num_images_failed += 1
            continue

        num_images += 1

        instances = image_record.get("instances", [])
        if not isinstance(instances, list):
            continue

        for instance in instances:
            num_instances += 1
            records = process_instance(
                image_bgr=image_bgr,
                image_path=image_path,
                instance=instance,
                candidate_regions=list(args.regions),
                output_dir=output_dir,
                args=args,
            )
            crop_records.extend(records)

    source_counter = Counter()
    region_counter = Counter()
    component_counter = Counter()
    class_counter = Counter()

    for record in crop_records:
        source_counter[str(record.get("source", "unknown"))] += 1
        region_counter[str(record.get("region", "unknown"))] += 1
        component_counter[str(record.get("component", record.get("region", "unknown")))] += 1
        class_counter[str(record.get("class_name", "unknown"))] += 1

    summary = {
        "landmarks_json": str(landmarks_json),
        "output_dir": str(output_dir),
        "candidate_regions": list(args.regions),
        "use_category_regions": bool(args.use_category_regions),
        "category_to_regions": CATEGORY_TO_REGIONS if bool(args.use_category_regions) else None,
        "max_outside_distance": float(args.max_outside_distance),
        "min_points": int(args.min_points),
        "pad_ratio": float(args.pad_ratio),
        "single_point_box_ratio": float(args.single_point_box_ratio),
        "fallback": bool(args.fallback),
        "num_images": num_images,
        "num_images_failed": num_images_failed,
        "num_instances": num_instances,
        "num_crop_records": len(crop_records),
        "num_success": sum(1 for r in crop_records if r.get("success")),
        "num_failed": sum(1 for r in crop_records if not r.get("success")),
        "num_fallback": sum(1 for r in crop_records if r.get("fallback")),
        "source_counts": dict(source_counter),
        "region_counts": dict(region_counter),
        "component_counts": dict(component_counter),
        "class_counts": dict(class_counter),
    }

    output = {
        "summary": summary,
        "crops": crop_records,
    }

    save_json(output, output_dir / "region_crops.json")
    save_json(summary, output_dir / "summary.json")

    print("[INFO] Region crop finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] Output JSON: {output_dir / 'region_crops.json'}")
    print(f"[INFO] Crop dir: {output_dir / 'crops'}")


if __name__ == "__main__":
    main()
