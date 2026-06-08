import argparse
import json
from pathlib import Path

import cv2


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_records(obj, inherited_image_path=None, out=None):
    """
    Recursively find instance records that contain landmarks/keypoints.
    It inherits image_path from parent nodes.
    """
    if out is None:
        out = []

    if isinstance(obj, dict):
        image_path = (
            obj.get("image_path")
            or obj.get("image")
            or obj.get("original_image_path")
            or obj.get("source_image")
            or obj.get("file_name")
            or obj.get("filename")
            or inherited_image_path
        )

        if isinstance(obj.get("landmarks"), list) or isinstance(obj.get("keypoints"), list):
            rec = dict(obj)
            if image_path is not None:
                rec["_image_path_for_viz"] = image_path
            out.append(rec)
            return out

        for v in obj.values():
            find_records(v, image_path, out)

    elif isinstance(obj, list):
        for item in obj:
            find_records(item, inherited_image_path, out)

    return out


def get_class_name(rec):
    return str(
        rec.get("class_name")
        or rec.get("category")
        or rec.get("label")
        or rec.get("det_class_name")
        or rec.get("category_name")
        or ""
    )


def get_image_path(rec):
    return (
        rec.get("_image_path_for_viz")
        or rec.get("image_path")
        or rec.get("image")
        or rec.get("original_image_path")
        or rec.get("source_image")
        or rec.get("file_name")
        or rec.get("filename")
    )


def get_landmarks(rec):
    return rec.get("landmarks") or rec.get("keypoints") or []


def get_xy(lm):
    if not isinstance(lm, dict):
        return None

    if "x" in lm and "y" in lm:
        return int(round(float(lm["x"]))), int(round(float(lm["y"])))

    if "point" in lm and isinstance(lm["point"], list) and len(lm["point"]) >= 2:
        return int(round(float(lm["point"][0]))), int(round(float(lm["point"][1])))

    if "xy" in lm and isinstance(lm["xy"], list) and len(lm["xy"]) >= 2:
        return int(round(float(lm["xy"][0]))), int(round(float(lm["xy"][1])))

    return None


def get_index(lm, fallback):
    """
    Display raw index only.
    If missing, fallback to 1-based order.
    """
    for k in ["index", "idx", "landmark_index", "id"]:
        if isinstance(lm, dict) and k in lm:
            try:
                return int(lm[k])
            except Exception:
                pass
    return fallback + 1


def get_bbox(rec):
    for k in ["bbox_xyxy", "box", "det_bbox", "bbox"]:
        b = rec.get(k)
        if isinstance(b, list) and len(b) == 4:
            try:
                return [int(round(float(x))) for x in b]
            except Exception:
                return None
    return None


def draw_number(img, text, x, y, scale=0.75):
    """
    Draw ONLY number text. No name, no class, no region.
    """
    h, w = img.shape[:2]
    x = max(0, min(int(x), w - 1))
    y = max(12, min(int(y), h - 1))

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # black outline
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # yellow number
    cv2.putText(img, text, (x, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--font-scale", type=float, default=0.75)
    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--point-radius", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(args.landmarks_json)
    records = find_records(data)

    class_filter = set(args.classes) if args.classes else None

    # group records by image path
    grouped = {}
    for rec in records:
        cls = get_class_name(rec)
        if class_filter is not None and cls not in class_filter:
            continue

        img_path = get_image_path(rec)
        if not img_path:
            continue

        grouped.setdefault(str(img_path), []).append(rec)

    items = list(grouped.items())
    if args.max_images > 0:
        items = items[: args.max_images]

    written = 0
    failed = 0

    for img_path, recs in items:
        img = cv2.imread(img_path)
        if img is None:
            print("[WARN] cannot read image:", img_path)
            failed += 1
            continue

        h, w = img.shape[:2]

        for rec in recs:
            if args.draw_bbox:
                bbox = get_bbox(rec)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))
                    # bbox only, no class text
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            landmarks = get_landmarks(rec)
            for i, lm in enumerate(landmarks):
                xy = get_xy(lm)
                if xy is None:
                    continue

                x, y = xy
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue

                idx = get_index(lm, i)

                # draw point
                cv2.circle(img, (x, y), args.point_radius + 2, (0, 0, 0), -1)
                cv2.circle(img, (x, y), args.point_radius, (0, 255, 0), -1)

                # draw ONLY number
                draw_number(
                    img,
                    str(idx),
                    x + 6,
                    y - 6,
                    scale=args.font_scale,
                )

        p = Path(img_path)
        out_path = output_dir / f"{p.stem}_numbers{p.suffix or '.jpg'}"
        if cv2.imwrite(str(out_path), img):
            written += 1
        else:
            failed += 1

    print(f"[INFO] records={len(records)}")
    print(f"[INFO] images={len(items)}")
    print(f"[INFO] written={written}")
    print(f"[INFO] failed={failed}")
    print(f"[INFO] output_dir={output_dir}")


if __name__ == "__main__":
    main()
