import argparse
import random
import shutil
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--src-yaml", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--num", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_root = Path(args.src-root) if False else Path(args.src_root)
    out_root = Path(args.out_root)

    src_img_dir = src_root / "images" / "val"
    src_lbl_dir = src_root / "labels" / "val"

    out_img_dir = out_root / "images" / "val"
    out_lbl_dir = out_root / "labels" / "val"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        img_paths.extend(src_img_dir.glob(ext))

    img_paths = sorted(img_paths)
    random.seed(args.seed)
    sampled = random.sample(img_paths, min(args.num, len(img_paths)))

    for img_path in sampled:
        label_path = src_lbl_dir / f"{img_path.stem}.txt"
        shutil.copy2(img_path, out_img_dir / img_path.name)

        if label_path.exists():
            shutil.copy2(label_path, out_lbl_dir / label_path.name)
        else:
            print(f"[WARN] missing label: {label_path}")

    with open(args.src_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str(out_root.resolve()).replace("\\", "/")
    cfg["train"] = "images/val"
    cfg["val"] = "images/val"

    out_yaml = out_root / "deepfashion2_13cls_val2000.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[DONE] sampled {len(sampled)} images")
    print(f"[OUT ROOT] {out_root}")
    print(f"[OUT YAML] {out_yaml}")

if __name__ == "__main__":
    main()
