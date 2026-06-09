
# ========== DeepFashion2 ==========
# deepfashion2_train_images: 191961
# deepfashion2_train_annos: 103822
# deepfashion2_val_images: 32153
# deepfashion2_val_annos: 32153

# ========== FashionAI Attributes ==========
# coat_length_labels: 1453
# collar_design_labels: 1082
# lapel_design_labels: 900
# neck_design_labels: 708
# neckline_design_labels: 2095
# pant_length_labels: 949
# skirt_length_labels: 1153
# sleeve_length_labels: 1740
# FashionAI total images: 10080


from pathlib import Path

ROOT = Path(r"D:\Aliintern\fashion-ai-data")

paths = {
    "deepfashion2_train_images": ROOT / "deepfashion2" / "train" / "image",
    "deepfashion2_train_annos": ROOT / "deepfashion2" / "train" / "annos",
    "deepfashion2_val_images": ROOT / "deepfashion2" / "validation" / "image",
    "deepfashion2_val_annos": ROOT / "deepfashion2" / "validation" / "annos",
}

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("========== DeepFashion2 ==========")
for name, path in paths.items():
    if not path.exists():
        print(f"{name}: NOT FOUND - {path}")
        continue

    if "image" in name:
        count = sum(1 for p in path.iterdir() if p.suffix.lower() in image_exts)
    else:
        count = sum(1 for p in path.iterdir() if p.suffix.lower() == ".json")

    print(f"{name}: {count}")

print("\n========== FashionAI Attributes ==========")

attr_root = (
    ROOT
    / "fashionai_attributes"
    / "round1_fashionAI_attributes_test_a"
    / "Images"
)

if not attr_root.exists():
    print(f"FashionAI Images root NOT FOUND: {attr_root}")
else:
    total = 0
    for d in sorted(attr_root.iterdir()):
        if not d.is_dir():
            continue
        count = sum(1 for p in d.iterdir() if p.suffix.lower() in image_exts)
        total += count
        print(f"{d.name}: {count}")
    print(f"FashionAI total images: {total}")
