import json
from pathlib import Path


TASKS = [
    "neck_design",
    "collar_design",
    "lapel_design",
    "sleeve_length",
    "coat_length",
    "pant_length",
    "skirt_length",
]

ROOT = Path("outputs")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x:.6f}"
    return str(x)


print("| task | best_val_acc | best_val_macro_f1 | test_acc | test_macro_f1 | test_weighted_f1 |")
print("|---|---:|---:|---:|---:|---:|")

for task in TASKS:
    out_dir = ROOT / f"p2_{task}_multiview_v2_pipeline_resnet18_seed2"
    best = read_json(out_dir / "best_metrics.json")
    test = read_json(out_dir / "test_metrics.json")

    print(
        "| {task} | {best_acc} | {best_f1} | {test_acc} | {test_f1} | {test_wf1} |".format(
            task=task,
            best_acc=fmt(best.get("accuracy")),
            best_f1=fmt(best.get("macro_f1")),
            test_acc=fmt(test.get("accuracy")),
            test_f1=fmt(test.get("macro_f1")),
            test_wf1=fmt(test.get("weighted_f1")),
        )
    )
