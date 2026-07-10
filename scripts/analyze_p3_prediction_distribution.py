import argparse
import json
import collections


def find_label_in_topk(topk_item):
    if not isinstance(topk_item, dict):
        return None
    for k in [
        "label_name",
        "label",
        "class_name",
        "name",
        "pred_label_name",
        "top1_label",
    ]:
        if k in topk_item and topk_item[k] is not None:
            return str(topk_item[k])
    return None


def find_conf_in_topk(topk_item):
    if not isinstance(topk_item, dict):
        return None
    for k in [
        "confidence",
        "score",
        "prob",
        "probability",
        "top1_confidence",
    ]:
        if k in topk_item and topk_item[k] is not None:
            try:
                return float(topk_item[k])
            except Exception:
                pass
    return None


def get_top1_label(row):
    # Common flat fields
    for k in [
        "pred_label_name",
        "pred_label",
        "prediction_label",
        "top1_label",
        "label_pred",
        "pred_name",
    ]:
        if k in row and row[k] is not None:
            return str(row[k])

    # prediction dict
    pred = row.get("prediction")
    if isinstance(pred, dict):
        for k in [
            "top1_label",
            "label_name",
            "label",
            "pred_label_name",
            "pred_label",
            "class_name",
        ]:
            if k in pred and pred[k] is not None:
                return str(pred[k])

    # topk list
    topk = row.get("topk") or row.get("topk_predictions") or row.get("predictions")
    if isinstance(topk, list) and len(topk) > 0:
        label = find_label_in_topk(topk[0])
        if label is not None:
            return label

    # maybe prediction itself is a string
    if isinstance(pred, str):
        return pred

    return "NONE"


def get_top1_conf(row):
    # Common flat fields
    for k in [
        "confidence",
        "score",
        "prob",
        "probability",
        "top1_confidence",
        "pred_confidence",
        "pred_score",
    ]:
        if k in row and row[k] is not None:
            try:
                return float(row[k])
            except Exception:
                pass

    # prediction dict
    pred = row.get("prediction")
    if isinstance(pred, dict):
        for k in [
            "top1_confidence",
            "confidence",
            "score",
            "prob",
            "probability",
            "pred_score",
        ]:
            if k in pred and pred[k] is not None:
                try:
                    return float(pred[k])
                except Exception:
                    pass

    # topk list
    topk = row.get("topk") or row.get("topk_predictions") or row.get("predictions")
    if isinstance(topk, list) and len(topk) > 0:
        conf = find_conf_in_topk(topk[0])
        if conf is not None:
            return conf

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-jsonl", required=True)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--debug-first", action="store_true")
    args = parser.parse_args()

    rows = []
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if args.debug_first and rows:
        print("[DEBUG FIRST ROW KEYS]")
        print(list(rows[0].keys()))
        print("[DEBUG FIRST ROW]")
        print(json.dumps(rows[0], ensure_ascii=False, indent=2))

    total = collections.Counter()
    by_class = collections.defaultdict(collections.Counter)
    by_component = collections.defaultdict(collections.Counter)
    conf_by_pred = collections.defaultdict(list)

    for r in rows:
        pred = get_top1_label(r)
        conf = get_top1_conf(r)

        garment_cls = (
            r.get("class_name")
            or r.get("garment_class")
            or r.get("category_name")
            or "UNKNOWN_CLASS"
        )

        comp = (
            r.get("component")
            or r.get("region")
            or r.get("region_name")
            or "UNKNOWN_COMPONENT"
        )

        total[pred] += 1
        by_class[garment_cls][pred] += 1
        by_component[comp][pred] += 1

        if conf is not None:
            conf_by_pred[pred].append(conf)

    print(f"[INFO] rows={len(rows)}")

    print("\n[TOTAL PREDICTION COUNTS]")
    for k, v in total.most_common():
        confs = conf_by_pred.get(k, [])
        if confs:
            avg_conf = sum(confs) / len(confs)
            min_conf = min(confs)
            max_conf = max(confs)
            print(f"{k}: {v}  avg_conf={avg_conf:.4f}  min={min_conf:.4f}  max={max_conf:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n[BY CLASS_NAME]")
    for cls, c in sorted(by_class.items(), key=lambda x: sum(x[1].values()), reverse=True):
        print(f"\n{cls}  n={sum(c.values())}")
        for k, v in c.most_common(args.topn):
            print(f"  {k}: {v}")

    print("\n[BY COMPONENT/REGION]")
    for comp, c in sorted(by_component.items(), key=lambda x: sum(x[1].values()), reverse=True):
        print(f"\n{comp}  n={sum(c.values())}")
        for k, v in c.most_common(args.topn):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
