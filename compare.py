import json
import os

TOOL_FILE = "reports/scan_20260624_194853.json"
TRUTH_FILE = "evaluation-s7.json"


def load_supported(path):
    with open(path) as fh:
        data = json.load(fh)
    out = {}
    for practice, entry in data.get("results_by_practice", {}).items():
        for f in entry.get("findings", []):
            key = (practice, os.path.basename(f["file_path"]))
            out[key] = bool(f.get("supported", False))
    return out


tool = load_supported(TOOL_FILE)
truth = load_supported(TRUTH_FILE)

TP = FP = FN = TN = 0
per_practice = {}
disagreements = []

for key in sorted(truth):
    practice, fname = key
    t = truth[key]
    d = tool.get(key, False)
    if t and d:
        cat = "TP"
    elif d and not t:
        cat = "FP"
    elif t and not d:
        cat = "FN"
    else:
        cat = "TN"

    cnt = per_practice.setdefault(practice, [0, 0, 0, 0])
    idx = {"TP": 0, "FP": 1, "FN": 2, "TN": 3}[cat]
    cnt[idx] += 1
    if cat == "TP": TP += 1
    elif cat == "FP": FP += 1
    elif cat == "FN": FN += 1
    else: TN += 1

    if cat in ("FP", "FN"):
        disagreements.append((practice, fname, t, d, cat))


def prf(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


print("=" * 78)
print("Per-practice (counts over the 6 scanned files)")
print("-" * 78)
print("{:<48} {:>3} {:>3} {:>3} {:>3}".format("Practice", "TP", "FP", "FN", "TN"))
for practice in sorted(per_practice):
    tp, fp, fn, tn = per_practice[practice]
    print("{:<48} {:>3} {:>3} {:>3} {:>3}".format(practice, tp, fp, fn, tn))

print("=" * 78)
print("Disagreements (where tool and ground truth differ)")
print("-" * 78)
if not disagreements:
    print("  none - perfect agreement")
for practice, fname, t, d, cat in disagreements:
    print("  [{}] {:<42} truth={} tool={}".format(
        cat, practice + " / " + fname, t, d))

acc, prec, rec, f1 = prf(TP, FP, FN, TN)
print("=" * 78)
print("OVERALL over {} (file, practice) pairs".format(TP + FP + FN + TN))
print("TP={}  FP={}  FN={}  TN={}".format(TP, FP, FN, TN))
print("Accuracy={:.3f}  Precision={:.3f}  Recall={:.3f}  F1={:.3f}".format(
    acc, prec, rec, f1))
print("=" * 78)