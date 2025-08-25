import json, pathlib
p = pathlib.Path(r"models/tfidf_lr_fast.metrics.json")
m = json.loads(p.read_text(encoding="utf-8"))
print("Classes:", m["classes"])
for c, s in m["metrics"].items():
    if isinstance(s, dict) and "f1-score" in s:
        print(f"{c:10s}  P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1-score']:.3f}")
print("Macro-F1:", m["metrics"]["macro avg"]["f1-score"])
