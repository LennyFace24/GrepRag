"""结果汇总 + 对比表格"""
import json, csv
from src.config import RESULTS_DIR, SIZES, MODES, BACKENDS
from src.scorer import score, acc

def _load(size, mode, backend):
    p = RESULTS_DIR / f"{size}_{mode}_{backend}.jsonl"
    if not p.exists(): return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def _scores(results):
    ss, errs = [], 0
    for r in results:
        pred, gt = r.get("prediction",""), r.get("ground_truth","")
        if pred.startswith("[ERROR]") or pred.startswith("[TIMEOUT]") or pred.startswith("[CLI_ERROR]") or pred.startswith("[RATE_LIMITED]"):
            errs += 1; ss.append(0.0)
        else:
            ss.append(score(pred, gt))
    return {"n": len(results), "f1": sum(ss)/len(ss) if ss else 0, "acc": acc(ss), "errs": errs}

def table():
    print("\n" + "="*80 + "\n  Grep vs RAG 实验结果\n" + "="*80)
    for s in SIZES:
        print(f"\n  数据集: {s}\n  {'Backend':<10} {'Grep F1':>9} {'Grep Acc':>9}  {'RAG F1':>9} {'RAG Acc':>9}  {'Δ F1':>9}\n  " + "─"*60)
        for b in BACKENDS:
            g = _scores(_load(s, "grep", b))
            r = _scores(_load(s, "vector", b))
            print(f"  {b:<10} {g['f1']:>9.4f} {g['acc']:>9.4f}  {r['f1']:>9.4f} {r['acc']:>9.4f}  {g['f1']-r['f1']:>+9.4f}")
    print(f"\n  Δ F1 = Grep - RAG (正=grep优, 负=RAG优)")

def csv_export():
    rows = []
    for s in SIZES:
        for m in MODES:
            for b in BACKENDS:
                for r in _load(s, m, b):
                    pred = r.get("prediction","")
                    f1 = score(pred, r.get("ground_truth","")) if not pred.startswith("[ERROR]") else 0
                    rows.append({"size": s, "mode": m, "backend": b, "qid": r.get("question_id",""),
                                 "query": r.get("query",""), "gt": r.get("ground_truth",""),
                                 "pred": pred, "f1": round(f1,4), "time": r.get("elapsed_seconds",0)})
    if rows:
        p = RESULTS_DIR / "detailed.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(f"\n  详细结果 → {p} ({len(rows)}条)")

if __name__ == "__main__":
    table(); csv_export()
