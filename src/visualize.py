"""结果可视化：终端图表 + matplotlib 图表"""
import json, numpy as np
from pathlib import Path
from src.config import RESULTS_DIR, SIZES, MODES, BACKENDS
from src.scorer import score

def _load(size, mode, backend):
    p = RESULTS_DIR / f"{size}_{mode}_{backend}.jsonl"
    if not p.exists(): return [], []
    f1s, ts = [], []
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); pred = r.get("prediction","")
        f1s.append(0.0 if pred.startswith("[ERROR]") or pred.startswith("[TIMEOUT]") or pred.startswith("[CLI_ERROR]")
                   else score(pred, r.get("ground_truth","")))
        ts.append(r.get("elapsed_seconds",0))
    return f1s, ts

def ascii_bar():
    print("\n" + "="*60 + "\n  F1 对比\n" + "="*60)
    for s in SIZES:
        print(f"\n  数据集: {s}\n  " + "─"*50)
        scores = {}
        for m in MODES:
            for b in BACKENDS:
                ss, _ = _load(s, m, b)
                if ss: scores[f"  {m}/{b}"] = np.mean(ss)
        if not scores: print("    (无数据)"); continue
        w = max(len(k) for k in scores)
        mx = max(scores.values()) or 1
        for k in sorted(scores):
            v = scores[k]; n = int(v/mx*40)
            print(f"{k:<{w}}  {'█'*n}{'░'*(40-n)}  {v:.3f}")

def table():
    print("\n" + "="*80 + "\n  Grep vs RAG\n" + "="*80)
    for s in SIZES:
        print(f"\n  数据集: {s}\n  {'Backend':<10} {'Grep F1':>9} {'Grep Acc':>9}  {'RAG F1':>9} {'RAG Acc':>9}  {'Δ F1':>9}\n  " + "─"*60)
        for b in BACKENDS:
            gs, _ = _load(s, "grep", b); rs, _ = _load(s, "vector", b)
            gf = np.mean(gs) if gs else 0; rf = np.mean(rs) if rs else 0
            ga = np.mean([x>=0.5 for x in gs]) if gs else 0; ra = np.mean([x>=0.5 for x in rs]) if rs else 0
            print(f"  {b:<10} {gf:>9.4f} {ga:>9.4f}  {rf:>9.4f} {ra:>9.4f}  {gf-rf:>+9.4f}")

def stats():
    print("\n" + "="*60 + "\n  耗时统计\n" + "="*60)
    for s in SIZES:
        for m in MODES:
            for b in BACKENDS:
                _, ts = _load(s, m, b)
                if ts: print(f"  {s}/{m}/{b}: avg={np.mean(ts):.1f}s  max={np.max(ts):.0f}s  total={np.sum(ts):.0f}s")

def plot(save_path=None):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        return print("\n  [跳过] matplotlib 未安装")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {"grep": "#4ECDC4", "vector": "#FF6B6B"}
    # 柱状图
    ax = axes[0,0]
    labels, gv, rv = [], [], []
    for s in SIZES:
        for b in BACKENDS:
            gs, _ = _load(s, "grep", b); rs, _ = _load(s, "vector", b)
            if gs or rs: labels.append(f"{b}\n({s})"); gv.append(np.mean(gs) if gs else 0); rv.append(np.mean(rs) if rs else 0)
    x = np.arange(len(labels))
    w = 0.35; ax.bar(x-w/2, gv, w, label="Grep", color=colors["grep"]); ax.bar(x+w/2, rv, w, label="RAG", color=colors["vector"])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("F1"); ax.set_title("Grep vs RAG"); ax.legend(); ax.set_ylim(0,1.05); ax.grid(axis="y", alpha=0.3)
    # 散点图
    ax = axes[0,1]
    for b in BACKENDS:
        gs, _ = _load(SIZES[0], "grep", b); rs, _ = _load(SIZES[0], "vector", b)
        if gs and rs: ax.scatter(gs, rs, alpha=0.6, label=b, s=30)
    ax.plot([0,1],[0,1],'k--',alpha=0.3); ax.set_xlabel("Grep F1"); ax.set_ylabel("RAG F1"); ax.set_title("每题对比"); ax.legend(); ax.grid(alpha=0.3)
    # F1 分布
    ax = axes[1,0]
    for m in MODES:
        all_s = []
        for b in BACKENDS: all_s.extend(_load(SIZES[0], m, b)[0])
        if all_s: ax.hist(all_s, bins=20, alpha=0.5, label=m, color=colors.get(m,"gray"))
    ax.set_xlabel("F1"); ax.set_ylabel("数"); ax.set_title("F1 分布"); ax.legend(); ax.grid(alpha=0.3)
    # 耗时
    ax = axes[1,1]
    bd, bl = [], []
    for m in MODES:
        for b in BACKENDS:
            _, ts = _load(SIZES[0], m, b)
            if ts: bd.append(ts); bl.append(f"{m}\n{b}")
    if bd:
        bp = ax.boxplot(bd, patch_artist=True)
        ax.set_xticklabels(bl, fontsize=8)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors.get(MODES[i%len(MODES)], "gray"))
    ax.set_ylabel("秒"); ax.set_title("耗时分布"); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = save_path or str(RESULTS_DIR / "comparison.png")
    plt.savefig(p, dpi=150); print(f"\n  图表 → {p}"); plt.close()

if __name__ == "__main__":
    table(); ascii_bar(); stats(); plot()
