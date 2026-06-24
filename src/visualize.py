"""
结果可视化：生成对比图表
"""
import json
from pathlib import Path
import numpy as np

from src.config import RESULTS_DIR, DATASET_SIZES, TOOL_MODES, BACKENDS
from src.scorer import qa_f1_zh_score


def _load_scores(dataset_size, tool_mode, backend):
    file_path = RESULTS_DIR / f"{dataset_size}_{tool_mode}_{backend}.jsonl"
    if not file_path.exists():
        return [], []
    f1_scores, elapsed = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pred = r.get("prediction", "")
            gt = r.get("ground_truth", "")
            if pred.startswith("[ERROR]") or pred.startswith("[TIMEOUT]") or pred.startswith("[RATE_LIMITED]"):
                f1_scores.append(0.0)
            else:
                f1_scores.append(qa_f1_zh_score(pred, gt))
            elapsed.append(r.get("elapsed_seconds", 0))
    return f1_scores, elapsed


# ──────────────────────────────────────────────────────
# 纯文本可视化（无需 matplotlib，终端直接看）
# ──────────────────────────────────────────────────────

def print_ascii_bar_chart():
    """终端 ASCII 柱状图"""
    print("\n" + "=" * 60)
    print("  F1 分数对比 (柱状图)")
    print("=" * 60)

    for ds in DATASET_SIZES:
        print(f"\n  数据集: {ds}")
        print(f"  {'─' * 50}")

        results = {}
        max_label = 0
        for tm in TOOL_MODES:
            for be in BACKENDS:
                scores, _ = _load_scores(ds, tm, be)
                if not scores:
                    continue
                label = f"  {tm}/{be}"
                max_label = max(max_label, len(label))
                avg = np.mean(scores)
                results[label] = avg

        if not results:
            print("    (无数据)")
            continue

        bar_max = 50 - max_label - 4
        max_score = max(results.values()) or 1.0
        for label in sorted(results.keys()):
            avg = results[label]
            bar_len = int(avg / max_score * bar_max) if max_score > 0 else 0
            bar = "█" * bar_len + "░" * (bar_max - bar_len)
            print(f"{label:<{max_label}}  {bar}  {avg:.3f}")


def print_score_table():
    """F1 + Accuracy 表格"""
    print("\n" + "=" * 80)
    print("  Grep vs RAG 实验结果")
    print("=" * 80)

    for ds in DATASET_SIZES:
        print(f"\n  数据集: {ds}")
        header = f"  {'Backend':<12} {'Grep F1':>9} {'Grep Acc':>9}  {'RAG F1':>9} {'RAG Acc':>9}  {'Δ F1':>9}"
        print(header)
        print(f"  {'─' * (len(header) - 2)}")

        for be in BACKENDS:
            grep_s, _ = _load_scores(ds, "grep", be)
            rag_s, _ = _load_scores(ds, "vector", be)

            if not grep_s and not rag_s:
                continue

            grep_f1 = np.mean(grep_s) if grep_s else 0
            rag_f1 = np.mean(rag_s) if rag_s else 0
            grep_acc = np.mean([s >= 0.5 for s in grep_s]) if grep_s else 0
            rag_acc = np.mean([s >= 0.5 for s in rag_s]) if rag_s else 0
            delta = grep_f1 - rag_f1

            print(f"  {be:<12} {grep_f1:>9.4f} {grep_acc:>9.4f}  {rag_f1:>9.4f} {rag_acc:>9.4f}  {delta:>+9.4f}")

    print(f"\n  Δ F1 = Grep - RAG (正=grep优, 负=RAG优)")


def print_per_question_detail():
    """每题详细对比（前 10 题）"""
    print("\n" + "=" * 70)
    print("  每题得分详情 (前 10 题)")
    print("=" * 70)

    for ds in DATASET_SIZES:
        for be in BACKENDS:
            grep_s, _ = _load_scores(ds, "grep", be)
            rag_s, _ = _load_scores(ds, "vector", be)
            if not grep_s and not rag_s:
                continue

            print(f"\n  {ds}/{be}")
            print(f"  {'#':<4} {'Grep':>7} {'RAG':>7} {'Win':>6}")
            print(f"  {'─' * 30}")

            n = min(10, max(len(grep_s), len(rag_s)))
            for i in range(n):
                g = grep_s[i] if i < len(grep_s) else 0
                r = rag_s[i] if i < len(rag_s) else 0
                if g > r:
                    win = "Grep"
                elif r > g:
                    win = "RAG"
                else:
                    win = "Tie"
                print(f"  {i+1:<4} {g:>7.4f} {r:>7.4f} {win:>6}")


def print_summary_stats():
    """汇总统计"""
    print("\n" + "=" * 60)
    print("  耗时统计")
    print("=" * 60)

    for ds in DATASET_SIZES:
        for tm in TOOL_MODES:
            for be in BACKENDS:
                _, elapsed = _load_scores(ds, tm, be)
                if not elapsed:
                    continue
                print(f"  {ds}/{tm}/{be}: avg={np.mean(elapsed):.1f}s, "
                      f"max={np.max(elapsed):.0f}s, total={np.sum(elapsed):.0f}s")


# ──────────────────────────────────────────────────────
# matplotlib 可视化（需要 pip install matplotlib）
# ──────────────────────────────────────────────────────

def plot_comparison(save_path: str | None = None):
    """生成完整的对比图（需要 matplotlib）"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("\n  [跳过] matplotlib 未安装，无法生成图表。安装: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {"grep": "#4ECDC4", "vector": "#FF6B6B"}

    # ── 1. 柱状图: 各配置 F1 对比 ──────────────────────
    ax = axes[0, 0]
    labels, grep_vals, rag_vals = [], [], []
    for ds in DATASET_SIZES:
        for be in BACKENDS:
            gs, _ = _load_scores(ds, "grep", be)
            rs, _ = _load_scores(ds, "vector", be)
            if gs or rs:
                labels.append(f"{be}\n({ds})")
                grep_vals.append(np.mean(gs) if gs else 0)
                rag_vals.append(np.mean(rs) if rs else 0)

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, grep_vals, w, label="Grep", color=colors["grep"])
    ax.bar(x + w/2, rag_vals, w, label="RAG", color=colors["vector"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("Grep vs RAG F1 Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # ── 2. 散点图: 每题 grep vs RAG ──────────────────
    ax = axes[0, 1]
    for be in BACKENDS:
        gs, _ = _load_scores(DATASET_SIZES[0], "grep", be)
        rs, _ = _load_scores(DATASET_SIZES[0], "vector", be)
        if gs and rs:
            ax.scatter(gs, rs, alpha=0.6, label=be, s=30)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="y=x (grep=RAG)")
    ax.set_xlabel("Grep F1")
    ax.set_ylabel("RAG F1")
    ax.set_title("Per-Question: Grep vs RAG")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 3. 直方图: F1 分布 ────────────────────────────
    ax = axes[1, 0]
    for tm in TOOL_MODES:
        all_scores = []
        for be in BACKENDS:
            ss, _ = _load_scores(DATASET_SIZES[0], tm, be)
            all_scores.extend(ss)
        if all_scores:
            ax.hist(all_scores, bins=20, alpha=0.5, label=tm, color=colors.get(tm, "gray"))

    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Count")
    ax.set_title("F1 Score Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 4. 箱线图: 耗时 ───────────────────────────────
    ax = axes[1, 1]
    box_data, box_labels = [], []
    for tm in TOOL_MODES:
        for be in BACKENDS:
            _, elapsed = _load_scores(DATASET_SIZES[0], tm, be)
            if elapsed:
                box_data.append(elapsed)
                box_labels.append(f"{tm}\n{be}")

    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True)
        ax.set_xticklabels(box_labels, fontsize=8)
        for patch, tm in zip(bp['boxes'], [t for t in TOOL_MODES for _ in BACKENDS]):
            patch.set_facecolor(colors.get(tm, "gray"))
    ax.set_ylabel("Seconds")
    ax.set_title("Response Time Distribution")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = save_path or str(RESULTS_DIR / "comparison.png")
    plt.savefig(out, dpi=150)
    print(f"\n  图表已保存至: {out}")
    plt.close()


# ──────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print_score_table()
    print_ascii_bar_chart()
    print_summary_stats()
    print_per_question_detail()
    plot_comparison()
