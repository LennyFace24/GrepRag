"""
结果报告模块：汇总实验结果，输出对比表格
"""
import json
from pathlib import Path

from src.config import RESULTS_DIR, DATASET_SIZES, TOOL_MODES, BACKENDS
from src.scorer import qa_f1_zh_score, compute_accuracy


def load_all_results(dataset_size: str, tool_mode: str, backend: str) -> list[dict]:
    """加载某组实验结果"""
    filename = f"{dataset_size}_{tool_mode}_{backend}.jsonl"
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        return []
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def calculate_scores(results: list[dict]) -> dict:
    """计算一组结果的评分统计"""
    if not results:
        return {"count": 0, "avg_f1": 0.0, "accuracy": 0.0, "errors": 0}

    f1_scores = []
    errors = 0
    for r in results:
        pred = r.get("prediction", "")
        gt = r.get("ground_truth", "")
        if pred.startswith("[ERROR]") or pred.startswith("[TIMEOUT]"):
            errors += 1
            f1_scores.append(0.0)
        else:
            f1_scores.append(qa_f1_zh_score(pred, gt))

    return {
        "count": len(results),
        "avg_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "accuracy": round(compute_accuracy(f1_scores, threshold=0.5), 4),
        "errors": errors,
    }


def print_report():
    """打印完整实验结果报告"""
    print("\n" + "=" * 90)
    print("                    Grep vs RAG 中文检索对比 — 实验结果")
    print("=" * 90)

    for dataset_size in DATASET_SIZES:
        print(f"\n{'─' * 90}")
        print(f"  数据集: {dataset_size}")
        print(f"{'─' * 90}")

        # 表头
        print(f"\n  {'Backend':<14} {'Grep F1':>10} {'Grep Acc':>10} {'RAG F1':>10} {'RAG Acc':>10} {'Δ F1':>10}")
        print(f"  {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")

        for backend in BACKENDS:
            grep_results = load_all_results(dataset_size, "grep", backend)
            rag_results = load_all_results(dataset_size, "vector", backend)

            grep_scores = calculate_scores(grep_results)
            rag_scores = calculate_scores(rag_results)

            delta_f1 = round(grep_scores["avg_f1"] - rag_scores["avg_f1"], 4)

            print(
                f"  {backend:<14} "
                f"{grep_scores['avg_f1']:>10.4f} "
                f"{grep_scores['accuracy']:>10.4f} "
                f"{rag_scores['avg_f1']:>10.4f} "
                f"{rag_scores['accuracy']:>10.4f} "
                f"{delta_f1:>+10.4f}"
            )

    print(f"\n{'=' * 90}")
    print(f"  Δ F1 = Grep_F1 - RAG_F1 (正值 = grep 更优, 负值 = RAG 更优)")
    print(f"{'=' * 90}\n")


def export_detailed_report():
    """导出每题详细结果到 CSV"""
    import csv

    csv_path = RESULTS_DIR / "detailed_results.csv"
    all_rows = []

    for dataset_size in DATASET_SIZES:
        for tool_mode in TOOL_MODES:
            for backend in BACKENDS:
                results = load_all_results(dataset_size, tool_mode, backend)
                for r in results:
                    pred = r.get("prediction", "")
                    gt = r.get("ground_truth", "")
                    f1 = qa_f1_zh_score(pred, gt) if not pred.startswith("[ERROR]") else 0.0
                    all_rows.append({
                        "dataset_size": dataset_size,
                        "tool_mode": tool_mode,
                        "backend": backend,
                        "question_id": r.get("question_id", ""),
                        "query": r.get("query", ""),
                        "ground_truth": gt,
                        "prediction": pred,
                        "f1_score": round(f1, 4),
                        "elapsed_seconds": r.get("elapsed_seconds", 0),
                    })

    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  详细结果导出至: {csv_path} ({len(all_rows)} 条记录)")


if __name__ == "__main__":
    print_report()
    export_detailed_report()
