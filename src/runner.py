"""
实验运行器：遍历实验矩阵，通过 CLI agent 收集结果
支持断点续传 + 并发
"""
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import (
    RESULTS_DIR, CTX_DIR,
    DATASET_SIZES, TOOL_MODES, BACKENDS,
    LIMIT_QUESTIONS, MAX_WORKERS,
    CLAUDE_CLI_MODEL, CODEX_CLI_MODEL,
)
from src.data_loader import load_questions, prepare_all_context_files
from src.cli_agent import CLIAgentRunner


def get_result_path(dataset_size: str, tool_mode: str, backend: str) -> Path:
    filename = f"{dataset_size}_{tool_mode}_{backend}.jsonl"
    return RESULTS_DIR / filename


def load_completed_ids(result_path: Path) -> set[str]:
    if not result_path.exists():
        return set()
    completed = set()
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                completed.add(obj.get("question_id", ""))
            except json.JSONDecodeError:
                continue
    return completed


def save_result(result_path: Path, record: dict):
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_experiment(
    dataset_size: str = "small",
    tool_mode: str = "grep",
    backend: str = "claude",
):
    print(f"\n{'='*60}")
    print(f"  实验配置: {dataset_size} | {tool_mode} | {backend}")
    print(f"{'='*60}\n")

    questions = load_questions(dataset_size, limit=LIMIT_QUESTIONS)
    print(f"  加载了 {len(questions)} 个问题")

    prepare_all_context_files(questions)
    print(f"  上下文文件已写入 {CTX_DIR}")

    result_path = get_result_path(dataset_size, tool_mode, backend)
    completed_ids = load_completed_ids(result_path)
    remaining = [q for q in questions if q.id not in completed_ids]
    print(f"  已完成: {len(completed_ids)}, 剩余: {len(remaining)}")

    if not remaining:
        print(f"  ✓ 全部完成, 跳过")
        return

    model = CLAUDE_CLI_MODEL if backend == "claude" else CODEX_CLI_MODEL

    if MAX_WORKERS > 1:
        print(f"  并发模式: {MAX_WORKERS} workers")

    _write_lock = threading.Lock()

    def run_one_question(q, idx, total):
        context_file = q.context_file_path()
        agent = CLIAgentRunner(backend=backend, tool_mode=tool_mode, model_name=model or None)

        print(f"\n  [{idx}/{total}] Q: {q.query[:80]}...")
        t_start = time.time()

        try:
            answer, raw_output = agent.run(q.query, context_file)
        except Exception as e:
            answer = f"[ERROR] {e}"
            raw_output = ""
            print(f"    ✗ 异常: {e}")

        elapsed = time.time() - t_start
        print(f"    A: {answer[:100]}...")
        print(f"    耗时: {elapsed:.1f}s")

        record = {
            "question_id": q.id,
            "dataset_size": dataset_size,
            "tool_mode": tool_mode,
            "backend": backend,
            "query": q.query,
            "ground_truth": q.answer,
            "prediction": answer,
            "raw_output": raw_output[:5000],   # 截断，防止日志爆炸
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
        }

        with _write_lock:
            save_result(result_path, record)

        return record

    if MAX_WORKERS <= 1:
        for idx, q in enumerate(remaining, start=1):
            run_one_question(q, idx, len(remaining))
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(run_one_question, q, i + 1, len(remaining)): q
                for i, q in enumerate(remaining)
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    q = futures[future]
                    print(f"    ✗ 问题 {q.id} 执行失败: {e}")

    print(f"\n  ✓ 配置 [{dataset_size}/{tool_mode}/{backend}] 完成!")
    print(f"    结果保存至: {result_path}")


def run_all():
    total_configs = len(DATASET_SIZES) * len(TOOL_MODES) * len(BACKENDS)
    print(f"\n{'#'*60}")
    print(f"  Grep vs RAG 中文检索对比实验 (CLI Agent)")
    print(f"  矩阵: {len(DATASET_SIZES)} sizes × {len(TOOL_MODES)} modes × {len(BACKENDS)} backends = {total_configs} 组")
    print(f"  Backends: {BACKENDS}")
    print(f"  {'#'*60}")

    for dataset_size in DATASET_SIZES:
        for tool_mode in TOOL_MODES:
            for backend in BACKENDS:
                run_experiment(
                    dataset_size=dataset_size,
                    tool_mode=tool_mode,
                    backend=backend,
                )

    print(f"\n{'#'*60}")
    print(f"  所有实验完成! 结果目录: {RESULTS_DIR}")
    print(f"  {'#'*60}")


if __name__ == "__main__":
    run_all()
