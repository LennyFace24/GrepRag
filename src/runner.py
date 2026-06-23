"""
实验运行器：遍历实验矩阵，运行 Agent，收集结果
支持断点续传 —— 已跑过的问题会自动跳过
"""
import json
import time
import sys
from pathlib import Path
from datetime import datetime

from src.config import (
    RESULTS_DIR, CTX_DIR,
    DATASET_SIZES, TOOL_MODES, BACKENDS,
    LIMIT_QUESTIONS, ANTHROPIC_MODEL, OPENAI_MODEL,
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
)
from src.data_loader import load_questions, prepare_all_context_files
from src.agent import AgentRunner


def should_skip_backend(backend: str) -> str | None:
    """
    检查某个 backend 是否应该跳过。
    返回 None 表示可以运行，返回字符串表示跳过原因。
    """
    if backend == "anthropic":
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("sk-ant-xxxxx"):
            return "未配置 ANTHROPIC_API_KEY（占位符或为空）"
    elif backend == "openai":
        if not OPENAI_API_KEY:
            return "未配置 OPENAI_API_KEY"
    return None


def get_result_path(dataset_size: str, tool_mode: str, backend: str) -> Path:
    """获取某组实验配置的结果文件路径"""
    filename = f"{dataset_size}_{tool_mode}_{backend}.jsonl"
    return RESULTS_DIR / filename


def load_completed_ids(result_path: Path) -> set[str]:
    """加载已跑过的问题 ID (用于断点续传)"""
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
    """追加一条实验结果到 JSONL 文件"""
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_experiment(
    dataset_size: str = "small",
    tool_mode: str = "grep",
    backend: str = "anthropic",
):
    """
    运行一组实验配置

    Args:
        dataset_size: 数据集大小 ("small" | "medium" | "large")
        tool_mode: 检索模式 ("grep" | "vector")
        backend: LLM backend ("anthropic" | "openai")
    """
    print(f"\n{'='*60}")
    print(f"  实验配置: {dataset_size} | {tool_mode} | {backend}")
    print(f"{'='*60}\n")

    # 加载数据
    questions = load_questions(dataset_size, limit=LIMIT_QUESTIONS)
    print(f"  加载了 {len(questions)} 个问题")

    # 写入上下文文件
    prepare_all_context_files(questions)
    print(f"  上下文文件已写入 {CTX_DIR}")

    # 断点续传
    result_path = get_result_path(dataset_size, tool_mode, backend)
    completed_ids = load_completed_ids(result_path)
    remaining = [q for q in questions if q.id not in completed_ids]
    print(f"  已完成: {len(completed_ids)}, 剩余: {len(remaining)}")

    if not remaining:
        print(f"  ✓ 全部完成, 跳过")
        return

    # 初始化 Agent
    print(f"  初始化 Agent...")
    agent = AgentRunner(backend=backend, tool_mode=tool_mode)

    # 逐题运行
    for idx, q in enumerate(remaining, start=1):
        context_file = q.context_file_path()

        print(f"\n  [{idx}/{len(remaining)}] Q: {q.query[:80]}...")

        t_start = time.time()

        try:
            answer = agent.run(q.query, context_file)
        except Exception as e:
            answer = f"[ERROR] {e}"
            print(f"    ✗ 异常: {e}")

        elapsed = time.time() - t_start
        print(f"    A: {answer[:100]}...")
        print(f"    耗时: {elapsed:.1f}s")

        record = {
            "question_id": q.id,
            "dataset_size": dataset_size,
            "tool_mode": tool_mode,
            "backend": backend,
            "model": agent._model,
            "query": q.query,
            "ground_truth": q.answer,
            "prediction": answer,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
        }
        save_result(result_path, record)

    print(f"\n  ✓ 配置 [{dataset_size}/{tool_mode}/{backend}] 完成!")
    print(f"    结果保存至: {result_path}")


def run_all():
    """运行完整的实验矩阵"""
    total_configs = len(DATASET_SIZES) * len(TOOL_MODES) * len(BACKENDS)
    print(f"\n{'#'*60}")
    print(f"  Grep vs RAG 中文检索对比实验")
    print(f"  实验矩阵: {len(DATASET_SIZES)} sizes × {len(TOOL_MODES)} modes × {len(BACKENDS)} backends = {total_configs} 组")
    print(f"  模型: Anthropic={ANTHROPIC_MODEL}, OpenAI={OPENAI_MODEL}")
    print(f"  {'#'*60}")

    skipped_all = True
    for dataset_size in DATASET_SIZES:
        for tool_mode in TOOL_MODES:
            for backend in BACKENDS:
                skip_reason = should_skip_backend(backend)
                if skip_reason:
                    print(f"\n  ⏭ 跳过 [{dataset_size}/{tool_mode}/{backend}]: {skip_reason}")
                    continue
                skipped_all = False
                run_experiment(
                    dataset_size=dataset_size,
                    tool_mode=tool_mode,
                    backend=backend,
                )

    if skipped_all:
        print(f"\n  ⚠ 所有 backend 都未配置 API Key，无法运行任何实验。")
        print(f"  请编辑 .env 文件，填入至少一个真实的 API Key。")

    print(f"\n{'#'*60}")
    print(f"  所有实验完成!")
    print(f"  结果目录: {RESULTS_DIR}")
    print(f"  {'#'*60}")


if __name__ == "__main__":
    run_all()
