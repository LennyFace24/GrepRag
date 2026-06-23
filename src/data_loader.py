"""
数据加载模块：从 CLongEval JSONL 文件加载数据, 写入对话上下文临时文件
"""
import json
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass
from src.config import DATA_DIR, CTX_DIR


@dataclass
class Question:
    """CLongEval 的一个测试样例"""
    id: str
    context: str       # 完整的多天对话文本
    query: str         # 用户提问
    answer: str        # 标准答案

    def write_context_file(self) -> Path:
        """将对话上下文写入临时文件, 返回文件路径 (供 grep 和 vector search 用)"""
        file_path = CTX_DIR / f"{self.id}.txt"
        file_path.write_text(self.context, encoding="utf-8")
        return file_path

    def context_file_path(self) -> Path:
        """获取已存在的上下文文件路径 (如果还没写入则返回预期路径)"""
        return CTX_DIR / f"{self.id}.txt"


def load_questions(dataset_size: str = "small", limit: int | None = None) -> list[Question]:
    """
    加载指定大小的 CLongEval 数据集

    Args:
        dataset_size: "small" | "medium" | "large"
        limit: 只加载前 N 条 (None = 全部)
    """
    file_path = DATA_DIR / f"{dataset_size}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")

    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # CLongEval long_conversation_memory 字段: context, query, answer, id
            q = Question(
                id=obj.get("id", f"unknown_{len(questions)}"),
                context=obj["context"],
                query=obj["query"],
                answer=obj["answer"],
            )
            questions.append(q)
            if limit and len(questions) >= limit:
                break

    return questions


def prepare_all_context_files(questions: list[Question]) -> list[Path]:
    """批量写入所有上下文文件, 返回文件路径列表"""
    paths = []
    for q in questions:
        paths.append(q.write_context_file())
    return paths


def get_question_count(dataset_size: str = "small") -> int:
    """快速获取某数据集的问题数量 (不加载全部数据)"""
    file_path = DATA_DIR / f"{dataset_size}.jsonl"
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count
