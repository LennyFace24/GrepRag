"""
检索工具：Grep（正则匹配） 和 Vector（语义搜索）

两个检索工具都用于 Agent 的 function calling 机制。
Agent 调用 search(query) → 工具返回格式化的结果文本 → Agent 继续推理。
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.config import VECTOR_TOP_K, GREP_CONTEXT_LINES, CHUNK_SIZE, CHUNK_OVERLAP
from src.embedding import BaseEmbedding, chunk_text


# ──────────────────────────────────────────────────────
# 工具返回的数据结构
# ──────────────────────────────────────────────────────

@dataclass
class SearchHit:
    """单个检索命中"""
    rank: int
    content: str        # 匹配的文本内容
    score: str          # 分数标识 (grep: 匹配次数, vector: 相似度)
    context_before: str = ""  # 前文
    context_after: str = ""   # 后文


# ──────────────────────────────────────────────────────
# Tool Schema (给 LLM 用的 function definition)
# ──────────────────────────────────────────────────────

# ── Tool Schema 内部定义 (不包含外层 wrapper) ──────────

_GREP_TOOL_DEF = {
    "name": "grep_search",
    "description": (
        "在对话记录文件中执行关键词/正则搜索。"
        "返回包含搜索词的完整行以及前后各1行上下文。"
        "适用于查找精确信息，如人名、日期、具体事件名称等。"
        "注意：中文同义词无法通过此工具找到 —— 如果搜索'心情不好'就不会匹配到'情绪低落'。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要搜索的关键词或正则表达式。可以是单个词、短语、或正则模式。中文请直接写完整的词或短语。"
            }
        },
        "required": ["query"]
    }
}

_VECTOR_TOOL_DEF = {
    "name": "semantic_search",
    "description": (
        "在对话记录中执行语义搜索。"
        "返回与查询语义最相关的文本片段（top-5）。"
        "适用于查找意思相近但用词不同的内容 —— 即使原文用了不同的词或说法也能找到。"
        "例如搜索'不高兴'可以找到'心情不好'、'有点难过'等同义表达。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "自然语言查询，描述你想查找的信息。不要写正则表达式，写完整的自然语言问题或描述。"
            },
            "top_k": {
                "type": "integer",
                "description": "返回的结果数量 (默认5)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}

# ── Anthropic 格式 (直接用作 tools 参数) ─────────────────

GREP_TOOL_ANTHROPIC = _GREP_TOOL_DEF
VECTOR_TOOL_ANTHROPIC = _VECTOR_TOOL_DEF

# ── OpenAI 格式 (外层包 type: function) ──────────────────

GREP_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": _GREP_TOOL_DEF["name"],
        "description": _GREP_TOOL_DEF["description"],
        "parameters": _GREP_TOOL_DEF["input_schema"],
    }
}

VECTOR_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": _VECTOR_TOOL_DEF["name"],
        "description": _VECTOR_TOOL_DEF["description"],
        "parameters": _VECTOR_TOOL_DEF["input_schema"],
    }
}


# ──────────────────────────────────────────────────────
# Grep 检索
# ──────────────────────────────────────────────────────

class GrepRetriever:
    """
    基于正则表达式的文本检索

    直接对文本文件做逐行正则匹配，返回匹配行及上下文。
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lines: list[str] | None = None  # 惰性加载

    @property
    def lines(self) -> list[str]:
        if self._lines is None:
            self._lines = self.file_path.read_text(encoding="utf-8").split("\n")
        return self._lines

    def search(self, query: str, context_lines: int = GREP_CONTEXT_LINES) -> str:
        """
        执行 grep 搜索

        Args:
            query: 搜索词或正则模式
            context_lines: 匹配行前后各展示几行

        Returns:
            格式化的搜索结果字符串
        """
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            # 如果不是合法正则, 尝试转义为普通字符串匹配
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        hits: list[tuple[int, str]] = []  # (行号, 行内容)
        for i, line in enumerate(self.lines):
            if pattern.search(line):
                hits.append((i, line))

        if not hits:
            return f"[grep] 未找到匹配 '{query}' 的内容。建议：尝试换一个关键词、缩短搜索词、或使用语义搜索工具。"

        # 格式化输出
        result_parts = [f"[grep] 搜索 '{query}' 找到 {len(hits)} 处匹配:\n"]
        for rank, (line_no, line_text) in enumerate(hits[:20], start=1):  # 最多返回20条
            # 添加上下文
            ctx_before = ""
            ctx_after = ""
            if context_lines > 0:
                for j in range(max(0, line_no - context_lines), line_no):
                    ctx_before += self.lines[j] + "\n"
                for j in range(line_no + 1, min(len(self.lines), line_no + context_lines + 1)):
                    ctx_after += self.lines[j] + "\n"

            result_parts.append(
                f"--- 匹配 #{rank} (第 {line_no + 1} 行) ---\n"
                f"{ctx_before}"
                f">>> {line_text}\n"
                f"{ctx_after}"
            )

        if len(hits) > 20:
            result_parts.append(f"\n...（共 {len(hits)} 条, 仅展示前 20 条。请缩小搜索范围以获取更精准结果）")

        return "\n".join(result_parts)


# ──────────────────────────────────────────────────────
# Vector / RAG 检索
# ──────────────────────────────────────────────────────

class VectorRetriever:
    """
    基于 embedding 的语义检索

    初始化时：将文本分块 → embed → 建向量索引
    查询时：embed query → 余弦相似度 → 返回 top-k 文本块
    """

    def __init__(self, file_path: Path, embedder: BaseEmbedding, top_k: int = VECTOR_TOP_K):
        self.file_path = file_path
        self.embedder = embedder
        self.top_k = top_k

        # 分块 + 建索引
        text = file_path.read_text(encoding="utf-8")
        self._chunks = chunk_text(text)
        self._embeddings: np.ndarray | None = None  # 惰性索引 (首次搜索时才 build)

    def _build_index(self):
        """构建向量索引 (惰性)"""
        if self._embeddings is not None:
            return
        print(f"  [VectorRetriever] 正在为文件建立向量索引 (共 {len(self._chunks)} 个 chunk)...")
        self._embeddings = self.embedder.embed(self._chunks)

    def search(self, query: str, top_k: int | None = None) -> str:
        """
        语义搜索

        Args:
            query: 自然语言查询
            top_k: 返回数量 (覆盖 self.top_k)

        Returns:
            格式化的搜索结果字符串
        """
        self._build_index()
        k = top_k or self.top_k

        # Embed 查询
        query_vec = self.embedder.embed([query])
        query_vec = query_vec.reshape(1, -1) if query_vec.ndim == 1 else query_vec

        # 余弦相似度 (embedding 已归一化, 直接点积即可)
        scores = np.dot(self._embeddings, query_vec.T).flatten()

        # Top-K
        top_indices = np.argsort(scores)[::-1][:k]

        result_parts = [f"[RAG] 语义搜索 '{query}' 的 Top-{k} 结果:\n"]
        for rank, idx in enumerate(top_indices, start=1):
            score = scores[idx]
            if score < 0.2:   # 相似度太低, 基本无关
                continue
            chunk_text_snippet = self._chunks[idx][:500]  # 截断过长文本
            result_parts.append(
                f"--- 结果 #{rank} (相似度: {score:.3f}) ---\n"
                f"{chunk_text_snippet}\n"
            )

        if len([1 for idx in top_indices if scores[idx] >= 0.2]) == 0:
            return f"[RAG] 未找到与 '{query}' 语义相关的内容。"

        return "\n".join(result_parts)


# ──────────────────────────────────────────────────────
# 工具路由
# ──────────────────────────────────────────────────────

def get_tool_search_fn(tool_mode: str, file_path: Path, embedder: BaseEmbedding | None = None) -> Callable[[str, ...], str]:
    """
    根据模式返回对应的搜索函数

    Args:
        tool_mode: "grep" | "vector"
        file_path: 对话文本文件路径
        embedder: vector 模式必需的 embedding 模型
    """
    if tool_mode == "grep":
        retriever = GrepRetriever(file_path)
        return retriever.search
    elif tool_mode == "vector":
        if embedder is None:
            raise ValueError("Vector 模式需要提供 embedder")
        retriever = VectorRetriever(file_path, embedder)
        return retriever.search
    else:
        raise ValueError(f"不支持的 tool_mode: {tool_mode}。可选: grep, vector")


def get_tool_schema(tool_mode: str, backend: str = "anthropic") -> dict:
    """返回对应模式的 tool schema (给 LLM function calling)"""
    if tool_mode == "grep":
        return GREP_TOOL_ANTHROPIC if backend == "anthropic" else GREP_TOOL_OPENAI
    elif tool_mode == "vector":
        return VECTOR_TOOL_ANTHROPIC if backend == "anthropic" else VECTOR_TOOL_OPENAI
    else:
        raise ValueError(f"不支持的 tool_mode: {tool_mode}。可选: grep, vector")
