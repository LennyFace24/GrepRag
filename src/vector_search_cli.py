#!/usr/bin/env python3
"""
向量语义搜索 CLI 工具
供 Claude Code / Codex CLI agent 通过 subprocess 调用

用法:
    python src/vector_search_cli.py <文件路径> <查询> [--top-k 5]

示例:
    python src/vector_search_cli.py context_files/abc.txt "小王子" --top-k 3
"""
import argparse
import sys
from pathlib import Path

# 确保能 import 同项目的模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding import BGEEmbedding, chunk_text
from src.config import VECTOR_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="向量语义搜索")
    parser.add_argument("file", help="要搜索的文本文件路径")
    parser.add_argument("query", help="自然语言查询")
    parser.add_argument("--top-k", type=int, default=VECTOR_TOP_K, help="返回结果数量")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[错误] 文件不存在: {args.file}")
        sys.exit(1)

    # 加载 embedding 模型
    print(f"[VectorSearch] 加载 BGE 模型...", file=sys.stderr)
    embedder = BGEEmbedding()

    # 分块 + embed
    text = file_path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    print(f"[VectorSearch] 共 {len(chunks)} 个文本块, 正在建索引...", file=sys.stderr)
    embeddings = embedder.embed(chunks)

    # 搜索
    query_vec = embedder.embed([args.query])
    scores = np.dot(embeddings, query_vec.T).flatten()
    top_indices = np.argsort(scores)[::-1][:args.top_k]

    for rank, idx in enumerate(top_indices, start=1):
        score = scores[idx]
        if score < 0.2:
            continue
        snippet = chunks[idx][:500]
        print(f"--- 结果 #{rank} (相似度: {score:.3f}) ---")
        print(snippet)
        print()

    if all(scores[idx] < 0.2 for idx in top_indices):
        print(f"[VectorSearch] 未找到与 '{args.query}' 语义相关的内容。")


if __name__ == "__main__":
    main()
