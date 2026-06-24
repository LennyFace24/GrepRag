#!/usr/bin/env python3
"""向量语义搜索 CLI，供 Codex/Claude agent 调用
用法: python src/vector_search_cli.py <文件> <查询> [--top-k 5]"""
import argparse, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embedding import create, chunk
from src.config import TOP_K, EMBED_PROVIDER, QWEN_EMBED_ENDPOINT, QWEN_EMBED_KEY, QWEN_EMBED_MODEL

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()
    f = Path(args.file)
    if not f.exists(): sys.exit(print(f"[错误] {args.file} 不存在", file=sys.stderr))
    if EMBED_PROVIDER == "qwen":
        e = create("qwen", model=QWEN_EMBED_MODEL)
    else:
        e = create("bge")
    chunks = chunk(f.read_text(encoding="utf-8"))
    print(f"[VectorSearch] {len(chunks)} chunks, 建索引中...", file=sys.stderr)
    emb = e.embed(chunks)
    qv = e.embed([args.query])
    sc = np.dot(emb, qv.T).flatten()
    top = np.argsort(sc)[::-1][:args.top_k]
    for i, idx in enumerate(top, 1):
        if sc[idx] < 0.2: continue
        print(f"--- #{i} (相似度:{sc[idx]:.3f}) ---\n{chunks[idx][:500]}\n")
    if all(sc[idx] < 0.2 for idx in top):
        print(f"未找到相关内容")

if __name__ == "__main__": main()
