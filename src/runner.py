"""实验运行器：遍历矩阵 → CLI agent → 收集结果"""
import json, time, threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import RESULTS_DIR, CTX_DIR, SIZES, MODES, BACKENDS, LIMIT, WORKERS, CLAUDE_MODEL, CODEX_MODEL, EMBED_PROVIDER, EMBED_MODEL, QWEN_EMBED_MODEL
from src.data_loader import load, prepare
from src.cli_agent import CLIAgentRunner

_WLOCK = threading.Lock()

def _path(size, mode, backend):
    return RESULTS_DIR / f"{size}_{mode}_{backend}.jsonl"

def _completed(p):
    if not p.exists(): return set()
    return {json.loads(l).get("question_id","") for l in open(p,encoding="utf-8") if l.strip()}

def _save(p, rec):
    with _WLOCK:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _run_one(q, idx, total, size, mode, backend, model):
    ctx = q.ctx_path()
    agent = CLIAgentRunner(backend, mode, model or None)
    print(f"\n  [{idx}/{total}] Q: {q.query[:80]}...")
    t0 = time.time()
    try:
        ans, raw = agent.run(q.query, ctx)
    except Exception as e:
        ans, raw = f"[ERROR] {e}", ""
        print(f"    ✗ {e}")
    t = time.time() - t0
    print(f"    A: {ans[:100]}...")
    print(f"    耗时: {t:.1f}s")
    _save(_path(size, mode, backend), {
        "question_id": q.id, "dataset_size": size, "tool_mode": mode, "backend": backend,
        "query": q.query, "ground_truth": q.answer, "prediction": ans,
        "raw_output": raw[:5000], "elapsed_seconds": round(t,2),
        "timestamp": datetime.now().isoformat(),
    })

def run_experiment(size="small", mode="grep", backend="claude"):
    emb_info = f"{EMBED_PROVIDER}/{QWEN_EMBED_MODEL}" if EMBED_PROVIDER == "qwen" else f"{EMBED_PROVIDER}/{EMBED_MODEL}"
    print(f"\n{'='*50}\n  实验: {size} | {mode} | {backend} | embed={emb_info}\n{'='*50}")
    qs = load(size, LIMIT)
    prepare(qs)
    remaining = [q for q in qs if q.id not in _completed(_path(size, mode, backend))]
    print(f"  已完成: {len(qs)-len(remaining)}, 剩余: {len(remaining)}")
    if not remaining: return print("  ✓ 全部完成")
    model = CLAUDE_MODEL if backend == "claude" else CODEX_MODEL
    if WORKERS > 1: print(f"  并发: {WORKERS} workers")
    if WORKERS <= 1:
        for i, q in enumerate(remaining, 1):
            _run_one(q, i, len(remaining), size, mode, backend, model)
    else:
        with ThreadPoolExecutor(WORKERS) as ex:
            fs = {ex.submit(_run_one, q, i+1, len(remaining), size, mode, backend, model): q
                  for i, q in enumerate(remaining)}
            for f in as_completed(fs):
                try: f.result()
                except Exception as e: print(f"    ✗ {fs[f].id}: {e}")
    print(f"\n  ✓ [{size}/{mode}/{backend}] 完成 → {_path(size,mode,backend)}")

def run_all():
    for s in SIZES:
        for m in MODES:
            for b in BACKENDS:
                run_experiment(s, m, b)
    print(f"\n{'#'*50}\n  全部完成 → {RESULTS_DIR}\n{'#'*50}")

if __name__ == "__main__":
    run_all()
