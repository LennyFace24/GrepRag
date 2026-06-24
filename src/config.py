"""全局配置，通过 .env 文件设置"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env" if (ROOT / ".env").exists() else None)

def _env(key, default=""):
    return os.getenv(key, default).split("#")[0].strip()

def _list(key, default):
    v = _env(key, "")
    return [x.strip() for x in v.split(",") if x.strip()] if v else default

# 路径
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CTX_DIR = ROOT / "context_files"
RESULTS_DIR.mkdir(exist_ok=True)
CTX_DIR.mkdir(exist_ok=True)

# CLI Agent
CLAUDE_MODEL = _env("CLAUDE_CLI_MODEL")
CODEX_MODEL = _env("CODEX_CLI_MODEL")
CLI_TIMEOUT = int(_env("CLI_AGENT_TIMEOUT") or 300)
CLI_TEMP = float(_env("CLI_TEMPERATURE") or 0.1)

# Embedding
EMBED_MODEL = _env("EMBEDDING_MODEL") or "BAAI/bge-small-zh"
EMBED_DEVICE = _env("EMBEDDING_DEVICE") or "cpu"

# 检索参数
TOP_K = int(_env("VECTOR_TOP_K") or 5)
CHUNK_SIZE = int(_env("CHUNK_SIZE") or 300)
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP") or 50)

# 实验矩阵
SIZES = _list("DATASET_SIZES", ["small"])
MODES = _list("TOOL_MODES", ["grep", "vector"])
BACKENDS = _list("BACKENDS", ["claude", "codex"])

# 运行控制
LIMIT = _env("LIMIT_QUESTIONS")
LIMIT = int(LIMIT) if LIMIT else None
WORKERS = int(_env("MAX_WORKERS") or 1)
RETRIES = int(_env("RETRY_COUNT") or 3)

# 反风控
DELAY_MIN = float(_env("RATE_LIMIT_DELAY_MIN") or 0.5)
DELAY_MAX = float(_env("RATE_LIMIT_DELAY_MAX") or 2.0)
RL_BACKOFF = float(_env("RATE_LIMIT_BACKOFF") or 5)
RL_RETRIES = int(_env("RATE_LIMIT_MAX_RETRIES") or 3)
