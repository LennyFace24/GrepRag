"""
全局配置：CLI agent, embedding 模型, 实验参数
所有可配置项通过 .env 文件或环境变量设置
复制 .env.example 为 .env 即可
"""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()

# ── 项目路径 ────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CTX_DIR = PROJECT_ROOT / "context_files"

RESULTS_DIR.mkdir(exist_ok=True)
CTX_DIR.mkdir(exist_ok=True)

# ── CLI Agent 配置 ──────────────────────────────────────
CLAUDE_CLI_MODEL = os.getenv("CLAUDE_CLI_MODEL", "")   # 留空=用 CLI 默认
CODEX_CLI_MODEL = os.getenv("CODEX_CLI_MODEL", "")     # 留空=用 CLI 默认
CLI_AGENT_TIMEOUT = int(os.getenv("CLI_AGENT_TIMEOUT", "300"))
CLI_TEMPERATURE = float(os.getenv("CLI_TEMPERATURE", "0.1"))  # CLI agent 推理温度

# ── Embedding 模型（vector search CLI 使用）─────────────
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ── 检索参数 ────────────────────────────────────────────
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
GREP_CONTEXT_LINES = int(os.getenv("GREP_CONTEXT_LINES", "1"))

# ── 实验矩阵 ────────────────────────────────────────────
def _parse_list(env_val: str, default: list[str]) -> list[str]:
    val = os.getenv(env_val, "")
    if not val.strip():
        return default
    return [item.strip() for item in val.split(",") if item.strip()]

DATASET_SIZES = _parse_list("DATASET_SIZES", ["small"])
TOOL_MODES = _parse_list("TOOL_MODES", ["grep", "vector"])
BACKENDS = _parse_list("BACKENDS", ["claude", "codex"])

# ── 运行控制 ────────────────────────────────────────────
_LIMIT_STR = os.getenv("LIMIT_QUESTIONS", "")
LIMIT_QUESTIONS = int(_LIMIT_STR) if _LIMIT_STR.strip() else None
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))

# ── 速率控制 / 反风控 ──────────────────────────────────
# 每次请求前随机延迟 (秒), 范围 [MIN, MAX]
RATE_LIMIT_DELAY_MIN = float(os.getenv("RATE_LIMIT_DELAY_MIN", "0.5"))
RATE_LIMIT_DELAY_MAX = float(os.getenv("RATE_LIMIT_DELAY_MAX", "2.0"))
# 触发 429 后等待秒数 (会指数退避: wait * 1, wait * 2, wait * 4...)
RATE_LIMIT_BACKOFF = float(os.getenv("RATE_LIMIT_BACKOFF", "5"))
RATE_LIMIT_MAX_RETRIES = int(os.getenv("RATE_LIMIT_MAX_RETRIES", "3"))
