"""
全局配置：API keys, 模型路径, 实验参数
所有可配置项都可以通过 .env 文件或环境变量设置
复制 .env.example 为 .env 并填入你的真实配置即可
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── 加载 .env 文件 ────────────────────────────────────────
# 优先找项目根目录下的 .env
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()  # 回退到当前目录

# ── 项目路径 ────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CTX_DIR = PROJECT_ROOT / "context_files"   # 存放对话文本临时文件

# 确保必要目录存在
RESULTS_DIR.mkdir(exist_ok=True)
CTX_DIR.mkdir(exist_ok=True)

# ── API Keys ────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── API Base URL（留空用官方默认）───────────────────────
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# ── LLM 模型配置 ────────────────────────────────────────
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── Embedding 模型配置 ───────────────────────────────────
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ── 检索参数 ────────────────────────────────────────────
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
GREP_CONTEXT_LINES = int(os.getenv("GREP_CONTEXT_LINES", "1"))

# ── Agent 参数 ──────────────────────────────────────────
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "8"))
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))

# ── 实验矩阵 (逗号分隔, 如 "small,medium") ──────────────
def _parse_list(env_val: str, default: list[str]) -> list[str]:
    """解析逗号分隔的环境变量为列表"""
    val = os.getenv(env_val, "")
    if not val.strip():
        return default
    return [item.strip() for item in val.split(",") if item.strip()]

DATASET_SIZES = _parse_list("DATASET_SIZES", ["small"])
TOOL_MODES = _parse_list("TOOL_MODES", ["grep", "vector"])
BACKENDS = _parse_list("BACKENDS", ["anthropic", "openai"])

# ── 运行控制 ────────────────────────────────────────────
_LIMIT_STR = os.getenv("LIMIT_QUESTIONS", "")
LIMIT_QUESTIONS = int(_LIMIT_STR) if _LIMIT_STR.strip() else None

RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))
