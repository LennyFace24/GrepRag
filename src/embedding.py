"""可插拔 Embedding：BGE(本地) / Qwen(API, TCP 长连接复用)"""
import json
import http.client
from abc import ABC, abstractmethod
import numpy as np
from src.config import EMBED_MODEL, EMBED_DEVICE, CHUNK_SIZE, CHUNK_OVERLAP

# 全局复用 TCP 连接
_conn: http.client.HTTPSConnection | None = None

def _get_conn() -> http.client.HTTPSConnection:
    global _conn
    if _conn is None:
        _conn = http.client.HTTPSConnection("ai.gitee.com")
    return _conn

class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...
    @property
    @abstractmethod
    def dim(self) -> int: ...

class BGEEmbedding(BaseEmbedding):
    """本地 BGE-small-zh"""
    def __init__(self, model: str = EMBED_MODEL, device: str = EMBED_DEVICE):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    @property
    def dim(self) -> int: return self._m.get_sentence_embedding_dimension()

class QwenEmbedding(BaseEmbedding):
    """Qwen Embedding (ai.gitee.com), stdlib http.client + TCP 长连接"""
    def __init__(self, endpoint: str, key: str = "", model: str = "Qwen3-Embedding-0.6B"):
        self._model = model
        self._key = key
        self._dim = 1024

    def embed(self, texts: list[str]) -> np.ndarray:
        body = json.dumps({
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        })
        headers = {"Content-Type": "application/json"}
        conn = _get_conn()
        conn.request("POST", "/v1/embeddings", body, headers)
        r = conn.getresponse()
        data = json.loads(r.read().decode("utf-8"))
        items = sorted(data["data"], key=lambda x: x["index"])
        return np.array([it["embedding"] for it in items], dtype=np.float32)

    @property
    def dim(self) -> int: return self._dim

def create(provider: str = "bge", **kw) -> BaseEmbedding:
    return {"bge": BGEEmbedding, "qwen": QwenEmbedding}[provider](**kw)

def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]
