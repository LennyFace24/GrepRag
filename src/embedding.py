"""可插拔 Embedding：默认 BGE, 预留 Qwen 接口"""
from abc import ABC, abstractmethod
import numpy as np
from src.config import EMBED_MODEL, EMBED_DEVICE, CHUNK_SIZE, CHUNK_OVERLAP

class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...
    @property
    @abstractmethod
    def dim(self) -> int: ...

class BGEEmbedding(BaseEmbedding):
    def __init__(self, model: str = EMBED_MODEL, device: str = EMBED_DEVICE):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._m.get_sentence_embedding_dimension()

class QwenEmbedding(BaseEmbedding):
    """预留: 对接 Qwen embedding API"""
    def __init__(self, endpoint: str, key: str):
        self._ep, self._key, self._dim = endpoint, key, 1024
    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError("请在 embed() 中填入 Qwen API 调用逻辑")
    @property
    def dim(self) -> int: return self._dim

def create(provider: str = "bge", **kw) -> BaseEmbedding:
    return {"bge": BGEEmbedding, "qwen": QwenEmbedding}[provider](**kw)

def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]
