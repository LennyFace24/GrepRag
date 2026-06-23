"""
可插拔的 Embedding 接口
默认使用 BAAI/bge-small-zh，预留 Qwen embedding 接口
"""
from abc import ABC, abstractmethod
import numpy as np

from src.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DEVICE, CHUNK_SIZE, CHUNK_OVERLAP


class BaseEmbedding(ABC):
    """Embedding 抽象基类 —— 所有 embedding 实现必须继承此类"""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        将文本列表转为 embedding 矩阵

        Args:
            texts: 文本列表, shape [batch_size]

        Returns:
            numpy array, shape [batch_size, dim]
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding 向量维度"""
        ...


class BGEEmbedding(BaseEmbedding):
    """
    BAAI/bge-small-zh 中文 embedding
    首次运行自动从 HuggingFace 下载 (~100MB)
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = EMBEDDING_DEVICE):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装, 请运行: pip install sentence-transformers"
            )

        self._model_name = model_name
        self._device = device
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,     # 余弦相似度需要归一化
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        return f"BGEEmbedding(model={self._model_name}, device={self._device})"


class QwenEmbedding(BaseEmbedding):
    """
    Qwen embedding 模型接口 (预留骨架)

    使用示例:
        embedder = QwenEmbedding(api_endpoint="https://your-qwen-endpoint", api_key="xxx")
    """

    def __init__(self, api_endpoint: str, api_key: str, model_name: str = "text-embedding-v3"):
        self._api_endpoint = api_endpoint
        self._api_key = api_key
        self._model_name = model_name
        self._dim = 1024   # Qwen embedding 默认维度, 实际以 API 返回为准

    def embed(self, texts: list[str]) -> np.ndarray:
        # TODO: 对接你的 Qwen embedding API
        # 示例结构:
        #   response = requests.post(
        #       self._api_endpoint,
        #       headers={"Authorization": f"Bearer {self._api_key}"},
        #       json={"model": self._model_name, "input": texts},
        #   )
        #   embeddings = [item["embedding"] for item in response.json()["data"]]
        raise NotImplementedError(
            "Qwen embedding 接口尚未实现。请在 embed() 方法中填入你的 API 调用逻辑。"
        )

    @property
    def dim(self) -> int:
        return self._dim


def create_embedding(provider: str = "bge", **kwargs) -> BaseEmbedding:
    """
    工厂函数: 按名称创建 embedding 实例

    Args:
        provider: "bge" | "qwen"
        **kwargs: 传给具体实现的参数
    """
    if provider == "bge":
        return BGEEmbedding(**kwargs)
    elif provider == "qwen":
        return QwenEmbedding(**kwargs)
    else:
        raise ValueError(f"不支持的 embedding provider: {provider}。可选: bge, qwen")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    将长文本切成带重叠的固定大小块

    Args:
        text: 输入文本
        chunk_size: 每块最大字符数
        overlap: 相邻块之间的重叠字符数

    Returns:
        文本块列表
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks
