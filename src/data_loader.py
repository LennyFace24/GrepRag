"""加载 CLongEval JSONL 数据集"""
import json
from pathlib import Path
from dataclasses import dataclass
from src.config import DATA_DIR, CTX_DIR

@dataclass
class Question:
    id: str; context: str; query: str; answer: str
    def write_ctx(self) -> Path:
        p = CTX_DIR / f"{self.id}.txt"
        p.write_text(self.context, encoding="utf-8")
        return p
    def ctx_path(self) -> Path:
        return CTX_DIR / f"{self.id}.txt"

def load(size: str = "small", limit: int | None = None) -> list[Question]:
    qs = []
    with open(DATA_DIR / f"{size}.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qs.append(Question(obj["id"], obj["context"], obj["query"], obj["answer"]))
            if limit and len(qs) >= limit: break
    return qs

def prepare(qs: list[Question]) -> list[Path]:
    return [q.write_ctx() for q in qs]
