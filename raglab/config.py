from dataclasses import asdict, dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
SEED = 42
MODES = ("bm25", "dense", "hybrid", "router", "reranker", "full")
LABELS = {
    "bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid RRF",
    "router": "Hybrid + Laya router", "reranker": "Hybrid + Laya reranker",
    "full": "Hybrid + Laya router + reranker",
}

@dataclass(frozen=True)
class Settings:
    embedding_model: str = "intfloat/multilingual-e5-small"
    laya_model: str = "convaiinnovations/laya-multilingual"
    chunk_tokens: int = 384
    overlap: int = 64
    threads: int = 4
    embedding_batch: int = 32
    rerank_batch: int = 4
    sparse_k: int = 30
    dense_k: int = 30
    candidate_k: int = 10
    context_k: int = 4
    rrf_k: int = 60
    router_threshold: float = 0.6

    def manifest(self):
        return asdict(self)

@dataclass(frozen=True)
class APIProfile:
    name: str
    base_url: str
    model: str
    key_env: str

def profiles():
    return {
        "sub2api": APIProfile("sub2api", os.getenv("SUB2API_BASE_URL", "http://127.0.0.1:8080/v1"), "gpt-5.6-luna", "SUB2API_API_KEY"),
        "routerai": APIProfile("routerai", "https://routerai.ru/api/v1", "deepseek/deepseek-v4.1-flash-1017", "ROUTERAI_API_KEY"),
    }
