from dataclasses import dataclass, field

@dataclass
class Document:
    id: str
    text: str
    title: str = ""
    url: str = ""

@dataclass
class Query:
    id: str
    text: str
    answers: list[str] = field(default_factory=list)

Qrels = dict[str, dict[str, int]]

@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    title: str = ""
    url: str = ""

@dataclass
class Hit:
    chunk: Chunk
    score: float

@dataclass
class SearchResult:
    hits: list[Hit]
    route: str
    confidence: float | None
    timings: dict[str, float]
    warnings: list[str] = field(default_factory=list)

@dataclass
class Answer:
    text: str
    citations: list[int]
    invalid_citations: list[int]
    missing_citations: bool
    usage: dict
    latency_ms: float
    model: str
    profile: str
    sources: list[Hit] = field(default_factory=list)

@dataclass
class Dataset:
    name: str
    documents: list[Document]
    queries: dict[str, list[Query]]
    qrels: dict[str, Qrels]
    manifest: dict
