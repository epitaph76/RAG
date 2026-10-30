import time
import threading
import math
from pathlib import Path
import numpy as np

from .config import ARTIFACTS, Settings, SEED, MODES
from .models import Models
from .storage import read_json, write_json, fingerprint, file_hash, package_versions
from .types import Chunk, Hit, SearchResult

def chunk_documents(documents, tokenizer, size=384, overlap=64, progress=None):
    if not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be smaller than chunk size")
    chunks = []
    for number, doc in enumerate(documents, 1):
        ids = tokenizer.encode(doc.text, add_special_tokens=False, truncation=False, verbose=False)
        for start in range(0, len(ids), size - overlap):
            text = tokenizer.decode(ids[start:start + size], skip_special_tokens=True)
            chunks.append(Chunk(f"{doc.id}:{start}", doc.id, text, doc.title, doc.url))
            if start + size >= len(ids):
                break
        if progress and number % 1000 == 0:
            progress(0, f"Разбиение документов: {number}/{len(documents)}")
    return chunks

def collapse_documents(hits):
    seen, result = set(), []
    for hit in hits:
        if hit.chunk.document_id not in seen:
            seen.add(hit.chunk.document_id)
            result.append(hit)
    return result

def rrf(rankings, k=60):
    scores, chunks = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            key = hit.chunk.id
            if key in seen:
                continue
            seen.add(key)
            chunks[key] = hit.chunk
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
    return [Hit(chunks[key], scores[key]) for key in sorted(scores, key=lambda key: (-scores[key], key))]

class Index:
    def __init__(self, dataset, models=None, settings=Settings(), root=ARTIFACTS):
        self.dataset, self.settings = dataset, settings
        self.models = models or Models(settings)
        self.root = Path(root) / "indices" / dataset.name
        self.chunks = []
        self.bm25 = self.faiss = None
        self.manifest = None
        self.lock = threading.RLock()

    def identity(self):
        # Resolve the pinned encoder before checking cache compatibility.
        _ = self.models.encoder
        return {"schema": 1, "dataset": fingerprint(self.dataset.manifest),
                "settings": self.settings.manifest(), "packages": package_versions(), "seed": SEED,
                "embedding_revision": self.models.revisions.get(self.settings.embedding_model, "test")}

    def build(self, progress=None):
        from filelock import FileLock
        self.root.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.root) + ".lock"):
            return self._build(progress)

    def _build(self, progress=None):
        import bm25s
        import faiss
        faiss.omp_set_num_threads(self.settings.threads)
        identity = self.identity()
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["identity"] != identity:
                raise ValueError("Несовместимый индекс: используйте --rebuild или кнопку «Перестроить индекс».")
            for filename, expected in manifest["files"].items():
                if file_hash(self.root / filename) != expected:
                    raise ValueError(f"Повреждён индекс {filename}. Перестройте его.")
            self.chunks = [Chunk(**c) for c in read_json(self.root / "chunks.json")]
            self.bm25 = bm25s.BM25.load(str(self.root / "bm25"), load_corpus=False)
            self.faiss = faiss.read_index(str(self.root / "dense.faiss"))
            self.manifest = manifest
            if progress:
                progress(1, "Индекс загружен с диска")
            return self
        if progress:
            progress(0, "Разбиение документов на чанки")
        chunks_path = self.root / "chunks.json"
        pending = self.root / "pending.json"
        if pending.exists() and read_json(pending).get("identity") == identity and chunks_path.exists():
            self.chunks = [Chunk(**c) for c in read_json(chunks_path)]
        else:
            self.chunks = chunk_documents(self.dataset.documents, self.models.encoder.tokenizer,
                                          self.settings.chunk_tokens, self.settings.overlap, progress)
            write_json(chunks_path, self.chunks)
            write_json(pending, {"identity": identity, "completed": 0})
        if not self.chunks:
            raise ValueError("В датасете нет текста")
        state = read_json(pending)
        encoder = self.models.encoder
        dimension = (encoder.get_embedding_dimension() if hasattr(encoder, "get_embedding_dimension")
                     else encoder.get_sentence_embedding_dimension())
        vector_path = self.root / "embeddings.npy"
        resume = vector_path.exists() and state["completed"] > 0
        order_path = self.root / "embedding-order.json"
        if not resume or not order_path.exists():
            # Keep stored row IDs stable; only execution order changes. Grouping similar
            # lengths avoids padding every short paragraph to a 384-token neighbour.
            done = state["completed"] if resume else 0
            order = list(range(done)) + sorted(range(done, len(self.chunks)), key=lambda i: (len(self.chunks[i].text), i))
            write_json(order_path, order)
        else:
            order = read_json(order_path)
        vectors = np.lib.format.open_memmap(vector_path, mode="r+" if resume else "w+", dtype="float32",
                                            shape=(len(self.chunks), dimension))
        batch = self.settings.embedding_batch
        for start in range(state["completed"], len(self.chunks), batch):
            stop = min(start + batch, len(self.chunks))
            positions = order[start:stop]
            texts = ["passage: " + self.chunks[i].text for i in positions]
            vectors[positions] = self.models.encoder.encode(texts, batch_size=batch,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            vectors.flush()
            write_json(pending, {"identity": identity, "completed": stop})
            if progress:
                progress(0.85 * stop / len(self.chunks), f"Embeddings: {stop}/{len(self.chunks)}")
        if progress:
            progress(0.88, "Построение BM25")
        tokens = bm25s.tokenize([c.text for c in self.chunks], stopwords=None, show_progress=False)
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens, show_progress=False)
        self.bm25.save(str(self.root / "bm25"))
        self.faiss = faiss.IndexFlatIP(dimension)
        self.faiss.add(np.asarray(vectors))
        faiss.write_index(self.faiss, str(self.root / "dense.faiss"))
        files = [p for p in self.root.rglob("*") if p.is_file() and p.name not in ("manifest.json", "pending.json")]
        self.manifest = {"identity": identity, "chunks": len(self.chunks), "documents": len(self.dataset.documents),
                         "files": {p.relative_to(self.root).as_posix(): file_hash(p) for p in files}}
        write_json(manifest_path, self.manifest)
        if progress:
            progress(1, "Индекс готов")
        return self

    def rebuild(self, progress=None):
        from filelock import FileLock
        self.root.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.root) + ".lock"):
            # Explicit rebuild also repairs corrupted chunks/vectors; an ordinary build
            # is the operation that resumes an interrupted compatible embedding job.
            for name in ("manifest.json", "pending.json"):
                (self.root / name).unlink(missing_ok=True)
            self.bm25 = self.faiss = self.manifest = None
            return self._build(progress)

    def bm25_search(self, query):
        import bm25s
        tokens = bm25s.tokenize([query], stopwords=None, show_progress=False)
        indices, scores = self.bm25.retrieve(tokens, k=min(self.settings.sparse_k, len(self.chunks)), show_progress=False)
        return [Hit(self.chunks[int(i)], float(s)) for i, s in zip(indices[0], scores[0]) if s > 0]

    def dense_search(self, query):
        vector = self.models.encoder.encode(["query: " + query], normalize_embeddings=True,
                                             convert_to_numpy=True, show_progress_bar=False)
        scores, indices = self.faiss.search(np.asarray(vector, dtype="float32"), min(self.settings.dense_k, len(self.chunks)))
        return [Hit(self.chunks[int(i)], float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    def route(self, query):
        questions = {"retriever": {"type": "choice", "instructions": "Choose the best retrieval method for this search query.",
            "criteria": {"bm25": "Exact names, identifiers, technical terms or literal keyword lookup",
                         "dense": "Paraphrases, conceptual meaning or semantic similarity",
                         "hybrid": "Mixed, ambiguous or uncertain query requiring both lexical and semantic search"}}}
        result = self.models.laya.predict(query, questions, model="multilingual", max_len=1024)
        answer = result["answers"]["retriever"]
        confidence = float(answer.get("confidence", 0))
        choice = answer.get("choice")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return "hybrid", None, ["Некорректная уверенность Laya: использован Hybrid."]
        if choice not in ("bm25", "dense", "hybrid") or confidence < self.settings.router_threshold:
            return "hybrid", confidence, ["Неуверенный маршрут Laya: использован Hybrid."]
        return choice, confidence, []

    def rerank(self, query, hits):
        questions = {"relevance": {"type": "score", "instructions": "How well does the passage provide evidence to answer the query?",
            "criteria": ["irrelevant", "related topic but no answer", "partially answers the query", "directly answers the query"]}}
        requests = [{"state": f"QUERY: {query}\nPASSAGE: {hit.chunk.text}", "questions": questions,
                     "model": "multilingual", "max_len": 1024} for hit in hits]
        if not requests:
            return []
        results = self.models.laya.predict_batch(requests, batch_size=self.settings.rerank_batch, sort_by_length=True)
        if len(results) != len(hits):
            raise ValueError("Invalid reranker batch size")
        scores = [float(r["answers"]["relevance"]["score"]) for r in results]
        if not all(math.isfinite(score) and 0 <= score <= 3 for score in scores):
            raise ValueError("Invalid reranker scores")
        return [Hit(hits[i].chunk, scores[i]) for i in sorted(range(len(hits)), key=lambda i: -scores[i])]

    def retrieve(self, query, mode="hybrid", limit=10):
        if mode not in MODES or not query.strip():
            raise ValueError("Выберите режим и введите непустой вопрос")
        if self.bm25 is None or self.faiss is None:
            raise ValueError("Сначала подготовьте индекс")
        with self.lock:
            started = time.perf_counter()
            timings, warnings, confidence = {}, [], None
            route = mode if mode in ("bm25", "dense") else "hybrid"
            if mode in ("router", "full"):
                before = time.perf_counter()
                try:
                    route, confidence, warnings = self.route(query)
                except Exception:
                    route = "hybrid"
                    warnings.append("Laya router недоступен: использован Hybrid.")
                timings["router"] = (time.perf_counter() - before) * 1000
            rankings = []
            for kind in ("bm25", "dense"):
                if route in (kind, "hybrid"):
                    before = time.perf_counter()
                    rankings.append(getattr(self, kind + "_search")(query))
                    timings[kind] = (time.perf_counter() - before) * 1000
            before = time.perf_counter()
            hits = collapse_documents(rrf(rankings, self.settings.rrf_k) if route == "hybrid" else rankings[0])
            timings["fusion"] = (time.perf_counter() - before) * 1000
            if mode in ("reranker", "full"):
                before = time.perf_counter()
                try:
                    top = self.settings.candidate_k
                    hits = self.rerank(query, hits[:top]) + hits[top:]
                except Exception:
                    warnings.append("Laya reranker недоступен: сохранён исходный порядок.")
                timings["reranker"] = (time.perf_counter() - before) * 1000
            timings["total"] = (time.perf_counter() - started) * 1000
            return SearchResult(hits[:limit], route, confidence, timings, warnings)
