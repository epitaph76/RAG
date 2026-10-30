from dataclasses import replace
from types import SimpleNamespace
import math
import numpy as np
import pytest

from raglab.config import Settings, profiles
from raglab.types import Document, Chunk, Hit, Dataset, Query, SearchResult, Answer
from raglab.storage import write_json, read_json
from raglab.data import rubq_adapter, answer_aliases
from raglab.retrieval import Index, chunk_documents, collapse_documents, rrf
from raglab.evaluation import retrieval_metrics, qa_metrics, fixed_queries
from raglab.generation import Budget, GenerationError, generate, validate_citations

class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()
    def decode(self, ids, **kwargs):
        return " ".join(ids)

class Encoder:
    tokenizer = Tokenizer()
    def get_sentence_embedding_dimension(self):
        return 8
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            v = np.ones(8, dtype="float32")
            for c in text:
                v[ord(c) % 8] += 1
            vectors.append(v / np.linalg.norm(v))
        return np.array(vectors)

def models(laya=None):
    return SimpleNamespace(encoder=Encoder(), revisions={}, laya=laya)

def hits():
    return [Hit(Chunk(f"d{i}:0", f"d{i}", text), 1.0 / (i + 1))
            for i, text in enumerate(("alpha apples", "beta oranges", "gamma pears"))]

def test_chunk_overlap_and_last_window():
    result = chunk_documents([Document("a", "one two three four five six seven")], Tokenizer(), 4, 1)
    assert [c.text for c in result] == ["one two three four", "four five six seven"]
    assert [c.id for c in result] == ["a:0", "a:3"]
    with pytest.raises(ValueError):
        chunk_documents([], Tokenizer(), 4, 4)

def test_rrf_shared_document_and_dedup():
    a, b, c = hits()
    ranking = rrf([[a, b], [b, c]])
    assert ranking[0].chunk.id == b.chunk.id
    assert ranking[0].score == pytest.approx(1 / 62 + 1 / 61)
    duplicate = Hit(Chunk("d0:10", "d0", "other chunk"), 0)
    assert len(collapse_documents([a, duplicate, b])) == 2

def test_metrics_are_document_level():
    scores = retrieval_metrics(["a", "a", "b"], {"b": 1, "c": 1})
    assert scores["Recall@10"] == 0.5
    assert scores["MRR@10"] == 0.5
    assert 0 < scores["nDCG@10"] < 1
    with pytest.raises(ValueError):
        retrieval_metrics([], {})

def test_qa_alias_and_russian_normalization():
    assert qa_metrics("Орёл [1]", ["Орел", "город Орел"])["EM"] == 1
    assert qa_metrics("Анна Каренина", ["Анна Каренина"])["F1"] == 1

def test_rubq_adapter_uses_only_paragraph_qrels(tmp_path):
    write_json(tmp_path / "RuBQ_2.0_paragraphs.json", [{"uid": 10, "ru_wiki_pageid": 7, "text": "Only the passage"}])
    entries = [{"uid": 1, "question_text": "A question", "answer_text": "Secret gold answer",
                "paragraphs_uids": {"with_answer": [10], "value": [10]}, "answers": []},
               {"uid": 2, "question_text": "No labels", "paragraphs_uids": {"with_answer": []}}]
    for split in ("dev", "test"):
        write_json(tmp_path / f"RuBQ_2.0_{split}.json", entries)
    docs, queries, qrels, counts = rubq_adapter(tmp_path)
    assert docs[0].text == "Only the passage"
    assert qrels["test"] == {"1": {"10": 1}}
    assert counts["test"]["excluded_no_paragraph_qrels"] == 1
    entries[0]["paragraphs_uids"]["with_answer"] = [999]
    write_json(tmp_path / "RuBQ_2.0_test.json", entries)
    with pytest.raises(ValueError, match="missing document"):
        rubq_adapter(tmp_path)

def test_real_bm25_faiss_persistence_and_cache_invalidation(tmp_path):
    ds = Dataset("fixture", [Document("a", "alpha apples"), Document("b", "beta oranges")], {}, {}, {"fixture": 1})
    settings = Settings(chunk_tokens=4, overlap=1)
    first = Index(ds, models(), settings, tmp_path).build()
    result = first.retrieve("alpha", "bm25")
    assert result.hits[0].chunk.document_id == "a"
    assert first.retrieve("oranges", "bm25").hits[0].chunk.document_id == "b"
    second = Index(ds, models(), settings, tmp_path).build()
    assert second.retrieve("alpha", "dense").hits == first.retrieve("alpha", "dense").hits
    assert np.allclose(np.linalg.norm(np.load(first.root / "embeddings.npy"), axis=1), 1)
    with pytest.raises(ValueError, match="Несовместимый"):
        Index(ds, models(), replace(settings, overlap=0), tmp_path).build()
    tampered = read_json(first.root / "chunks.json")
    tampered[0]["text"] = "corrupted content"
    write_json(first.root / "chunks.json", tampered)
    with pytest.raises(ValueError, match="Повреждён"):
        Index(ds, models(), settings, tmp_path).build()
    repaired = Index(ds, models(), settings, tmp_path).rebuild()
    assert repaired.chunks[0].text == "alpha apples"
    assert repaired.retrieve("alpha", "bm25").hits[0].chunk.document_id == "a"

@pytest.mark.parametrize("choice,confidence,expected", [("bm25", 0.9, "bm25"), ("dense", 0.4, "hybrid"), ("bad", 0.99, "hybrid"), ("dense", float("nan"), "hybrid")])
def test_router_confidence_fallback(choice, confidence, expected):
    laya = SimpleNamespace(predict=lambda *a, **k: {"answers": {"retriever": {"choice": choice, "confidence": confidence}}})
    idx = Index(SimpleNamespace(name="fixture"), models(laya))
    assert idx.route("query")[0] == expected

def test_reranker_batch_order_and_ties():
    calls = []
    def predict(requests, **kwargs):
        calls.append((requests, kwargs))
        return [{"answers": {"relevance": {"score": s}}} for s in [1, 3, 3]]
    idx = Index(SimpleNamespace(name="fixture"), models(SimpleNamespace(predict_batch=predict)))
    result = idx.rerank("query", hits())
    assert [h.chunk.document_id for h in result] == ["d1", "d2", "d0"]
    assert calls[0][1]["batch_size"] == 4
    assert all(r["model"] == "multilingual" for r in calls[0][0])

def test_router_failure_is_visible_and_preserves_hybrid():
    idx = Index(SimpleNamespace(name="fixture"), models())
    idx.bm25 = idx.faiss = True
    idx.bm25_search = lambda q: hits()
    idx.dense_search = lambda q: hits()
    idx.route = lambda q: (_ for _ in ()).throw(RuntimeError("unavailable"))
    result = idx.retrieve("q", "router")
    assert result.route == "hybrid" and result.warnings
    assert result.hits

def test_citation_range_and_missing():
    assert validate_citations("Факт [1], ошибка [0] и [99].", 2) == ([1], [0, 99], False)
    assert validate_citations("Факт без ссылки", 2)[2]
    assert not validate_citations("Недостаточно данных в источниках.", 0)[2]

def test_budget_counts_failed_attempts_without_retries(tmp_path):
    class Broken:
        calls = 0
        def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("sk-a-secret-that-must-not-appear")
    broken = Broken()
    client = SimpleNamespace(chat=SimpleNamespace(completions=broken))
    budget = Budget(tmp_path / "budget.json")
    with pytest.raises(GenerationError) as error:
        generate("q", hits(), profiles()["sub2api"], budget=budget, client=client)
    assert "sk-" not in str(error.value)
    assert budget.status()["used"] == 2 and broken.calls == 1
    state = budget.status()
    state["limit"] = 2
    write_json(budget.path, state)
    with pytest.raises(GenerationError):
        generate("q", hits(), profiles()["sub2api"], budget=budget, client=client)
    assert broken.calls == 2
    assert budget.status()["used"] == 3
    assert "limit" not in budget.status()

def test_generator_sources_and_usage(tmp_path):
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model="fixed-model", choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ [1]."))],
                               usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 30, "completion_tokens": 5}))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = generate("q", hits()[:1], profiles()["sub2api"], budget=Budget(tmp_path / "budget.json"), client=client)
    assert answer.citations == [1] and not answer.invalid_citations
    assert answer.usage["prompt_tokens"] == 30
    assert len(captured["messages"]) == 2
    assert captured["model"] == "gpt-5.6-luna"

def test_embedding_resume_preserves_row_mapping(tmp_path):
    class Interrupted(Encoder):
        calls = 0
        def encode(self, texts, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("interrupted")
            return super().encode(texts, **kwargs)
    documents = [Document(str(i), "alpha " * (i + 1)) for i in range(6)]
    ds = Dataset("resume", documents, {}, {}, {"fixture": 2})
    settings = Settings(embedding_batch=2, chunk_tokens=50, overlap=1)
    broken = SimpleNamespace(encoder=Interrupted(), revisions={})
    idx = Index(ds, broken, settings, tmp_path / "resumed")
    with pytest.raises(RuntimeError, match="interrupted"):
        idx.build()
    assert read_json(idx.root / "pending.json")["completed"] == 2
    assert not (idx.root / "manifest.json").exists()
    resumed = Index(ds, models(), settings, tmp_path / "resumed").build()
    fresh = Index(ds, models(), settings, tmp_path / "fresh").build()
    assert np.allclose(np.load(resumed.root / "embeddings.npy"), np.load(fresh.root / "embeddings.npy"))

def test_budget_is_shared_and_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / "budget.json"
    def reserve(_):
        try:
            return Budget(path).reserve("test", "test", "test")
        except GenerationError:
            return None
    with ThreadPoolExecutor(max_workers=5) as pool:
        attempts = list(pool.map(reserve, range(25)))
    assert sorted(x for x in attempts if x is not None) == list(range(2, 27))
    assert Budget(path).status()["used"] == 26
    assert Budget(path).status()["unlimited"] is True

def test_legacy_exhausted_budget_is_migrated_without_losing_history(tmp_path):
    path = tmp_path / "budget.json"
    write_json(path, {"used": 20, "limit": 20, "events": [{"attempt": 20, "purpose": "old"}]})
    ledger = Budget(path)
    ledger.migrate()
    assert ledger.reserve("sub2api", "test", "new") == 21
    state = read_json(path)
    assert "limit" not in state and state["unlimited"] is True
    assert state["events"][0]["purpose"] == "old"
    assert len(state["events"]) == 2

def test_completed_model_snapshot_is_available_without_network(monkeypatch, tmp_path):
    import raglab.models as module
    import huggingface_hub
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path / "artifacts")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"fixture")
    monkeypatch.setattr(huggingface_hub, "model_info", lambda repo: SimpleNamespace(sha="pinned-revision"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snapshot))
    assert module.Models().snapshot("fixture/repo") == str(snapshot)
    def offline(*a, **k):
        raise RuntimeError("network unavailable")
    monkeypatch.setattr(huggingface_hub, "model_info", offline)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    assert module.Models().snapshot("fixture/repo") == str(snapshot)

def test_no_evidence_checks_do_not_spend_again(monkeypatch, tmp_path):
    import raglab.evaluation as evaluation
    monkeypatch.setattr(evaluation, "ARTIFACTS", tmp_path)
    calls = []
    def fake_generate(question, sources, profile, api_key=None, **kwargs):
        calls.append(question)
        return Answer("Недостаточно данных в источниках.", [], [], False, {}, 1, profile.model, profile.name, sources)
    monkeypatch.setattr(evaluation, "generate", fake_generate)
    first = evaluation.no_evidence_checks(profiles()["sub2api"])
    second = evaluation.no_evidence_checks(profiles()["sub2api"])
    assert all(row["refusal"] for row in first)
    assert second == first and len(calls) == 2

def test_qa_resume_does_not_repeat_successful_paid_cases(monkeypatch, tmp_path):
    import raglab.evaluation as evaluation
    monkeypatch.setattr(evaluation, "ARTIFACTS", tmp_path)
    queries = [Query(str(i), f"question {i}", ["gold"]) for i in range(4)]
    dataset = Dataset("rubq", [], {"test": queries}, {"test": {q.id: {"d0": 1} for q in queries}}, {"fixture": 3})
    idx = SimpleNamespace(dataset=dataset, settings=Settings(), models=models(), manifest={"identity": {"fixture": 3}},
        retrieve=lambda *a, **k: SearchResult(hits(), "hybrid", None, {"total": 1}))
    calls = []
    def fake_generate(question, sources, profile, api_key=None, **kwargs):
        calls.append(question)
        if len(calls) == 3:
            raise GenerationError("planned failure")
        text = "gold [1]" if question.startswith("question") else "Недостаточно данных в источниках."
        return Answer(text, [1] if sources else [], [], False, {"prompt_tokens": 20, "completion_tokens": 5}, 1, profile.model, profile.name, sources)
    monkeypatch.setattr(evaluation, "generate", fake_generate)
    with pytest.raises(GenerationError, match="planned failure"):
        evaluation.evaluate_qa(idx, profiles()["sub2api"])
    assert len(read_json(tmp_path / "qa-checkpoint.json")["completed"]) == 2
    path, report = evaluation.evaluate_qa(idx, profiles()["sub2api"])
    assert path.exists() and report["no_evidence_passed"]
    assert len(calls) == 19  # 18 successful cases plus the one failed attempt.
    assert len(report["rows"]) == 18

def test_atomic_json_retries_windows_reader_lock(monkeypatch, tmp_path):
    import raglab.storage as storage
    destination = tmp_path / "progress.json"
    write_json(destination, {"completed": 10})
    original_replace = storage.os.replace
    attempts = []
    def transient_lock(source, target):
        attempts.append(1)
        if len(attempts) < 3:
            assert read_json(destination) == {"completed": 10}
            error = PermissionError("reader holds destination")
            error.winerror = 5
            raise error
        return original_replace(source, target)
    monkeypatch.setattr(storage.os, "replace", transient_lock)
    monkeypatch.setattr(storage.time, "sleep", lambda delay: None)
    write_json(destination, {"completed": 20})
    assert len(attempts) == 3
    assert read_json(destination) == {"completed": 20}
    assert not list(tmp_path.glob("*.tmp"))

def test_atomic_json_persistent_error_keeps_old_checkpoint(monkeypatch, tmp_path):
    import raglab.storage as storage
    destination = tmp_path / "progress.json"
    write_json(destination, {"completed": 10})
    def locked(source, target):
        error = PermissionError("persistent lock")
        error.winerror = 32
        raise error
    monkeypatch.setattr(storage.os, "replace", locked)
    monkeypatch.setattr(storage.time, "sleep", lambda delay: None)
    with pytest.raises(PermissionError):
        write_json(destination, {"completed": 20})
    assert read_json(destination) == {"completed": 10}
    assert not list(tmp_path.glob("*.tmp"))
