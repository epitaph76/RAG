import csv
import math
import random
import re
import threading
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import numpy as np

from .config import ARTIFACTS, SEED, LABELS
from .storage import write_json, read_json, fingerprint
from .generation import generate, REFUSAL
from .types import Chunk, Hit

class MemorySampler:
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.05):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()

def retrieval_metrics(document_ids, qrels):
    ranked = list(dict.fromkeys(document_ids))
    if not qrels:
        raise ValueError("A retrieval metric requires nonempty positive qrels")
    result = {}
    for k in (5, 10, 20):
        found = sum(docid in qrels for docid in ranked[:k])
        result[f"Recall@{k}"] = found / len(qrels)
    result["MRR@10"] = next((1 / rank for rank, docid in enumerate(ranked[:10], 1) if docid in qrels), 0)
    dcg = sum((2 ** qrels.get(docid, 0) - 1) / math.log2(rank + 1) for rank, docid in enumerate(ranked[:10], 1))
    ideal = sum((2 ** score - 1) / math.log2(rank + 1) for rank, score in enumerate(sorted(qrels.values(), reverse=True)[:10], 1))
    result["nDCG@10"] = dcg / ideal
    return result

def normalize_answer(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold().replace("ё", "е")).split())

def qa_metrics(answer, aliases):
    answer = normalize_answer(re.sub(r"\[\d+\]", "", answer))
    def score(alias):
        gold = normalize_answer(alias)
        common = sum((Counter(answer.split()) & Counter(gold.split())).values())
        f1 = 2 * common / (len(answer.split()) + len(gold.split())) if common else float(answer == gold)
        return float(answer == gold), f1
    scores = [score(a) for a in aliases if a]
    return {"EM": max((s[0] for s in scores), default=0), "F1": max((s[1] for s in scores), default=0)}

def fixed_queries(dataset, split="test", count=None):
    eligible = sorted((q for q in dataset.queries[split] if q.id in dataset.qrels[split]), key=lambda q: q.id)
    if count is None or count >= len(eligible):
        return eligible
    return random.Random(SEED).sample(eligible, count)

def save_report(report, rows, name):
    directory = ARTIFACTS / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{name}-{stamp}.json"
    report["rows"] = rows
    write_json(path, report)
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row if not isinstance(row[k], (dict, list))))
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return path

def no_evidence_checks(profile, api_key=None):
    """Two paid checks, cached independently of an index so they can run early."""
    checkpoint = ARTIFACTS / "no-evidence-checkpoint.json"
    identity = fingerprint({"profile": asdict(profile), "schema": 1})
    state = read_json(checkpoint) if checkpoint.exists() else {}
    completed = state.get("completed", {}) if state.get("identity") == identity else {}
    rows = []
    for number, evidence in enumerate(([], [Hit(Chunk("fixture:0", "fixture", "В офисе стоит три стула.", "Контрольный источник"), 1.0)])):
        case = f"no-evidence:{number}"
        if case not in completed:
            answer = generate("Какой пароль у аккаунта исследователя в системе Zeta-9381?", evidence, profile, api_key, purpose="no-evidence-test")
            completed[case] = {"query_id": case, "mode": "no-evidence", "answer": asdict(answer),
                               "refusal": REFUSAL.casefold() in answer.text.casefold()}
            write_json(checkpoint, {"identity": identity, "completed": completed})
        rows.append(completed[case])
    return rows

def evaluate(index, modes, count=None, progress=None):
    dataset = index.dataset
    queries = fixed_queries(dataset, count=count)
    if not queries:
        raise ValueError("No eligible test queries")
    rows, summary = [], []
    total = len(modes) * len(queries)
    for mode in modes:
        # Unmeasured warm-up: model download/load is reported separately from warm latency.
        before = time.perf_counter()
        warm = index.retrieve(queries[0].text, mode, limit=20)
        warmup_ms = (time.perf_counter() - before) * 1000
        mode_rows = []
        with MemorySampler() as memory:
            for q in queries:
                result = index.retrieve(q.text, mode, limit=20)
                row = {"dataset": dataset.name, "mode": mode, "query_id": q.id,
                       **retrieval_metrics([h.chunk.document_id for h in result.hits], dataset.qrels["test"][q.id]),
                       "latency_ms": result.timings["total"], "route": result.route,
                       "fallback": bool(result.warnings), "warnings": result.warnings,
                       "document_ids": [h.chunk.document_id for h in result.hits], "timings": result.timings}
                mode_rows.append(row)
                rows.append(row)
                if progress:
                    progress(len(rows) / total, f"{LABELS[mode]}: {len(mode_rows)}/{len(queries)}")
        summary.append({"dataset": dataset.name, "mode": mode, "queries": len(queries),
                        **{k: float(np.mean([r[k] for r in mode_rows])) for k in ("Recall@5", "Recall@10", "Recall@20", "MRR@10", "nDCG@10")},
                        "p50_ms": float(np.percentile([r["latency_ms"] for r in mode_rows], 50)),
                        "p95_ms": float(np.percentile([r["latency_ms"] for r in mode_rows], 95)),
                        "peak_rss_gib": memory.peak / 1024**3, "warmup_ms": warmup_ms,
                        "fallback_rate": sum(r["fallback"] for r in mode_rows) / len(mode_rows)})
    report = {"kind": "retrieval", "dataset": dataset.manifest, "index": index.manifest["identity"],
              "model_revisions": index.models.revisions, "seed": SEED, "split": "test", "query_ids": [q.id for q in queries],
              "subset": count is not None, "summary": summary,
              "note": "Warm latency; first call/load timed separately. RSS includes models and index, excludes Docker/OS."}
    path = save_report(report, rows, dataset.name + ("-subset" if count else "-full"))
    return path, report

def evaluate_qa(index, profile, api_key=None, progress=None):
    """16 paired answers plus two controlled no-evidence tests; max_retries=0."""
    if index.dataset.name != "rubq":
        raise ValueError("QA gold answers are available only for RuBQ")
    queries = fixed_queries(index.dataset, count=4)
    rows = []
    # Resume successful/failed cases without spending more attempts on rerun.
    checkpoint = ARTIFACTS / "qa-checkpoint.json"
    identity = fingerprint({"index": index.manifest["identity"], "profile": asdict(profile), "ids": [q.id for q in queries], "schema": 1})
    state = read_json(checkpoint) if checkpoint.exists() else {}
    completed = state.get("completed", {}) if state.get("identity") == identity else {}
    for q in queries:
        for mode in ("hybrid", "router", "reranker", "full"):
            case = q.id + ":" + mode
            if case not in completed:
                with MemorySampler() as memory:
                    result = index.retrieve(q.text, mode)
                    answer = generate(q.text, result.hits[:index.settings.context_k], profile, api_key, purpose="qa-evaluation")
                completed[case] = {"dataset": "rubq", "query_id": q.id, "question": q.text, "mode": mode,
                    **qa_metrics(answer.text, q.answers), "answer": asdict(answer), "latency_ms": result.timings["total"] + answer.latency_ms,
                    "prompt_tokens": answer.usage.get("prompt_tokens"), "completion_tokens": answer.usage.get("completion_tokens"),
                    "citation_valid": not answer.invalid_citations and not answer.missing_citations,
                    "search_warnings": result.warnings, "context_characters": sum(len(h.chunk.text) for h in answer.sources),
                    "context_tokens_e5": sum(len(index.models.encoder.tokenizer.encode(h.chunk.text, add_special_tokens=False)) for h in answer.sources),
                    "peak_rss_gib": memory.peak / 1024**3}
                write_json(checkpoint, {"identity": identity, "completed": completed})
            rows.append(completed[case])
            if progress:
                progress(len(rows) / 18, f"QA: {len(rows)}/18")
    rows.extend(no_evidence_checks(profile, api_key))
    if progress:
        progress(1, "QA: 18/18")
    summary = []
    def usage_total(selected, key):
        values = [row[key] for row in selected]
        return sum(values) if all(value is not None for value in values) else None
    for mode in ("hybrid", "router", "reranker", "full"):
        selected = [r for r in rows if r["mode"] == mode]
        summary.append({"mode": mode, "queries": len(selected), **{key: float(np.mean([r[key] for r in selected])) for key in ("EM", "F1", "latency_ms")},
                        "prompt_tokens": usage_total(selected, "prompt_tokens"),
                        "completion_tokens": usage_total(selected, "completion_tokens"),
                        "p50_ms": float(np.percentile([r["latency_ms"] for r in selected], 50)),
                        "p95_ms": float(np.percentile([r["latency_ms"] for r in selected], 95)),
                        "peak_rss_gib": max(r["peak_rss_gib"] for r in selected),
                        "context_tokens_e5": sum(r["context_tokens_e5"] for r in selected)})
    report = {"kind": "qa", "profile": asdict(profile), "dataset": index.dataset.manifest,
              "index": index.manifest["identity"], "model_revisions": index.models.revisions,
              "seed": SEED, "query_ids": [q.id for q in queries],
              "summary": summary, "no_evidence_passed": all(r["refusal"] for r in rows if r["mode"] == "no-evidence"),
              "note": "Four-query smoke sample; not a statistically reliable quality estimate. EM/F1 use complete answers without citations."}
    return save_report(report, rows, "rubq-qa"), report
