"""Official dataset adapters; annotations are never indexed as document text."""
import json
import urllib.request
import urllib.parse
import subprocess
from pathlib import Path

from .config import DATA, SEED
from .storage import file_hash, read_json, write_json
from .types import Dataset, Document, Query

def download(url, destination: Path, progress=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "hybrid-laya-lab/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        size = 0
        while block := response.read(1024 * 1024):
            out.write(block)
            size += len(block)
            if progress:
                progress(min(size / total, 1) if total else 0, f"Загрузка {destination.name}: {size // 1024**2} МБ")
    temporary.replace(destination)

def answer_aliases(entry):
    values = []
    for answer in entry.get("answers", []):
        if answer.get("type") == "literal":
            values.append(str(answer.get("value", "")))
        values.extend([answer.get("label", ""), *answer.get("wp_names", [])])
        values.extend(answer.get("wd_names", {}).get("ru", []))
    if entry.get("answer_text"):
        values.append(entry["answer_text"])
    return list(dict.fromkeys(str(v) for v in values if v))

def rubq_adapter(raw: Path):
    paragraphs = read_json(raw / "RuBQ_2.0_paragraphs.json")
    documents = [Document(str(p["uid"]), p["text"], f"Wikipedia · {p['ru_wiki_pageid']}",
                          f"https://ru.wikipedia.org/?curid={p['ru_wiki_pageid']}") for p in paragraphs]
    valid = {d.id for d in documents}
    queries, qrels, counts = {}, {}, {}
    for split in ("dev", "test"):
        entries = read_json(raw / f"RuBQ_2.0_{split}.json")
        queries[split], qrels[split] = [], {}
        for entry in entries:
            qid = str(entry["uid"])
            relevant = {str(x): 1 for x in entry.get("paragraphs_uids", {}).get("with_answer", [])}
            missing = relevant.keys() - valid
            if missing:
                raise ValueError(f"RuBQ: missing document IDs: {sorted(missing)[:5]}")
            queries[split].append(Query(qid, entry["question_text"], answer_aliases(entry)))
            if relevant:
                qrels[split][qid] = relevant
        counts[split] = {"queries": len(entries), "eligible": len(qrels[split]),
                         "excluded_no_paragraph_qrels": len(entries) - len(qrels[split])}
    return documents, queries, qrels, counts

def scifact_hf_adapter(raw: Path):
    import pyarrow.parquet as pq
    entries = pq.read_table(raw / "corpus.parquet").to_pylist()
    docs = [Document(str(e["_id"]), e["text"], e.get("title", ""),
                     "https://www.semanticscholar.org/search?q=" + urllib.parse.quote(e.get("title", ""))) for e in entries]
    queries = {str(e["_id"]): Query(str(e["_id"]), e["text"]) for e in pq.read_table(raw / "queries.parquet").to_pylist()}
    splits, qrels, counts = {}, {}, {}
    for split in ("train", "test"):
        rel = {}
        for line in (raw / f"{split}.tsv").read_text(encoding="utf-8").splitlines()[1:]:
            qid, docid, score = line.split("\t")
            if int(score) > 0:
                rel.setdefault(qid, {})[docid] = int(score)
        name = "dev" if split == "train" else split
        splits[name] = [queries[qid] for qid in sorted(rel)]
        qrels[name] = rel
        counts[name] = {"queries": len(rel), "eligible": len(rel), "excluded_no_paragraph_qrels": 0}
    return docs, splits, qrels, counts

def prepare(name="pdd", progress=None, root=DATA):
    if name != "pdd":
        raise ValueError("В личном RAG доступна только база ПДД России")
    directory = root / name
    if not (directory / "manifest.json").exists():
        from .pdd import build
        build(directory, progress=progress)
    return load_dataset(name, root)

def load_dataset(name, root=DATA):
    directory = root / name
    manifest = read_json(directory / "manifest.json")
    if manifest["schema"] != 1 or file_hash(directory / "dataset.json") != manifest["normalized_sha256"]:
        raise ValueError("Данные изменились или повреждены. Подготовьте датасет заново в новой папке.")
    data = read_json(directory / "dataset.json")
    return Dataset(name, [Document(**d) for d in data["documents"]],
                   {k: [Query(**q) for q in v] for k, v in data["queries"].items()}, data["qrels"], manifest)
