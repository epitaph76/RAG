"""Explicit three-request integration check; no quality benchmark is claimed."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from raglab.config import DATA, ARTIFACTS, profiles
from raglab.data import prepare
from raglab.generation import REFUSAL, UsageLedger, generate
from raglab.retrieval import Index
from raglab.storage import write_json
from raglab.traffic import TrafficKnowledge

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=list(profiles()), default="sub2api")
    parser.add_argument("--key-stdin", action="store_true")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "reports/pdd-integration.json")
    args = parser.parse_args()
    profile = profiles()[args.profile]
    if args.key_stdin:
        os.environ[profile.key_env] = sys.stdin.readline().strip()
    dataset = prepare()
    index = Index(dataset).build()
    knowledge = TrafficKnowledge(dataset.documents, DATA / "pdd")
    cases = [
        ("comparison", "Покажи знаки 3.27 и 3.28. Чем они отличаются и на кого не распространяются?", {"3.27", "3.28"}),
        ("photo", "Покажи фото знака «Главная дорога» и объясни его действие.", {"2.1"}),
        ("unknown", "Покажи знак 9.99 и объясни его значение.", set()),
    ]
    report = {"date": datetime.now(ZoneInfo("Europe/Moscow")).date().isoformat(), "profile": profile.name, "model": profile.model,
              "dataset_sha256": dataset.manifest["normalized_sha256"], "mode": "hybrid",
              "scope": "Integration checks, not independent quality evaluation", "used_before": UsageLedger().status()["used"], "cases": []}
    for name, question, expected in cases:
        result = index.retrieve(question)
        sources = knowledge.sources(question, result.hits)
        answer = generate(question, sources, profile, purpose="pdd-chat", images=knowledge.images(question, sources))
        images = knowledge.images(question, sources, answer.text)
        for asset in images:
            knowledge.image_bytes(asset)
        correct_images = {asset["code"] for asset in images} == expected
        photo_check = name != "photo" or any(asset["kind"] == "real_world_photo_crop" for asset in images)
        refusal_check = name != "unknown" or REFUSAL in answer.text
        no_false_image_refusal = not images or not any(text in answer.text.casefold() for text in ("показать не могу", "картинку показать не могу", "изображения нет"))
        passed = correct_images and photo_check and refusal_check and no_false_image_refusal and not answer.invalid_citations and not answer.missing_citations
        report["cases"].append({"case": name, "question": question, "answer": answer.text,
            "sources": [hit.chunk.document_id for hit in sources],
            "images": [{key: asset[key] for key in ("id", "code", "kind", "sha256", "source_number")} for asset in images],
            "citations": answer.citations, "usage": answer.usage, "generation_ms": answer.latency_ms,
            "asset_checks": correct_images, "passed": passed})
        report["used_after"] = UsageLedger().status()["used"]
        write_json(args.output, report)
        print(json.dumps({"case": name, "passed": passed, "images": len(images)}, ensure_ascii=False), flush=True)
    if not all(case["passed"] for case in report["cases"]):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
