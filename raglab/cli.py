"""Commands for the personal PDD knowledge base."""
import argparse
from dataclasses import asdict
import json
import os
import sys
from .config import Settings, profiles, DATA
from .data import prepare
from .models import Models
from .retrieval import Index
from .traffic import TrafficKnowledge
from .generation import UsageLedger, generate

def main():
    parser = argparse.ArgumentParser(description="Личный RAG по ПДД России")
    parser.add_argument("command", choices=["prepare", "index", "ask", "verify", "usage", "budget", "models"])
    parser.add_argument("--dataset", choices=["pdd"], default="pdd", help="Единственная рабочая база — ПДД России")
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--question")
    parser.add_argument("--profile", choices=["sub2api", "routerai"], default="sub2api")
    parser.add_argument("--key-stdin", action="store_true", help="Read the key from a private input pipe")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit answer, source and image metadata as JSON")
    parser.add_argument("--add", type=int, help="Deprecated compatibility option; requests are unlimited")
    args = parser.parse_args()
    if args.key_stdin:
        os.environ[profiles()[args.profile].key_env] = sys.stdin.readline().strip()
    if args.command in ("usage", "budget"):
        print(json.dumps(UsageLedger().migrate(), indent=2))
        return
    if args.command == "verify":
        from .pdd import verify
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return
    settings = Settings()
    models = Models(settings)
    if args.command == "models":
        _ = models.encoder
        _ = models.laya
        print(json.dumps(models.revisions, indent=2))
        return
    def progress(fraction, message):
        if not args.json:
            print(f"{fraction:.0%} {message}", flush=True)
    dataset = prepare("pdd", progress)
    if args.command == "prepare":
        print(json.dumps(dataset.manifest, ensure_ascii=False, indent=2))
        return
    if args.command == "ask" and not args.question:
        parser.error("--question is required")
    index = Index(dataset, models, settings)
    (index.rebuild if args.rebuild else index.build)(progress)
    if args.command == "index":
        return
    knowledge = TrafficKnowledge(dataset.documents, DATA / "pdd")
    result = index.retrieve(args.question, args.mode)
    sources = knowledge.sources(args.question, result.hits, settings.context_k)
    answer = generate(args.question, sources, profiles()[args.profile], purpose="pdd-chat", images=knowledge.images(args.question, sources))
    images = knowledge.images(args.question, sources, answer.text)
    if args.json:
        print(json.dumps({"answer": asdict(answer), "images": images, "retrieval": asdict(result)}, ensure_ascii=False))
    else:
        print(answer.text)
        for image in images:
            print(f"Изображение {image['code']}: {DATA / 'pdd' / image['local_path']}")
        print(json.dumps(answer.usage, ensure_ascii=False))

if __name__ == "__main__":
    main()
