import argparse
import json
import os
import sys
from .config import Settings, profiles, ARTIFACTS
from .storage import write_json
from .data import prepare
from .models import Models
from .retrieval import Index
from .generation import UsageLedger, generate
from .evaluation import evaluate, evaluate_qa, no_evidence_checks

def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG + Laya")
    parser.add_argument("command", choices=["prepare", "index", "evaluate", "qa", "ask", "usage", "budget", "models", "benchmark", "check-api", "export-results"])
    parser.add_argument("--dataset", choices=["rubq", "scifact"], help="Default: RuBQ for individual commands; both datasets for benchmark")
    parser.add_argument("--suite", choices=["baseline", "laya"], default="baseline")
    parser.add_argument("--count", type=int)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--question")
    parser.add_argument("--profile", choices=["sub2api", "routerai"], default="sub2api")
    parser.add_argument("--key-stdin", action="store_true", help="Read API key from hidden pipe input; never save it")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--add", type=int, help="Deprecated compatibility option; requests are unlimited")
    args = parser.parse_args()
    if args.key_stdin:
        os.environ[profiles()[args.profile].key_env] = sys.stdin.readline().strip()
    if args.command in ("usage", "budget"):
        print(json.dumps(UsageLedger().migrate(), indent=2))
        return
    if args.command == "check-api":
        rows = no_evidence_checks(profiles()[args.profile])
        print(json.dumps([{"case": row["query_id"], "refusal": row["refusal"], "text": row["answer"]["text"], "usage": row["answer"]["usage"]} for row in rows], ensure_ascii=False, indent=2))
        return
    if args.command == "export-results":
        from .reporting import export_results
        print(export_results())
        return
    settings = Settings()
    models = Models(settings)
    if args.command == "models":
        _ = models.encoder
        _ = models.laya
        print(json.dumps(models.revisions, indent=2))
        return
    last = [-1]
    def progress(fraction, message):
        step = int(fraction * 20)
        if step != last[0] or fraction == 1:
            print(f"{fraction:.0%} {message}", flush=True)
            last[0] = step
    if args.command == "benchmark":
        import psutil
        from filelock import FileLock
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        marker = ARTIFACTS / "benchmark-running.json"
        with FileLock(str(ARTIFACTS / "benchmark.lock")):
            write_json(marker, {"pid": os.getpid(), "create_time": psutil.Process().create_time()})
            try:
                for name in ((args.dataset,) if args.dataset else ("scifact", "rubq")):
                    print(f"=== {name}: index ===", flush=True)
                    index = Index(prepare(name, progress), models, settings).build(progress)
                    for modes, count in ((("bm25", "dense", "hybrid"), None), (("hybrid", "router", "reranker", "full"), 50)):
                        path, report = evaluate(index, modes, count, progress)
                        print(path, flush=True)
                        print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
            finally:
                marker.unlink(missing_ok=True)
        return
    dataset = prepare(args.dataset or "rubq", progress)
    if args.command == "prepare":
        print(json.dumps(dataset.manifest, ensure_ascii=False, indent=2))
        return
    index = Index(dataset, models, settings)
    (index.rebuild if args.rebuild else index.build)(progress)
    if args.command == "index":
        return
    if args.command == "evaluate":
        modes = ("bm25", "dense", "hybrid") if args.suite == "baseline" else ("hybrid", "router", "reranker", "full")
        count = args.count if args.count is not None else (None if args.suite == "baseline" else 50)
        path, report = evaluate(index, modes, count, progress)
        print(path)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    elif args.command == "qa":
        path, report = evaluate_qa(index, profiles()[args.profile], progress=progress)
        print(path)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    elif args.command == "ask":
        if not args.question:
            parser.error("--question is required")
        result = index.retrieve(args.question, args.mode)
        answer = generate(args.question, result.hits[:settings.context_k], profiles()[args.profile])
        print(answer.text)
        print(json.dumps(answer.usage, ensure_ascii=False))

if __name__ == "__main__":
    main()
