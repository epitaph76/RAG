"""Export aggregate, credential-free evidence for the public repository."""
import platform
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil

from .config import ROOT, ARTIFACTS, LABELS
from .storage import read_json, write_json

def cpu_name():
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or platform.machine()

def export_results():
    reports = []
    for dataset in ("scifact", "rubq"):
        for suite in ("full", "subset"):
            paths = sorted((ARTIFACTS / "reports").glob(f"{dataset}-{suite}-*.json"))
            if not paths:
                raise ValueError(f"Missing completed benchmark: {dataset}/{suite}")
            report = read_json(paths[-1])
            reports.append({"dataset": dataset, "suite": suite, "summary": report["summary"],
                "query_ids": report["query_ids"], "seed": report["seed"],
                "data_manifest": report["dataset"], "index": report["index"],
                "model_revisions": report["model_revisions"], "source_report": paths[-1].name})
    paths = sorted((ARTIFACTS / "reports").glob("rubq-qa-*.json"))
    if not paths:
        raise ValueError("Missing completed QA benchmark")
    qa = read_json(paths[-1])
    junit = ARTIFACTS / "test-results.xml"
    if not junit.exists():
        raise ValueError("Run pytest --junitxml=artifacts/test-results.xml before exporting")
    suites = ET.parse(junit).getroot().iter("testsuite")
    tests = [{key: int(suite.attrib.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")} for suite in suites]
    totals = {key: sum(suite[key] for suite in tests) for key in ("tests", "failures", "errors", "skipped")}
    if totals["failures"] or totals["errors"] or not totals["tests"]:
        raise ValueError("Tests have not passed")
    output = {"schema": 1, "generated_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "hardware": {"cpu": cpu_name(), "logical_cpus": psutil.cpu_count(),
                     "ram_gib": psutil.virtual_memory().total / 1024**3, "os": platform.platform()},
        "tests": totals, "retrieval": reports,
        "qa": {"model": qa["profile"]["model"], "profile": qa["profile"]["name"], "summary": qa["summary"],
               "query_ids": qa["query_ids"], "no_evidence_passed": qa["no_evidence_passed"]}}
    write_json(ROOT / "benchmarks" / "results.json", output)
    lines = ["# Измерения Hybrid RAG + Laya", "", f"Дата: {output['generated_at']}. {output['hardware']['cpu']}, {output['hardware']['ram_gib']:.1f} GiB RAM, CPU, четыре потока.",
        "", "Результаты получены на этом ноутбуке. Полные JSON/CSV с ранжированиями остаются локально в `artifacts/reports/`; агрегаты и точные query IDs опубликованы в [results.json](benchmarks/results.json).", ""]
    for suite, title in (("full", "Baseline · полная тестовая выборка"), ("subset", "Laya · одинаковые 50 запросов на датасет")):
        lines.extend([f"## {title}", "", "| Данные | Режим | N | Recall@10 | nDCG@10 | MRR@10 | p50, мс | p95, мс | RSS, GiB | Fallback |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for report in reports:
            if report["suite"] == suite:
                for row in report["summary"]:
                    lines.append(f"| {report['dataset']} | {LABELS[row['mode']]} | {row['queries']} | {row['Recall@10']:.4f} | {row['nDCG@10']:.4f} | {row['MRR@10']:.4f} | {row['p50_ms']:.1f} | {row['p95_ms']:.1f} | {row['peak_rss_gib']:.2f} | {row['fallback_rate']:.0%} |")
        lines.append("")
    lines.extend(["## QA smoke-test · четыре вопроса RuBQ", "", f"Генератор: `{qa['profile']['model']}` через `{qa['profile']['name']}`. Проверки недостаточного контекста: {'2/2' if qa['no_evidence_passed'] else 'есть ошибки'}.", "",
        "| Режим | EM | F1 | Input tokens | Output tokens | p50, мс | p95, мс |",
        "|---|---:|---:|---:|---:|---:|---:|"])
    for row in qa["summary"]:
        lines.append(f"| {LABELS[row['mode']]} | {row['EM']:.3f} | {row['F1']:.3f} | {row['prompt_tokens']} | {row['completion_tokens']} | {row['p50_ms']:.1f} | {row['p95_ms']:.1f} |")
    lines.extend(["", "## Как читать результаты", "",
        "- p50/p95 retrieval — тёплые измерения; загрузка и первый запрос записаны отдельно в исходных отчётах. RSS относится к процессу приложения и не включает Windows/Docker.",
        "- Fallback включает низкую уверенность router и ошибки Laya. Предупреждения сохранены по каждому запросу; fallback нельзя выдавать за успешное решение router.",
        "- Полную выборку и выборку из 50 запросов следует сравнивать отдельно. Все четыре Laya-конфигурации используют одни query IDs, seed=42; настройки на test не подбирались.",
        "- Четыре QA-вопроса — проверка интеграции, а не статистически надёжная оценка. EM/F1 считаются по полному ответу без ссылок, поэтому поясняющие предложения снижают оценку.",
        "- Проверка цитат контролирует наличие ссылок и номера переданных источников; семантическую корректность каждого утверждения она не доказывает.",
        "", f"Автоматические тесты: **{totals['tests']}**, ошибок: **{totals['failures'] + totals['errors']}**.",
        "", "Воспроизведение: `python -m raglab.cli benchmark`, затем `python -m raglab.cli qa` с выбранным API-профилем. Статистика обращений: `python -m raglab.cli usage`; внутренней квоты нет. Версии данных/моделей и параметры находятся в JSON; зависимости — в `requirements.lock.txt`.", ""])
    (ROOT / "BENCHMARKS.md").write_text("\n".join(lines), encoding="utf-8")
    return ROOT / "BENCHMARKS.md"
