"""Versioned Russian traffic-law corpus and linked sign images (no generated law text).

Run ``python -m raglab.pdd`` to collect a snapshot; ``--refresh`` fetches new sources.
The rich JSONL keeps provenance; dataset.json is the existing RAG adapter format.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import requests

from .config import DATA, SEED
from .storage import file_hash, read_json, write_json

CONSULTANT = "https://www.consultant.ru"
PDD_URL = CONSULTANT + "/document/cons_doc_LAW_2709/"
KOAP_URL = CONSULTANT + "/document/cons_doc_LAW_34661/"
GOST_URL = "https://rags.ru/documents/dop_documents/0/odm/0/gost.html"
GOST_OFFICIAL = "https://protect.gost.ru/gost/details/39b75217-29bd-492b-94f4-c0dccb72b299"
AMENDMENT = "https://publication.pravo.gov.ru/document/0001202608310048"
USAGE_URL = CONSULTANT + "/document/cons_doc_LAW_348566/"
USAGE_OFFICIAL = "https://protect.gost.ru/gost/details/94160176-3252-4e7d-8534-2d172e6aa0de"
CORRECTION_OFFICIAL = "https://protect.gost.ru/gost/changesdetails/5e229d99-c816-43b1-ac09-ae7b63eb4879"
RTSD_REPO = "eleldar/rtsd_cleaned"
CODE = re.compile(r"[1-8]\.\d+(?:\.\d+)?")


def clean(text):
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


class Collector:
    def __init__(self, directory, refresh=False, progress=None):
        self.directory = Path(directory)
        self.refresh = refresh
        self.progress = progress or (lambda p, m: print(m, flush=True))
        self.sources = {}
        self.errors = []

    def fetch(self, url, relative=None):
        relative = relative or "raw/" + hashlib.sha256(url.encode()).hexdigest()[:24] + ".html"
        path = self.directory / relative
        sidecar = path.with_suffix(path.suffix + ".source.json")
        if self.refresh or not path.exists() or not sidecar.exists():
            response = requests.get(url, timeout=(5, 10) if url == AMENDMENT else (15, 60),
                headers={"User-Agent": "RagLab-PDD/1.0 (research corpus)"})
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(response.content)
            temporary.replace(path)
            write_json(sidecar, {"url": url, "resolved_url": response.url,
                "retrieved_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
                "content_type": response.headers.get("Content-Type", ""), "sha256": file_hash(path)})
        metadata = read_json(sidecar)
        if file_hash(path) != metadata["sha256"]:
            raise ValueError(f"Corrupted source: {relative}")
        self.sources[url] = {**metadata, "local_path": relative}
        return path

    def soup(self, url):
        return BeautifulSoup(self.fetch(url).read_bytes(), "html.parser")


def edition(soup):
    match = re.search(r"ред\. от (\d{2}\.\d{2}\.\d{4})", soup.get_text(" ", strip=True))
    if not match:
        changes = soup.select_one(".doc-roll__content")
        dates = re.findall(r"от (\d{2}\.\d{2}\.\d{4})", changes.get_text() if changes else "")
        if dates:
            return max(datetime.strptime(d, "%d.%m.%Y").date() for d in dates).isoformat()
        raise ValueError("Source has no edition date")
    return datetime.strptime(match[1], "%d.%m.%Y").date().isoformat()


def toc(soup, base):
    area = soup.select_one(".document-page__toc")
    if not area:
        raise ValueError(f"Missing table of contents: {base}")
    return [(clean(a.get_text(" ", strip=True)), urljoin(base, a["href"]))
            for a in area.select("a[href]")]


def legal_blocks(soup):
    """Strip publisher annotations, never substantive footnotes or table cells."""
    area = soup.select_one(".document-page__content")
    if not area or not area.select_one("h1"):
        raise ValueError("Missing legal content (redirect, captcha or changed layout)")
    for element in area.select(".doc-edit, .doc-insert, .document-page__toc, script, style"):
        element.decompose()
    output = []
    for element in area.select("p, table"):
        if element.find_parent("table") or element.find_parent("h1"):
            continue
        text = clean(element.get_text(" ", strip=True))
        if not text or re.fullmatch(r"[-_]+", text):
            continue
        anchor = element.find("a", id=True)
        output.append((text, anchor["id"] if anchor else ""))
    if not output:
        raise ValueError("Empty legal page")
    return output


def split_blocks(blocks, numbered=True):
    groups, buffer, anchor = [], [], ""
    for text, local_anchor in blocks:
        pattern = r"^\d+\.\d+(?:\.\d+){0,2}\s" if numbered == "standard" else r"^\d+(?:\.\d+){0,3}\.\s"
        start = numbered and re.match(pattern, text)
        if start and buffer:
            groups.append((buffer, anchor))
            buffer = []
        if not buffer:
            anchor = local_anchor
        buffer.append(text)
    if buffer:
        groups.append((buffer, anchor))
    return groups


def parse_legal_page(collector, spec):
    family, number, title, url, revision = spec
    soup = collector.soup(url)
    try:
        page_revision = edition(soup)
    except ValueError:
        page_revision = None
    if page_revision is not None and page_revision != revision:
        raise ValueError(f"Edition changed during collection: {url}")
    blocks = legal_blocks(soup)
    # Entire articles and sign groups preserve exceptions and cross-references.
    numbered = "standard" if family == "sign_application_standard" else family in ("pdd", "admission", "malfunctions", "decree")
    rows = []
    for index, (paragraphs, anchor) in enumerate(split_blocks(blocks, numbered)):
        body = "\n\n".join(paragraphs)
        heading = f"{family}:{number}:{index}"
        clause_pattern = r"^(\d+\.\d+(?:\.\d+){0,2})\s" if numbered == "standard" else r"^(\d+(?:\.\d+){0,3})\.\s"
        clause = re.match(clause_pattern, paragraphs[0])
        status = "repealed" if re.match(r"^\d+(?:\.\d+)*\.\s+(?:Утратил[аи]? силу|Исключен)", body, re.I) else "active"
        rows.append({"id": heading, "title": title + (f" · п. {clause[1]}" if clause and numbered else ""),
            "text": f"{title}. Редакция от {revision}.\n\n{body}",
            "url": url + ("#" + anchor if anchor else ""), "source_url": url,
            "family": family, "section": number, "clause": clause[1] if clause and numbered else None,
            "edition_date": revision, "jurisdiction": "RU", "language": "ru", "status": status,
            "edition_basis": "root_publisher_header_or_latest_amending_act; not_independent_official_consolidation",
            "content_type": "normative_text", "license": "standard_text; source_attribution_retained" if numbered == "standard" else "official_document_text_GK_RF_1259_6_1",
            "sign_codes": sorted(set(CODE.findall(body))) if family == "sign_rules" else []})
    return rows


def expand_table(table):
    """Expand row/col spans to keep each image aligned with its printed code."""
    grid = {}
    for r, row in enumerate(table.find_all("tr")):
        c = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while (r, c) in grid:
                c += 1
            width, height = int(cell.get("colspan", 1)), int(cell.get("rowspan", 1))
            for dr in range(height):
                for dc in range(width):
                    grid[r + dr, c + dc] = cell
            c += width
    return grid


def parse_sign_catalog(soup):
    result = []
    # Appendix A spans Par917..Par2200 in this source. Use headings rather than
    # hardcoded IDs so an upstream re-export does not silently map the wrong table.
    area = soup.select_one(".contener_doc")
    if area is None:
        raise ValueError("ГОСТ: missing document body")
    in_appendix = False
    for element in area.find_all(["div", "table"], recursive=False):
        text = clean(element.get_text(" ", strip=True))
        if element.name == "div" and text.startswith("ИЗОБРАЖЕНИЯ, НОМЕРА И НАИМЕНОВАНИЯ ЗНАКОВ"):
            in_appendix = True
        if element.name == "div" and text.startswith("ИЗОБРАЖЕНИЯ ЗНАКОВ НА МАСШТАБНОЙ СЕТКЕ"):
            break
        if not in_appendix or element.name != "table":
            continue
        grid = expand_table(element)
        seen_cells = set()
        for (r, c), cell in grid.items():
            code = clean(cell.get_text(" ", strip=True))
            if id(cell) in seen_cells or not CODE.fullmatch(code):
                continue
            seen_cells.add(id(cell))
            image_cell = None
            for previous in range(r - 1, max(-1, r - 5), -1):
                candidate = grid.get((previous, c))
                if candidate is not None and candidate.find("img"):
                    image_cell = candidate
                    break
            if image_cell is None:
                raise ValueError(f"ГОСТ: no image above code {code}")
            name = ""
            for name_row in (r, r - 1, r - 2):
                for column in range(c + 1, c + 6):
                    candidate = grid.get((name_row, column))
                    if candidate is not None:
                        value = clean(candidate.get_text(" ", strip=True))
                        if re.search("[А-Яа-я]", value) and "номер знака" not in value.lower():
                            name = value
                            break
                if name:
                    break
            if not name:
                raise ValueError(f"ГОСТ: missing name {code}")
            anchor = cell.find("a", id=True)
            for variant, img in enumerate(image_cell.select("img[src]"), 1):
                result.append({"id": f"gost-sign:{code}:{variant}", "code": code, "name": name,
                    "image_url": urljoin(GOST_URL, img["src"]),
                    "source_url": GOST_URL + ("#" + anchor["id"] if anchor else ""),
                    "standard": "ГОСТ Р 52290-2024", "effective_from": "2026-01-01",
                    "official_catalog_url": GOST_OFFICIAL, "kind": "reference_illustration",
                    "license": "standard_image; source attribution retained; no blanket CC license asserted"})
    unique = {row["id"]: row for row in result}
    if len(unique) < 300:
        raise ValueError(f"ГОСТ: suspiciously small sign catalog ({len(unique)})")
    return list(unique.values())


def download_sign(collector, row):
    suffix = Path(row["image_url"]).suffix
    path = collector.fetch(row["image_url"], f"images/signs/{row['code']}-{row['id'].rsplit(':', 1)[1]}{suffix}")
    from PIL import Image
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        size = image.size
    return {**row, "local_path": path.relative_to(collector.directory).as_posix(),
        "sha256": file_hash(path), "width": size[0], "height": size[1]}


def collect_photos(collector, valid_codes, per_class=3):
    """Small deterministic real-photo sample; the full original parquet is pinned."""
    import pyarrow.parquet as pq
    from PIL import Image
    api_url = f"https://huggingface.co/api/datasets/{RTSD_REPO}"
    metadata = json.loads(collector.fetch(api_url, "raw/rtsd_repository.json").read_text(encoding="utf-8"))
    revision = metadata["sha"]
    filename = next(x["rfilename"] for x in metadata["siblings"] if x["rfilename"].endswith(".parquet"))
    url = f"https://huggingface.co/datasets/{RTSD_REPO}/resolve/{revision}/{filename}"
    path = collector.fetch(url, "raw/rtsd_photos.parquet")
    candidates, output = {}, []
    ordered_codes = sorted(valid_codes, key=len, reverse=True)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for entry in batch.to_pylist():
            label = entry["sign_class"]
            if not re.fullmatch(r"[A-Za-z0-9_]+", label):
                continue
            normalized = label.replace("_", ".")
            code = next((code for code in ordered_codes if normalized == code or normalized.startswith(code + ".")), None)
            if code is None:
                continue
            payload = entry["image"].get("bytes")
            if not payload:
                continue
            with Image.open(io.BytesIO(payload)) as image:
                size = image.size
            # Keep the largest distinct crops per original class, preserving
            # numeric variants (e.g. 3_24_n40) instead of dropping them.
            if min(size) < 48:
                continue
            pool = candidates.setdefault(label, [])
            digest = hashlib.sha256(payload).hexdigest()
            if any(row[2] == digest for row in pool):
                continue
            pool.append((size[0] * size[1], entry["sign_id"], digest, code, payload))
            pool.sort(key=lambda row: (-row[0], row[1], row[2]))
            del pool[per_class:]
    for label, pool in sorted(candidates.items()):
        for number, (_, sign_id, _, code, payload) in enumerate(pool, 1):
            image = Image.open(io.BytesIO(payload))
            image.load()
            relative = f"images/photos/{label}-{number}.jpg"
            destination = collector.directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(destination, quality=95)
            output.append({"id": f"rtsd:{label}:{number}", "code": code, "original_class": label,
                "variant_suffix": label.replace("_", ".")[len(code):].lstrip("."),
                "kind": "real_world_photo_crop", "local_path": relative,
                "sha256": file_hash(destination), "width": image.width, "height": image.height,
                "source_url": f"https://huggingface.co/datasets/{RTSD_REPO}/tree/{revision}",
                "revision": revision, "original_sign_id": sign_id,
                "license": "not_specified_by_mirror; verify upstream rights before redistribution",
                "legal_currentness": "historical_photo; code matched to 2024 catalog; not evidence of current law",
                "split": "reference_only_not_a_benchmark"})
    return output


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sign_definitions(collector, specs):
    definitions = {}
    for family, _, _, url, _ in specs:
        if family != "sign_rules":
            continue
        groups, codes, buffer = [], [], []
        for text, anchor in legal_blocks(collector.soup(url)):
            match = re.match(r'^((?:[1-8]\.\d+(?:\.\d+)?(?:\s*<[^>]+>)?\s*[,–—-]?\s*)+)["«]', text)
            if match:
                if buffer:
                    groups.append((codes, buffer))
                prefix = match[1]
                codes, buffer = [], []
                for code_range in re.finditer(r'([1-8]\.\d+(?:\.\d+)?)(?:\s*-\s*([1-8]\.\d+(?:\.\d+)?))?', prefix):
                    start, end = code_range.groups()
                    codes.append(start)
                    if end:
                        a, b = start.split('.'), end.split('.')
                        if len(a) == len(b) and a[:-1] == b[:-1]:
                            codes.extend('.'.join(a[:-1] + [str(n)]) for n in range(int(a[-1]) + 1, int(b[-1]) + 1))
                        else:
                            codes.append(end)
                codes.extend(re.findall(r'([1-8]\.\d+(?:\.\d+)?)(?:\s*<[^>]+>)?\s+["«]', text))
                codes = list(dict.fromkeys(codes))
            if codes:
                buffer.append(text)
        if buffer:
            groups.append((codes, buffer))
        for codes, paragraphs in groups:
            for code in codes:
                definitions[code] = {"text": "\n\n".join(paragraphs), "url": url}
    return definitions


def build(directory=DATA / "pdd", refresh=False, photos=True, progress=None):
    collector = Collector(directory, refresh, progress)
    directory = collector.directory
    pdd = collector.soup(PDD_URL)
    pdd_revision = edition(pdd)
    specs, category, section_count = [], "decree", 0
    for title, url in toc(pdd, PDD_URL):
        if "Утратило силу" in title:
            continue
        if title == "Правила дорожного движения Российской Федерации":
            category = "pdd"
            continue
        if title.startswith("Приложение 1."):
            category = "sign_rules"
            continue
        if title.startswith("Приложение 2."):
            category = "markings"
            continue
        if title.startswith("Основные положения"):
            category = "admission"
        if title.startswith("Приложение. Перечень"):
            category = "malfunctions"
        number = re.match(r"^(\d+)\.", title)
        key = number[1] if number else "general"
        if category == "pdd":
            section_count += 1
        specs.append((category, key, ("ПДД РФ · " if category in ("pdd", "sign_rules", "markings") else "Постановление №1090 · ") + title, url, pdd_revision))
    if section_count != 26:
        raise ValueError(f"Expected all 26 PDD sections, got {section_count}")
    koap = collector.soup(KOAP_URL)
    koap_revision = edition(koap)
    extra = {"2.6.1", "27.12", "27.12.1", "27.13", "30.1", "30.2", "30.3", "32.2", "32.6", "32.7"}
    for title, url in toc(koap, KOAP_URL):
        match = re.match(r"Статья (\d+(?:\.\d+)+)\.", title)
        if match and (match[1].startswith("12.") or match[1] in extra) and "Утратил" not in title:
            specs.append(("koap", match[1], "КоАП РФ · " + title, url, koap_revision))
    # Relevant criminal liability, the road safety law and compulsory insurance.
    for base, family, label, articles in [
        (CONSULTANT + "/document/cons_doc_LAW_10699/", "uk", "УК РФ", {"264", "264.1", "264.2", "264.3", "266", "267", "268"}),
        (CONSULTANT + "/document/cons_doc_LAW_8585/", "road_safety_law", "196-ФЗ О безопасности дорожного движения", None),
        (CONSULTANT + "/document/cons_doc_LAW_36528/", "osago_law", "40-ФЗ Об ОСАГО", None),
    ]:
        soup = collector.soup(base)
        revision = edition(soup)
        for title, url in toc(soup, base):
            match = re.match(r"Статья (\d+(?:\.\d+)*)\.", title)
            if match and (articles is None or match[1] in articles) and "Утратил" not in title:
                specs.append((family, match[1], label + " · " + title, url, revision))
    usage = collector.soup(USAGE_URL)
    usage_revision = edition(usage)
    for title, url in toc(usage, USAGE_URL):
        if title.startswith("Приложение"):
            break
        match = re.match(r"^(\d+(?:\.\d+)*)\s", title)
        table = re.match(r"^Таблица (\d+)", title)
        if match and ("." in match[1] or int(match[1]) <= 4) or table:
            key = match[1] if match else "table-" + table[1]
            specs.append(("sign_application_standard", key, "ГОСТ Р 52289-2019 · " + title, url, usage_revision))
    collector.progress(.05, f"Загрузка {len(specs)} правовых разделов и статей")
    with ThreadPoolExecutor(max_workers=4) as pool:
        parsed = list(pool.map(lambda spec: parse_legal_page(collector, spec), specs))
    records = [row for page in parsed for row in page]
    collector.progress(.25, "Каталог изображений ГОСТ Р 52290-2024")
    official = collector.soup(GOST_OFFICIAL)
    if "01.01.2026" not in official.get_text():
        raise ValueError("Could not verify standard effective date")
    correction_urls = {urljoin(GOST_OFFICIAL, a["href"]) for a in official.select('a[href*="/gost/changesdetails/"]')}
    if correction_urls - {CORRECTION_OFFICIAL}:
        raise ValueError("New sign-standard correction requires review before replacing this snapshot")
    signs = parse_sign_catalog(collector.soup(GOST_URL))
    with ThreadPoolExecutor(max_workers=4) as pool:
        signs = list(pool.map(lambda row: download_sign(collector, row), signs))
    rules = {str(i): row["id"] for i in range(1, 9) for row in records
             if row["family"] == "sign_rules" and row["section"] == str(i)}
    definitions = sign_definitions(collector, specs)
    collector.soup(USAGE_OFFICIAL)
    collector.soup(CORRECTION_OFFICIAL)
    for sign in signs:
        sign["rules_document_id"] = rules[sign["code"].split(".")[0]]
        definition = definitions.get(sign["code"])
        sign["pdd_definition_status"] = "matched" if definition else "not_found_in_consolidated_pdd_appendix"
        sign["standard_correction"] = {"effective_from": "2026-08-10", "official_url": CORRECTION_OFFICIAL,
            "scope": "Appendix G table G.3 letter width; Appendix A sign illustrations unaffected",
            "scope_source": "https://base.garant.ru/414741817/"}
        legal_text = f"\n\nПриложение 1 ПДД РФ (ред. {pdd_revision}):\n{definition['text']}" if definition else "\n\nОтдельное описание этого кода в загруженном приложении 1 ПДД не найдено. Применение следует проверять по ГОСТ Р 52289-2019."
        application = [r for r in records if r["family"] == "sign_application_standard"
                       and re.search(r"(?<![\d.])" + re.escape(sign["code"]) + r"(?![\d.])", r["text"])
                       and "Знак" in r["text"]]
        if application:
            # Prefer the clause that introduces the sign over incidental references.
            preferred = next((r for r in application if re.search(r"Знак(?:и|ом|ов)?\s+" + re.escape(sign["code"]) + r"(?![\d.])", r["text"])), application[0])
            sign["application_document_id"] = preferred["id"]
            sign["application_source_url"] = preferred["url"]
            legal_text += "\n\n" + preferred["text"]
        records.append({"id": sign["id"], "title": f"Знак {sign['code']} · {sign['name']}",
            "text": f"Дорожный знак {sign['code']} «{sign['name']}». ГОСТ Р 52290-2024, приложение А. Действует с 01.01.2026. Изображение: {sign['local_path']}. Общие условия и исключения см. в соответствующей группе приложения 1 ПДД РФ." + legal_text,
            "url": definition["url"] if definition else sign.get("application_source_url", sign["source_url"]),
            "source_url": GOST_URL, "related_source_urls": [sign["source_url"], *([definition["url"]] if definition else []), *([sign["application_source_url"]] if "application_source_url" in sign else [])], "family": "sign_catalog",
            "section": sign["code"], "edition_date": "2024-12-25", "effective_from": "2026-01-01",
            "status": "active", "jurisdiction": "RU", "language": "ru", "content_type": "sign_reference",
            "image_path": sign["local_path"], "sign_codes": [sign["code"]], "license": sign["license"]})
    collector.progress(.6, f"Сохранено {len(signs)} изображений; сбор фотографий")
    photo_rows = collect_photos(collector, {s["code"] for s in signs}) if photos else []
    # Official publication can be temporarily unreachable; record the verification
    # gap explicitly without replacing the consolidated law with an older version.
    try:
        collector.fetch(AMENDMENT)
        official_amendment = "downloaded; consolidated text not reconstructed from amendment"
    except requests.RequestException as error:
        official_amendment = "unreachable; edition cross-check required"
        collector.errors.append({"url": AMENDMENT, "error": type(error).__name__})
    for row in records:
        source = collector.sources.get(row["source_url"], {})
        row["retrieved_at"] = source.get("retrieved_at")
        row["source_sha256"] = source.get("sha256")
    active = [r for r in records if r["status"] == "active"]
    normalized = {"documents": [{k: row[k] for k in ("id", "text", "title", "url")} for row in active],
        "queries": {"dev": [], "test": []}, "qrels": {"dev": {}, "test": {}}}
    write_json(directory / "dataset.json", normalized)
    jsonl(directory / "corpus.jsonl", records)
    jsonl(directory / "signs.jsonl", signs)
    jsonl(directory / "photos.jsonl", photo_rows)
    write_json(directory / "sources.json", list(collector.sources.values()))
    counts = dict(Counter(r["family"] for r in active))
    coverage = {"pdd_sections": sorted({int(r["section"]) for r in records if r["family"] == "pdd"}),
        "sign_groups": sorted(rules), "families": counts, "sign_images": len(signs),
        "unique_sign_codes": len({s["code"] for s in signs}), "real_photos": len(photo_rows),
        "photo_classes": len({p["code"] for p in photo_rows}), "errors": collector.errors}
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate corpus IDs")
    write_json(directory / "coverage.json", coverage)
    manifest = {"schema": 1, "dataset": "pdd", "snapshot_date": datetime.now(ZoneInfo("Europe/Moscow")).date().isoformat(),
        "revision": {"pdd": pdd_revision, "koap": koap_revision, "sign_standard": "ГОСТ Р 52290-2024", "application_standard": usage_revision},
        "sources": [PDD_URL, KOAP_URL, GOST_URL, GOST_OFFICIAL, USAGE_URL, USAGE_OFFICIAL, CORRECTION_OFFICIAL], "documents": len(active),
        "normalized_sha256": file_hash(directory / "dataset.json"), "seed": SEED,
        "license": "mixed; see per-record provenance and DATASET_CARD.md",
        "splits": {s: {"queries": 0, "eligible": 0, "excluded_no_paragraph_qrels": 0} for s in ("dev", "test")},
        "evaluation": "No independent gold labels; retrieval examples are not a benchmark",
        "official_amendment_verification": official_amendment,
        "coverage": coverage,
        "files": {name: file_hash(directory / name) for name in ("corpus.jsonl", "signs.jsonl", "photos.jsonl", "sources.json", "coverage.json")}}
    write_json(directory / "manifest.json", manifest)
    card = (f"# ПДД РФ: снимок {manifest['snapshot_date']}\n\n"
        f"Документов: {len(active)}; эталонов знаков: {len(signs)}; фотографий: {len(photo_rows)}.\n\n"
        f"ПДД: редакция {pdd_revision}; КоАП: {koap_revision}; применение знаков: ГОСТ Р 52289-2019, редакция {usage_revision}.\n\n"
        "Форматы: dataset.json для RAG; corpus.jsonl с метаданными; signs.jsonl и photos.jsonl со связями и локальными путями; sources.json с датами загрузки и SHA-256.\n\n"
        "Источники нормативных текстов — консолидированные публикации КонсультантПлюс. Статус стандартов проверяется по карточкам Росстандарта.\n\n"
        "Фотографии RTSD исторические; лицензия зеркала не указана. Перед распространением уточните права у авторов. Единой открытой лицензии для всего корпуса не заявляется. Атрибуция и сведения о правах сохранены в каждой записи.\n\n"
        f"Проверка официального опубликования последнего изменения ПДД: {official_amendment}.\n\n"
        "Независимых тестовых вопросов с gold-разметкой нет. Это корпус знаний, а не готовый бенчмарк. Распознавание фотографий не реализовано.\n\n"
        "Проверка файлов: `python -m raglab.pdd --verify`. Обновление: `python -m raglab.pdd --refresh`, затем `python -m raglab.cli index --dataset pdd --rebuild`. Новые поправки к стандарту требуют проверки содержания.\n\n"
        "Основные источники:\n\n" + "\n".join(f"- {url}" for url in manifest["sources"]) + "\n")
    (directory / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    collector.progress(1, f"Готово: {len(active)} документов, {len(signs)} изображений, {len(photo_rows)} фотографий")
    return manifest


def verify(directory=DATA / "pdd"):
    from PIL import Image
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    expected = {"dataset.json": manifest["normalized_sha256"], **manifest["files"]}
    for relative, digest in expected.items():
        if file_hash(directory / relative) != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
    sources = read_json(directory / "sources.json")
    for source in sources:
        if file_hash(directory / source["local_path"]) != source["sha256"]:
            raise ValueError(f"Corrupted original source: {source['url']}")
    rows = [json.loads(line) for line in (directory / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate corpus IDs")
    if {int(r["section"]) for r in rows if r["family"] == "pdd"} != set(range(1, 27)):
        raise ValueError("Incomplete PDD coverage")
    signs = [json.loads(line) for line in (directory / "signs.jsonl").read_text(encoding="utf-8").splitlines()]
    photos = [json.loads(line) for line in (directory / "photos.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = {r["id"] for r in rows}
    codes = {s["code"] for s in signs}
    if not {"3.24", "5.41", "5.42", "6.2.1", "8.27"} <= codes:
        raise ValueError("Missing current sign classes")
    for asset in signs + photos:
        path = (directory / asset["local_path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Asset path escapes dataset")
        if file_hash(path) != asset["sha256"]:
            raise ValueError(f"Image checksum mismatch: {asset['id']}")
        with Image.open(path) as image:
            image.verify()
        if asset["code"] not in codes or ("rules_document_id" in asset and asset["rules_document_id"] not in ids):
            raise ValueError(f"Dangling image reference: {asset['id']}")
    dataset = read_json(directory / "dataset.json")
    if len(dataset["documents"]) != manifest["documents"]:
        raise ValueError("Document count mismatch")
    return {"documents": manifest["documents"], "verified_sources": len(sources),
        "verified_images": len(signs) + len(photos), "pdd_sections": 26, "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA / "pdd")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-photos", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Verify saved files and images without network access")
    args = parser.parse_args()
    result = verify(args.output) if args.verify else build(args.output, args.refresh, not args.no_photos)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
