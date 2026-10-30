"""PDD context and images resolved from the local corpus, independently of the LLM."""
import hashlib
import json
from pathlib import Path
import re

from .types import Chunk, Hit

CODE = r"(?<![\d.])([1-9]\.\d+(?:\.\d+)?)(?!\d|\.\d)"

def normalize(text):
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold().replace("ё", "е")).split())

def requested_codes(question, signs):
    """An article number is not a sign; bare sign numbers and exact names work."""
    explicit = re.findall(CODE, question)
    if explicit and (re.search(r"\b(?:знак\w*|табличк\w*)", question, re.I)
                     or re.fullmatch(r"\s*\d+\.\d+(?:\.\d+)?[?.\s]*", question)):
        return list(dict.fromkeys(explicit))
    text = " " + normalize(question) + " "
    return list(dict.fromkeys(sign["code"] for sign in signs
        if " " + normalize(sign["name"]) + " " in text))

def mentions_code(text, code):
    target = tuple(map(int, code.split(".")))
    for match in re.finditer(CODE + r"(?:\s*[-–—]\s*([1-9]\.\d+(?:\.\d+)?)(?!\d|\.\d))?", text):
        start, end = match.groups()
        if start == code:
            return True
        if end:
            a, b = tuple(map(int, start.split("."))), tuple(map(int, end.split(".")))
            if len(a) == len(b) == len(target) and a[:-1] == b[:-1] == target[:-1] and a <= target <= b:
                return True
    return False

def relevant_rules(text, codes):
    paragraphs = text.split("\n\n")
    selected = {0} if paragraphs else set()
    for i, paragraph in enumerate(paragraphs):
        if "утратил" not in paragraph.casefold() and any(mentions_code(paragraph, code) for code in codes):
            selected.add(i)
            if re.match(r"^(?:для знаков|[1-9]\.\d+)", paragraph, re.I):
                heading = next((j for j in range(i - 1, -1, -1) if paragraphs[j].rstrip().endswith(":")), None)
                if heading is not None:
                    selected.add(heading)
    return "\n\n".join(paragraphs[i] for i in sorted(selected)) if len(selected) > 1 else ""

class TrafficKnowledge:
    def __init__(self, documents, directory):
        self.directory = Path(directory)
        self.documents = {doc.id: doc for doc in documents}
        self.signs = self._rows("signs.jsonl")
        self.photos = self._rows("photos.jsonl")
        self.by_id = {row["id"]: row for row in self.signs}

    def _rows(self, name):
        path = self.directory / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []

    def sources(self, question, hits, limit=4):
        codes = requested_codes(question, self.signs)
        if not codes:
            return hits[:limit]
        signs = [next((row for row in self.signs if row["code"] == code), None) for code in codes]
        signs = [row for row in signs if row and row["id"] in self.documents][:limit]
        if not signs:
            return []
        output, seen = [], set()
        def add(document, text=None):
            if document.id not in seen:
                seen.add(document.id)
                output.append(Hit(Chunk(document.id + ":context", document.id,
                    document.text if text is None else text, document.title, document.url), 1.0))
        for sign in signs:
            document = self.documents[sign["id"]]
            # The linked standard is added separately with its own source URL.
            text = document.text.split("\n\nГОСТ Р 52289-2019", 1)[0]
            text = re.sub(r"Изображение: [^\n]+?\.(?= Общие)", "", text)
            add(document, text)
        for sign in signs:
            document = self.documents.get(sign.get("rules_document_id"))
            if document:
                text = relevant_rules(document.text, codes)
                if text:
                    add(document, text)
        for sign in signs:
            document = self.documents.get(sign.get("application_document_id"))
            if document:
                add(document)
        for hit in hits:
            if hit.chunk.document_id.startswith("gost-sign:") and hit.chunk.document_id not in seen:
                continue
            if hit.chunk.document_id not in seen:
                seen.add(hit.chunk.document_id)
                output.append(hit)
        return output[:limit]

    def images(self, question, sources, answer=None):
        codes = requested_codes(question, self.signs)
        cited = set(map(int, re.findall(r"\[(\d+)\]", answer or "")))
        found, seen = [], set()
        for number, hit in enumerate(sources, 1):
            sign = self.by_id.get(hit.chunk.document_id)
            if not sign or sign["code"] in seen:
                continue
            if codes and sign["code"] not in codes and (answer is None or number not in cited or not mentions_code(answer, sign["code"])):
                continue
            if answer is not None and number not in cited and sign["code"] not in codes:
                continue
            seen.add(sign["code"])
            found.append({**sign, "source_number": number})
            if re.search(r"фото|на дороге|реальн\w* сним", question, re.I):
                found.extend({**photo, "name": sign["name"], "source_number": number}
                    for photo in [p for p in self.photos if p["code"] == sign["code"]][:2])
        return found

    def image_bytes(self, asset):
        """Do not render URLs, path traversal, or a mismatched picture from a record."""
        known = next((row for row in self.signs + self.photos if row["id"] == asset.get("id")), None)
        if known is None:
            raise ValueError("Изображение отсутствует в каталоге")
        path = (self.directory / known["local_path"]).resolve()
        if not path.is_relative_to((self.directory / "images").resolve()):
            raise ValueError("Некорректный путь изображения")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != known["sha256"]:
            raise ValueError("Изображение не прошло проверку целостности")
        return content
