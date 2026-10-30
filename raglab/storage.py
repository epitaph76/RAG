import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from dataclasses import asdict, is_dataclass
from importlib.metadata import version, PackageNotFoundError

def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents a cancelled index/report from looking complete.
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=json_default)
        # On Windows, a reader (UI progress, antivirus, or another process) can
        # briefly hold the destination without FILE_SHARE_DELETE. Preserve the old
        # complete JSON until replacement succeeds; never fall back to truncation.
        delays = (0.02, 0.04, 0.08, 0.16, 0.25, 0.25)
        for attempt in range(len(delays) + 1):
            try:
                os.replace(temp, path)
                break
            except PermissionError as error:
                if getattr(error, "winerror", None) not in (5, 32) or attempt == len(delays):
                    raise
                time.sleep(delays[attempt])
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=json_default).encode()).hexdigest()

def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def package_versions():
    result = {}
    for name in ("torch", "bm25s", "sentence-transformers", "faiss-cpu", "laya", "transformers", "numpy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "missing"
    return result
