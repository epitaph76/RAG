from threading import RLock
from pathlib import Path
import os
from .config import ARTIFACTS, Settings
from .storage import read_json, write_json

# Standard HTTPS downloads work reliably behind local proxies; Xet can stall here.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

class Models:
    def __init__(self, settings=Settings()):
        self.settings = settings
        self._encoder = None
        self._laya = None
        self.lock = RLock()
        self.revisions = {}

    def snapshot(self, repo):
        from huggingface_hub import model_info, snapshot_download
        from filelock import FileLock
        path = ARTIFACTS / "model-revisions.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            revisions = read_json(path) if path.exists() else {}
            if repo not in revisions:
                revisions[repo] = model_info(repo).sha
                write_json(path, revisions)
        self.revisions[repo] = revisions[repo]
        snapshots_path = ARTIFACTS / "model-snapshots.json"
        with FileLock(str(snapshots_path) + ".lock"):
            snapshots = read_json(snapshots_path) if snapshots_path.exists() else {}
        cached = snapshots.get(repo, {})
        if cached.get("revision") == revisions[repo] and all(
            (Path(cached["path"]) / name).is_file() and (Path(cached["path"]) / name).stat().st_size == size
            for name, size in cached.get("files", {}).items()
        ) and cached.get("files"):
            return cached["path"]
        local = snapshot_download(repo, revision=revisions[repo],
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
        files = {p.relative_to(local).as_posix(): p.stat().st_size for p in Path(local).rglob("*") if p.is_file()}
        with FileLock(str(snapshots_path) + ".lock"):
            snapshots = read_json(snapshots_path) if snapshots_path.exists() else {}
            snapshots[repo] = {"revision": revisions[repo], "path": local, "files": files}
            write_json(snapshots_path, snapshots)
        return local

    @property
    def encoder(self):
        with self.lock:
            if self._encoder is None:
                import torch
                from sentence_transformers import SentenceTransformer
                torch.set_num_threads(self.settings.threads)
                self._encoder = SentenceTransformer(self.snapshot(self.settings.embedding_model), device="cpu", trust_remote_code=False)
                self._encoder.max_seq_length = 512
            return self._encoder

    @property
    def laya(self):
        with self.lock:
            if self._laya is None:
                import torch
                from laya import Router
                torch.set_num_threads(self.settings.threads)
                self._laya = Router(models={"multilingual": self.snapshot(self.settings.laya_model)},
                                    device="cpu", max_loaded=1)
            return self._laya
