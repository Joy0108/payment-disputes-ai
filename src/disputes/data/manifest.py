"""Content-addressed data manifest.

The part of DVC that matters for reproducibility, without the tool: every
tracked file gets a content hash and a size, written to a manifest that is
committed. A model version records the manifest hash it trained against, so
"which data produced this model" has an answer that survives the data being
regenerated.

``.dvc`` pointer files are emitted alongside so the same tracking works if the
project is later put behind real DVC remote storage; the format is the same
small YAML DVC itself writes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import DATA_DIR


@dataclass
class TrackedFile:
    name: str
    path: str
    md5: str
    size_bytes: int
    rows: int | None
    tracked_at: str


class DataManifest:
    def __init__(self, path: Path | None = None):
        self.path = path or (DATA_DIR / "manifest.json")
        self.files: dict[str, TrackedFile] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.files = {k: TrackedFile(**v) for k, v in payload.get("files", {}).items()}

    def track(self, name: str, path: Path, write_pointer: bool = True) -> TrackedFile:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        digest, size, rows = _hash_file(path)
        entry = TrackedFile(
            name=name,
            path=_portable_path(path),
            md5=digest,
            size_bytes=size,
            rows=rows,
            tracked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.files[name] = entry
        if write_pointer:
            _write_dvc_pointer(path, entry)
        return entry

    def verify(self) -> dict[str, Any]:
        """Re-hash every tracked file and report what moved.

        This is what makes "the model is stale" a detectable state rather than
        a suspicion: if the manifest hash for gold/complaints no longer matches
        the file, every model trained against it is training on different data
        than it claims.
        """
        changed, missing = [], []
        for name, entry in self.files.items():
            path = _resolve(entry.path)
            if not path.exists():
                missing.append(name)
                continue
            digest, size, _rows = _hash_file(path)
            if digest != entry.md5:
                changed.append({"name": name, "expected": entry.md5, "actual": digest, "size_delta": size - entry.size_bytes})
        return {"tracked": len(self.files), "changed": changed, "missing": missing, "clean": not changed and not missing}

    def fingerprint(self) -> str:
        """One hash over the whole manifest. What a model version records."""
        payload = json.dumps({k: v.md5 for k, v in sorted(self.files.items())}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"fingerprint": self.fingerprint(), "files": {k: asdict(v) for k, v in sorted(self.files.items())}},
                indent=2,
            ),
            encoding="utf-8", newline="\n",
        )
        return self.path

    def summary(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint(),
            "files": {k: {"md5": v.md5[:12], "rows": v.rows, "size_bytes": v.size_bytes} for k, v in sorted(self.files.items())},
        }


def _portable_path(path: Path) -> str:
    """Repo-relative where possible, absolute otherwise.

    A manifest is only portable if its paths are relative to the repository, but
    a file outside it is a legitimate thing to track (a mounted extract, a
    temporary fixture) and crashing on one is not.
    """
    try:
        return str(path.relative_to(DATA_DIR.parent)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _resolve(entry_path: str) -> Path:
    candidate = Path(entry_path)
    return candidate if candidate.is_absolute() else (DATA_DIR.parent / entry_path)


def _hash_file(path: Path) -> tuple[str, int, int | None]:
    digest = hashlib.md5(usedforsecurity=False)
    size = 0
    rows = 0
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
            rows += chunk.count(b"\n")
    return digest.hexdigest(), size, (rows if path.suffix in {".csv", ".jsonl"} else None)


def _write_dvc_pointer(path: Path, entry: TrackedFile) -> None:
    pointer = path.with_suffix(path.suffix + ".dvc")
    pointer.write_text(
        "outs:\n"
        f"- md5: {entry.md5}\n"
        f"  size: {entry.size_bytes}\n"
        f"  path: {path.name}\n",
        encoding="utf-8", newline="\n",
    )
