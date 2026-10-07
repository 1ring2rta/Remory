"""Immutable, owner-bound checkpoints. A failed encode never replaces live state."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import io
import json
from pathlib import Path
import re
import secrets
import sqlite3

import numpy as np


@dataclass
class Memory:
    handle: str
    owner: str
    identity: str
    prefix_ids: tuple[int, ...]
    summary_ids: tuple[int, ...]
    embeddings: np.ndarray
    slot_depths: tuple[int, ...]
    receipt: dict


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = str(Path(path).expanduser().resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS memories (
                handle TEXT PRIMARY KEY, owner TEXT NOT NULL,
                metadata TEXT NOT NULL, tensor BLOB NOT NULL)""")
        Path(self.path).chmod(0o600)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _key(owner: str, handle: str | None = None):
        if not isinstance(owner, str) or not 1 <= len(owner) <= 256:
            raise ValueError("owner must be a nonempty session ID of at most 256 characters")
        if handle is not None and not re.fullmatch(r"rm_[0-9a-f]{48}", handle):
            raise ValueError("invalid memory handle")

    def put(self, *, owner, identity, prefix_ids, summary_ids, embeddings,
            slot_depths, receipt) -> Memory:
        self._key(owner)
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or not array.shape[0] or not np.isfinite(array).all():
            raise ValueError("memory must be a nonempty finite matrix")
        if len(slot_depths) != array.shape[0] or any(type(d) is not int or d < 1 for d in slot_depths):
            raise ValueError("invalid logical memory depths")
        handle = "rm_" + secrets.token_hex(24)
        metadata = dict(identity=identity, prefix_ids=list(prefix_ids), summary_ids=list(summary_ids),
                        slot_depths=list(slot_depths), receipt=receipt)
        tensor = io.BytesIO()
        np.save(tensor, array, allow_pickle=False)
        with self._connect() as db:
            db.execute("INSERT INTO memories VALUES (?, ?, ?, ?)",
                       (handle, owner, json.dumps(metadata, allow_nan=False), tensor.getvalue()))
        return self.get(owner, handle)

    def get(self, owner: str, handle: str) -> Memory:
        self._key(owner, handle)
        with self._connect() as db:
            row = db.execute("SELECT metadata, tensor FROM memories WHERE handle=? AND owner=?",
                             (handle, owner)).fetchone()
        if row is None:
            raise KeyError("memory not found for this session")
        meta = json.loads(row[0])
        return Memory(handle, owner, meta["identity"], tuple(meta["prefix_ids"]),
                      tuple(meta["summary_ids"]), np.load(io.BytesIO(row[1]), allow_pickle=False),
                      tuple(meta["slot_depths"]), meta["receipt"])

    def delete(self, owner: str, handle: str) -> bool:
        self._key(owner, handle)
        with self._connect() as db:
            return db.execute("DELETE FROM memories WHERE handle=? AND owner=?",
                              (handle, owner)).rowcount == 1
