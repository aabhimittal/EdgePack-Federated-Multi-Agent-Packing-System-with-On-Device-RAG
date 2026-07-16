"""Encrypted on-device vector store (SQLite + AES-256-GCM).

Threat model: the SQLite file at rest (device backup, stolen storage, cloud
sync) reveals nothing — text AND vectors are encrypted per record, with the
record ID bound as GCM associated data.  Only record IDs and the KDF salt are
plaintext.

Search: vectors are decrypted **into memory only** and held in a normalized
matrix for fast cosine top-k.  Document text is decrypted lazily, one record
at a time, only for returned hits.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..crypto.encryption import RecordCipher, new_salt
from .embeddings import Embedder, HashingEmbedder, cosine_top_k

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB);
CREATE TABLE IF NOT EXISTS docs (
    id TEXT PRIMARY KEY,
    enc_text BLOB NOT NULL,
    enc_vector BLOB NOT NULL
);
"""


@dataclass
class SearchHit:
    doc_id: str
    score: float
    text: str
    metadata: dict


class EncryptedVectorStore:
    def __init__(self, path: str | Path, passphrase: str, embedder: Embedder | None = None):
        self.path = Path(path)
        self.embedder = embedder or HashingEmbedder()
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(_SCHEMA)

        salt = self._get_meta("salt")
        if salt is None:
            salt = new_salt()
            self._set_meta("salt", salt)
            self._set_meta("dim", str(self.embedder.dim).encode())
        stored_dim = int(self._get_meta("dim").decode())
        if stored_dim != self.embedder.dim:
            raise ValueError(f"store was built with dim={stored_dim}, embedder has dim={self.embedder.dim}")

        self.cipher = RecordCipher.from_passphrase(passphrase, salt)
        # in-memory (plaintext) search index: ids + normalized vector matrix
        self._ids: list[str] = []
        self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float64)
        self._load_index()

    # -------------------------------------------------------------- metadata
    def _get_meta(self, key: str) -> bytes | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: bytes) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
        self.conn.commit()

    # ------------------------------------------------------------------ CRUD
    def add(self, text: str, metadata: dict | None = None, doc_id: str | None = None) -> str:
        doc_id = doc_id or uuid.uuid4().hex
        vec = self.embedder.embed(text)
        payload = json.dumps({"text": text, "metadata": metadata or {}}).encode("utf-8")
        enc_text = self.cipher.encrypt(payload, f"text:{doc_id}")
        enc_vec = self.cipher.encrypt(vec.astype(np.float32).tobytes(), f"vec:{doc_id}")
        self.conn.execute(
            "INSERT OR REPLACE INTO docs VALUES (?, ?, ?)", (doc_id, enc_text, enc_vec)
        )
        self.conn.commit()
        self._index_add(doc_id, vec)
        return doc_id

    def add_many(self, texts: list[str]) -> list[str]:
        return [self.add(t) for t in texts]

    def get(self, doc_id: str) -> tuple[str, dict]:
        row = self.conn.execute("SELECT enc_text FROM docs WHERE id=?", (doc_id,)).fetchone()
        if row is None:
            raise KeyError(doc_id)
        payload = json.loads(self.cipher.decrypt(row[0], f"text:{doc_id}"))
        return payload["text"], payload["metadata"]

    def delete(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM docs WHERE id=?", (doc_id,))
        self.conn.commit()
        if doc_id in self._ids:
            i = self._ids.index(doc_id)
            self._ids.pop(i)
            self._matrix = np.delete(self._matrix, i, axis=0)

    def __len__(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        qvec = self.embedder.embed(query)
        hits = []
        for row, score in cosine_top_k(qvec, self._matrix, k):
            doc_id = self._ids[row]
            text, metadata = self.get(doc_id)  # decrypt lazily, per hit
            hits.append(SearchHit(doc_id=doc_id, score=score, text=text, metadata=metadata))
        return hits

    # ----------------------------------------------------------------- index
    def _index_add(self, doc_id: str, vec: np.ndarray) -> None:
        norm = np.linalg.norm(vec)
        v = vec / norm if norm > 0 else vec
        self._ids.append(doc_id)
        self._matrix = np.vstack([self._matrix, v[None, :]])

    def _load_index(self) -> None:
        """Decrypt all vectors into RAM once at open (text stays encrypted)."""
        ids, rows = [], []
        for doc_id, enc_vec in self.conn.execute("SELECT id, enc_vector FROM docs"):
            raw = self.cipher.decrypt(enc_vec, f"vec:{doc_id}")
            vec = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
            norm = np.linalg.norm(vec)
            ids.append(doc_id)
            rows.append(vec / norm if norm > 0 else vec)
        self._ids = ids
        self._matrix = np.vstack(rows) if rows else np.zeros((0, self.embedder.dim))

    def close(self) -> None:
        self.conn.close()
