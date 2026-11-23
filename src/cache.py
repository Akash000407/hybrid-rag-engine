import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config

logger = config.get_logger("cache")


def get_corpus_version() -> str:
    """Hash manifest.json so cache automatically invalidates when PDFs change."""
    if not config.MANIFEST_PATH.exists():
        return "empty"
    raw = config.MANIFEST_PATH.read_bytes()
    return hashlib.sha256(raw).hexdigest()[:16]


class CacheManager:
    """Persistent SQLite query cache (versioned by corpus hash) + telemetry logger."""

    def __init__(self, db_path: Path = config.CACHE_DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    corpus_version TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (corpus_version, mode, normalized_query)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    retrieval_ms REAL NOT NULL,
                    ttft_ms REAL NOT NULL,
                    total_ms REAL NOT NULL,
                    model_used TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.strip().lower().split())

    def get(self, query: str, mode: str) -> dict[str, Any] | None:
        version = get_corpus_version()
        norm_q = self._normalize(query)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT answer, sources_json, model_used
                FROM query_cache
                WHERE corpus_version = ? AND mode = ? AND normalized_query = ?
                """,
                (version, mode, norm_q),
            ).fetchone()

        if not row:
            return None

        return {
            "answer": row[0],
            "sources": json.loads(row[1]),
            "model_used": row[2],
        }

    def set(
        self,
        query: str,
        mode: str,
        answer: str,
        sources: list[dict[str, Any]],
        model_used: str,
    ) -> None:
        version = get_corpus_version()
        norm_q = self._normalize(query)
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO query_cache
                (corpus_version, mode, normalized_query, answer, sources_json, model_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (version, mode, norm_q, answer, json.dumps(sources), model_used, now),
            )

    def log_telemetry(
        self,
        query: str,
        mode: str,
        cache_hit: bool,
        retrieval_ms: float,
        ttft_ms: float,
        total_ms: float,
        model_used: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO telemetry_logs
                (timestamp, query, mode, cache_hit, retrieval_ms, ttft_ms, total_ms, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    query,
                    mode,
                    int(cache_hit),
                    round(retrieval_ms, 2),
                    round(ttft_ms, 2),
                    round(total_ms, 2),
                    model_used,
                ),
            )