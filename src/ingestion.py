import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import chromadb
import pdfplumber

from src import config

logger = config.get_logger("ingestion")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 fingerprint of a file to detect additions or edits."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def sanitize_id(name: str) -> str:
    """Convert filename into a clean, ChromaDB-safe ID prefix."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def split_markdown_blocks(md_text: str) -> list[tuple[str, bool]]:
    """
    Split a Markdown page into logical blocks while keeping tables atomic.
    Returns a list of (block_text, is_table).
    """
    lines = md_text.splitlines()
    blocks: list[tuple[str, bool]] = []
    buffer: list[str] = []
    in_table = False

    def flush():
        nonlocal buffer, in_table
        text = "\n".join(buffer).strip()
        if text:
            blocks.append((text, in_table))
        buffer = []

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")

        if is_table_line:
            if not in_table:
                flush()
                in_table = True
            buffer.append(line)
        else:
            if in_table:
                flush()
                in_table = False

            # Start a new block on blank lines or Markdown headings
            if not stripped:
                flush()
            elif stripped.startswith("#"):
                flush()
                buffer.append(line)
            else:
                buffer.append(line)

    flush()
    return blocks


def split_large_table(table_text: str, max_chars: int) -> list[str]:
    """If a single table is huge, split by rows while repeating its header."""
    rows = [r for r in table_text.splitlines() if r.strip()]
    if len(rows) <= 3:
        return [table_text]

    header = "\n".join(rows[:2])  # Column names + |---|---| separator
    data_rows = rows[2:]
    pieces: list[str] = []
    current = header

    for row in data_rows:
        if len(current) + len(row) + 1 > max_chars and current != header:
            pieces.append(current)
            current = f"{header}\n{row}"
        else:
            current = f"{current}\n{row}"

    if current != header:
        pieces.append(current)
    return pieces


def chunk_markdown_page(
    md_text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[tuple[str, bool]]:
    """
    Pack Markdown blocks into table-safe chunks.
    Returns a list of (chunk_text, has_table).
    """
    blocks = split_markdown_blocks(md_text)
    chunks: list[tuple[str, bool]] = []
    current_parts: list[str] = []
    current_len = 0
    current_has_table = False

    for block_text, is_table in blocks:
        # Handle oversized tables cleanly without losing column headers
        if is_table and len(block_text) > int(chunk_size * 1.8):
            if current_parts:
                chunks.append(("\n\n".join(current_parts), current_has_table))
                current_parts, current_len, current_has_table = [], 0, False
            for sub_table in split_large_table(block_text, int(chunk_size * 1.5)):
                chunks.append((sub_table, True))
            continue

        # Flush current chunk if adding this block would exceed target size
        if current_parts and (current_len + len(block_text) + 2 > chunk_size):
            flushed = "\n\n".join(current_parts)
            chunks.append((flushed, current_has_table))

            # Carry clean tail overlap for plain prose (skip repeating full tables)
            tail = "" if current_has_table else flushed[-overlap:].strip()
            current_parts = [tail] if tail else []
            current_len = len(tail)
            current_has_table = False

        current_parts.append(block_text)
        current_len += len(block_text) + 2
        if is_table:
            current_has_table = True

    if current_parts:
        chunks.append(("\n\n".join(current_parts), current_has_table))

    return [
        (text.strip(), has_tbl)
        for text, has_tbl in chunks
        if len(text.strip()) >= config.MIN_CHUNK_CHARS
    ]


def table_to_markdown(rows: list[list[Any]]) -> str:
    """Convert a pdfplumber table (list of rows) into a clean Markdown table."""
    cleaned_rows: list[list[str]] = []
    for row in rows:
        if not row:
            continue
        cells = [
            re.sub(r"\s+", " ", str(cell or "")).strip().replace("|", "\\|")
            for cell in row
        ]
        if any(cells):
            cleaned_rows.append(cells)

    if len(cleaned_rows) < 2:
        return ""

    col_count = max(len(r) for r in cleaned_rows)
    padded = [r + [""] * (col_count - len(r)) for r in cleaned_rows]

    header = "| " + " | ".join(padded[0]) + " |"
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    body = ["| " + " | ".join(r) + " |" for r in padded[1:]]
    return "\n".join([header, separator, *body])


def parse_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Parse a single PDF into layout-aware Markdown chunks with pure-Python pdfplumber."""
    logger.info("Parsing PDF into Markdown: %s", pdf_path.name)
    file_prefix = sanitize_id(pdf_path.stem)
    parsed_chunks: list[dict[str, Any]] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            # Extract prose outside table bounding boxes to avoid duplicate text
            if table_bboxes:
                def outside_tables(obj: dict[str, Any]) -> bool:
                    if "x0" not in obj or "top" not in obj:
                        return True
                    x_mid = (obj["x0"] + obj["x1"]) / 2
                    y_mid = (obj["top"] + obj["bottom"]) / 2
                    return not any(
                        b[0] <= x_mid <= b[2] and b[1] <= y_mid <= b[3]
                        for b in table_bboxes
                    )

                prose_text = page.filter(outside_tables).extract_text() or ""
            else:
                prose_text = page.extract_text() or ""

            md_parts = [prose_text.strip()] if prose_text.strip() else []
            for table in tables:
                md_table = table_to_markdown(table.extract())
                if md_table:
                    md_parts.append(md_table)

            md_text = "\n\n".join(md_parts).strip()
            if not md_text:
                continue

            page_chunks = chunk_markdown_page(md_text)
            for chunk_idx, (chunk_text, has_table) in enumerate(page_chunks):
                parsed_chunks.append(
                    {
                        "id": f"{file_prefix}_p{page_num}_c{chunk_idx}",
                        "text": chunk_text,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_num,
                            "has_table": has_table,
                        },
                    }
                )

    return parsed_chunks


def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sync_documents(data_dir: Path = config.DATA_DIR) -> dict[str, int]:
    """
    Incrementally sync all PDFs in data_dir into ChromaDB and chunks.json.
    - New or modified PDFs (SHA-256 changed) are parsed and indexed.
    - Unchanged PDFs are skipped in milliseconds.
    - Deleted PDFs are cleanly purged from both ChromaDB and BM25 chunk store.
    """
    start_time = time.time()
    manifest: dict[str, dict[str, Any]] = _load_json(config.MANIFEST_PATH, {})
    chunks_store: dict[str, dict[str, Any]] = _load_json(config.CHUNKS_PATH, {})

    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)

    pdf_files = sorted(data_dir.glob("*.pdf"))
    current_filenames = {p.name for p in pdf_files}

    stats = {"indexed": 0, "skipped": 0, "deleted": 0, "total_chunks": 0}

    # 1. Purge PDFs that were removed from data/
    for old_name in list(manifest.keys()):
        if old_name not in current_filenames:
            old_ids = manifest[old_name].get("chunk_ids", [])
            if old_ids:
                collection.delete(ids=old_ids)
                for cid in old_ids:
                    chunks_store.pop(cid, None)
            manifest.pop(old_name, None)
            stats["deleted"] += 1
            logger.info("Removed deleted file from index: %s", old_name)

    # 2. Index new or modified PDFs
    for pdf_path in pdf_files:
        file_hash = compute_sha256(pdf_path)
        prev_entry = manifest.get(pdf_path.name)

        if prev_entry and prev_entry.get("sha256") == file_hash:
            stats["skipped"] += 1
            continue

        # If file was modified, remove its stale chunks before re-indexing
        if prev_entry and prev_entry.get("chunk_ids"):
            old_ids = prev_entry["chunk_ids"]
            collection.delete(ids=old_ids)
            for cid in old_ids:
                chunks_store.pop(cid, None)
            logger.info("Detected edit in %s; replacing old chunks...", pdf_path.name)

        try:
            new_chunks = parse_pdf(pdf_path)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", pdf_path.name, exc)
            continue

        if not new_chunks:
            logger.warning(
                "No extractable text found in %s (may be a scanned image PDF).",
                pdf_path.name,
            )
            continue

        # Add to ChromaDB in batches of 100
        batch_size = 100
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )

        # Update BM25 single-source-of-truth store and manifest
        for c in new_chunks:
            chunks_store[c["id"]] = c

        manifest[pdf_path.name] = {
            "sha256": file_hash,
            "chunk_ids": [c["id"] for c in new_chunks],
            "num_chunks": len(new_chunks),
        }
        stats["indexed"] += 1
        logger.info("Indexed %s (%d chunks)", pdf_path.name, len(new_chunks))

    _save_json(config.MANIFEST_PATH, manifest)
    _save_json(config.CHUNKS_PATH, chunks_store)

    stats["total_chunks"] = len(chunks_store)
    elapsed = time.time() - start_time
    logger.info(
        "Sync complete in %.2fs | indexed=%d, skipped=%d, deleted=%d, total_chunks=%d",
        elapsed,
        stats["indexed"],
        stats["skipped"],
        stats["deleted"],
        stats["total_chunks"],
    )
    return stats


if __name__ == "__main__":
    summary = sync_documents()
    chunks_store = _load_json(config.CHUNKS_PATH, {})
    table_chunks = [c for c in chunks_store.values() if c["metadata"].get("has_table")]
    print(f"\nDetected {len(table_chunks)} table-bearing chunks across your PDFs.")
    if table_chunks:
        sample = table_chunks[0]
        print(
            f"Sample table chunk ({sample['metadata']['source']}, Page {sample['metadata']['page']}):\n"
            f"{sample['text'][:350]}..."
        )