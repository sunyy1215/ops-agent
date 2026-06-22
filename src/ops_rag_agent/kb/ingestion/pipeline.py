from __future__ import annotations

from pathlib import Path
import uuid
from typing import Any

from ops_rag_agent.config import settings


def ingest_file(
    file_path: str,
    *,
    collection_name: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    docs = [{"id": path.stem or f"doc-{uuid.uuid4().hex[:8]}", "text": text, "metadata": {"source": str(path)}}]
    return ingest_inline_documents(docs, collection_name=collection_name, dry_run=dry_run)


def ingest_directory(
    directory_path: str,
    *,
    collection_name: str | None = None,
    dry_run: bool = False,
    glob: str = "**/*",
) -> dict[str, Any]:
    directory = Path(directory_path)
    docs: list[dict[str, Any]] = []
    for path in directory.glob(glob):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(
            {
                "id": path.stem or f"doc-{uuid.uuid4().hex[:8]}",
                "text": text,
                "metadata": {"source": str(path)},
            }
        )
    return ingest_inline_documents(docs, collection_name=collection_name, dry_run=dry_run)


def ingest_inline_documents(
    docs: list[dict[str, Any]],
    *,
    collection_name: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    normalized_docs = [
        {
            "id": str(item.get("id") or f"doc-{uuid.uuid4().hex[:8]}"),
            "text": str(item.get("text") or ""),
            "metadata": dict(item.get("metadata") or {}),
        }
        for item in docs
    ]
    chunk_count = sum(1 for item in normalized_docs if item["text"].strip())
    return {
        "run_id": f"ingest-{uuid.uuid4().hex[:8]}",
        "dry_run": dry_run,
        "collection": collection_name or settings.milvus_collection,
        "docs": normalized_docs,
        "stats": {
            "documents_total": len(normalized_docs),
            "chunks_upserted": chunk_count,
        },
        "duplicate_relations": [],
    }
