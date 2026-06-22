from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ops_rag_agent.config import settings


@dataclass
class LongTermMemoryRecord:
    memory_id: str
    memory_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    content: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: str = "memory://local"


class LongTermMemoryStore(Protocol):
    def recall(
        self,
        user_query: str,
        *,
        user_id: Optional[str] = None,
        top_k: int = 3,
    ) -> list[LongTermMemoryRecord]:
        ...

    def write(self, records: list[LongTermMemoryRecord]) -> list[str]:
        ...


class InMemoryLongTermMemoryStore:
    def __init__(self) -> None:
        self._records: dict[str, LongTermMemoryRecord] = {}

    def recall(
        self,
        user_query: str,
        *,
        user_id: Optional[str] = None,
        top_k: int = 3,
    ) -> list[LongTermMemoryRecord]:
        terms = {token for token in user_query.lower().split() if token}
        scored: list[tuple[float, LongTermMemoryRecord]] = []
        for record in self._records.values():
            if user_id and record.user_id not in {None, user_id}:
                continue

            content_terms = set(record.content.lower().split())
            overlap = len(terms & content_terms)
            if overlap == 0:
                continue

            boosted = overlap + (0.2 if record.memory_type == "preference" else 0.0)
            scored.append((boosted, record))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)
        return [self._with_score(record, score) for score, record in ranked[:top_k]]

    def write(self, records: list[LongTermMemoryRecord]) -> list[str]:
        stored_ids: list[str] = []
        for record in records:
            self._records[record.memory_id] = record
            stored_ids.append(record.memory_id)
        return stored_ids

    @staticmethod
    def _with_score(record: LongTermMemoryRecord, score: float) -> LongTermMemoryRecord:
        return LongTermMemoryRecord(
            memory_id=record.memory_id,
            memory_type=record.memory_type,
            user_id=record.user_id,
            session_id=record.session_id,
            content=record.content,
            tags=list(record.tags),
            metadata=dict(record.metadata),
            score=score,
            source=record.source,
        )


class MilvusLongTermMemoryStore:
    """Milvus-backed memory interface placeholder."""

    def __init__(self, collection_name: Optional[str] = None) -> None:
        self.collection_name = collection_name or settings.memory_collection

    def recall(
        self,
        user_query: str,
        *,
        user_id: Optional[str] = None,
        top_k: int = 3,
    ) -> list[LongTermMemoryRecord]:
        # Placeholder: connect embedding + vector recall in Milvus here.
        del user_query, user_id, top_k
        return []

    def write(self, records: list[LongTermMemoryRecord]) -> list[str]:
        # Placeholder: connect Milvus upsert here.
        return [record.memory_id for record in records]
