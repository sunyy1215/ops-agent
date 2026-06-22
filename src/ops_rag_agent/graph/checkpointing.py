from __future__ import annotations

import pickle
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from ops_rag_agent.config import settings


def _nested_defaultdict() -> defaultdict[str, dict[str, Any]]:
    return defaultdict(dict)


class PersistentCheckpointSaver(InMemorySaver):
    """Persist LangGraph checkpoints to disk while reusing InMemorySaver semantics."""

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        self._lock = RLock()
        super().__init__()
        self._load_from_disk()

    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        result = super().put(config, checkpoint, metadata, new_versions)
        self._persist_to_disk()
        return result

    def put_writes(
        self,
        config: dict[str, Any],
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path)
        self._persist_to_disk()

    def _load_from_disk(self) -> None:
        if not self.file_path.exists():
            return
        with self._lock:
            with self.file_path.open("rb") as fh:
                payload = pickle.load(fh)

            storage = defaultdict(_nested_defaultdict)
            for thread_id, namespaces in payload.get("storage", {}).items():
                storage[thread_id] = defaultdict(dict, namespaces)

            self.storage = storage
            self.writes = defaultdict(dict, payload.get("writes", {}))
            self.blobs = defaultdict(bytes, payload.get("blobs", {}))

    def _persist_to_disk(self) -> None:
        payload = {
            "storage": {thread_id: dict(namespaces) for thread_id, namespaces in self.storage.items()},
            "writes": dict(self.writes),
            "blobs": dict(self.blobs),
        }
        tmp_path = self.file_path.with_suffix(self.file_path.suffix + ".tmp")

        with self._lock:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("wb") as fh:
                pickle.dump(payload, fh)
            tmp_path.replace(self.file_path)


@lru_cache(maxsize=1)
def build_checkpointer() -> PersistentCheckpointSaver:
    return PersistentCheckpointSaver(settings.langgraph_checkpoint_path)
