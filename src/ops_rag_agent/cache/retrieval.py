from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol


class RetrievalCache(Protocol):
    def get(self, key: str) -> Optional[list[dict[str, Any]]]:
        ...

    def set(self, key: str, value: list[dict[str, Any]], ttl_seconds: int) -> None:
        ...


@dataclass(frozen=True)
class CacheNamespace:
    layer: str
    name: str

    def prefix(self) -> str:
        return f"{self.layer}:{self.name}"


@dataclass(frozen=True)
class CacheEnvelope:
    key: str
    namespace: str
    ttl_seconds: int


class CacheKeyBuilder:
    """Builds stable retrieval cache keys for memory/redis backends."""

    @staticmethod
    def build(namespace: CacheNamespace, payload: dict[str, Any]) -> CacheEnvelope:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return CacheEnvelope(
            key=f"{namespace.prefix()}:{digest}",
            namespace=namespace.prefix(),
            ttl_seconds=int(payload.get("ttl_seconds", 120)),
        )


class InMemoryRetrievalCache:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def get(self, key: str) -> Optional[list[dict[str, Any]]]:
        item = self._entries.get(key)
        if item is None:
            return None

        expires_at, value = item
        if expires_at < time.time():
            self._entries.pop(key, None)
            return None
        return value

    def set(self, key: str, value: list[dict[str, Any]], ttl_seconds: int) -> None:
        self._entries[key] = (time.time() + ttl_seconds, value)


class RedisRetrievalCache:
    """Optional Redis cache adapter.

    This is intentionally soft-wired: when the redis dependency or URL is absent,
    the cache behaves as a disabled adapter and simply returns cache misses.
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self.redis_url = redis_url

    def get(self, key: str) -> Optional[list[dict[str, Any]]]:
        client = self._build_client()
        if client is None:
            return None

        payload = client.get(key)
        if not payload:
            return None
        return json.loads(payload)

    def set(self, key: str, value: list[dict[str, Any]], ttl_seconds: int) -> None:
        client = self._build_client()
        if client is None:
            return
        client.set(name=key, value=json.dumps(value, ensure_ascii=True), ex=ttl_seconds)

    def _build_client(self) -> Any:
        if not self.redis_url:
            return None
        try:
            import redis
        except ImportError:
            return None

        try:
            return redis.from_url(self.redis_url, decode_responses=True)
        except Exception:
            return None


class CompositeRetrievalCache:
    """Two-level cache wrapper: memory first, redis second."""

    def __init__(
        self,
        memory_cache: Optional[RetrievalCache] = None,
        redis_cache: Optional[RetrievalCache] = None,
    ) -> None:
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache

    def get(self, key: str) -> Optional[list[dict[str, Any]]]:
        if self.memory_cache is not None:
            value = self.memory_cache.get(key)
            if value is not None:
                return value

        if self.redis_cache is None:
            return None

        value = self.redis_cache.get(key)
        if value is not None and self.memory_cache is not None:
            self.memory_cache.set(key, value, ttl_seconds=30)
        return value

    def set(self, key: str, value: list[dict[str, Any]], ttl_seconds: int) -> None:
        if self.memory_cache is not None:
            self.memory_cache.set(key, value, ttl_seconds=ttl_seconds)
        if self.redis_cache is not None:
            self.redis_cache.set(key, value, ttl_seconds=ttl_seconds)
