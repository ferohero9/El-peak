"""
El-Peak — cache.py (P2 Redis layer)
Implements Redis-backed rate limiting, AI response caching, and
cross-instance single-flight locking used by main.py.
"""
import os
import json
import time
import logging
from typing import Optional, Any

import redis.asyncio as redis

logger = logging.getLogger("el-peak.cache")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

REDIS_CLIENT: Optional["redis.Redis"] = None


async def init_redis():
    global REDIS_CLIENT
    if REDIS_CLIENT is None:
        REDIS_CLIENT = redis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        try:
            await REDIS_CLIENT.ping()
            logger.info("Redis connected.")
        except Exception as e:
            logger.error("Redis connection failed: %s", e)
            # Keep REDIS_CLIENT set; individual calls will fail gracefully below.


async def close_redis():
    global REDIS_CLIENT
    if REDIS_CLIENT is not None:
        try:
            await REDIS_CLIENT.close()
        except Exception:
            pass
        REDIS_CLIENT = None


def _client() -> "redis.Redis":
    if REDIS_CLIENT is None:
        raise RuntimeError("Redis client not initialized. Call init_redis() in lifespan.")
    return REDIS_CLIENT


# -----------------------------
# Rate limiting (P0.5 / P2.2)
# -----------------------------
async def check_rate_limit_redis(key: str, seconds: int) -> tuple[bool, int]:
    """
    Returns (allowed, wait_seconds). Uses SET NX with expiry as an atomic
    per-key cooldown — first call in the window sets the key and succeeds;
    subsequent calls within `seconds` fail until it expires.
    Fails OPEN (allowed=True) if Redis is unreachable, so a Redis outage
    does not take down auth/AI endpoints entirely.
    """
    try:
        client = _client()
        rl_key = f"ratelimit:{key}"
        was_set = await client.set(rl_key, "1", nx=True, ex=seconds)
        if was_set:
            return True, 0
        ttl = await client.ttl(rl_key)
        return False, max(1, ttl if ttl and ttl > 0 else seconds)
    except Exception as e:
        logger.warning("Rate limit check failed (failing open): %s", e)
        return True, 0


# -----------------------------
# AI response cache (P1.1 / P2.2)
# -----------------------------
async def get_ai_cache_redis(cache_key: str, ttl: int) -> Optional[dict]:
    try:
        client = _client()
        raw = await client.get(f"aicache:{cache_key}")
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning("AI cache read failed: %s", e)
        return None


async def set_ai_cache_redis(cache_key: str, payload: Any, ttl: int) -> None:
    try:
        client = _client()
        await client.set(f"aicache:{cache_key}", json.dumps(payload, ensure_ascii=False), ex=ttl)
    except Exception as e:
        logger.warning("AI cache write failed: %s", e)


# -----------------------------
# Single-flight lock (P1.1 / P2.2)
# -----------------------------
class RedisSingleFlightLock:
    """
    Minimal distributed lock: SET NX with a short TTL as a safety net so a
    crashed holder can't wedge the key forever. Not a full Redlock — fine
    for "avoid duplicate Groq calls," not for correctness-critical locking.
    """
    def __init__(self, key: str, ttl: int = 15):
        self._key = f"lock:{key}"
        self._ttl = ttl
        self._token = str(time.time())

    async def acquire(self) -> bool:
        try:
            client = _client()
            return bool(await client.set(self._key, self._token, nx=True, ex=self._ttl))
        except Exception as e:
            logger.warning("Lock acquire failed: %s", e)
            return False

    async def release(self) -> None:
        try:
            client = _client()
            # Only delete if we still hold it (best-effort, not atomic CAS).
            current = await client.get(self._key)
            if current == self._token:
                await client.delete(self._key)
        except Exception as e:
            logger.warning("Lock release failed: %s", e)


def get_single_flight_lock_redis(cache_key: str) -> RedisSingleFlightLock:
    return RedisSingleFlightLock(cache_key)
