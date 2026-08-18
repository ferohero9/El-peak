"""
El-Peak — Redis Cache & Rate Limiting Layer (P2.2)
Gracefully falls back to in-memory state if Redis is not configured or unavailable.
"""
import os
import json
import time
import logging
import asyncio
import redis.asyncio as redis
from redis.asyncio.lock import Lock

logger = logging.getLogger("el-peak.cache")

REDIS_URL = os.getenv("REDIS_URL", "").strip()
_redis_pool: redis.Redis | None = None
_REDIS_ENABLED = False

# In-memory fallbacks for when Redis is not available
_mem_cache = {}
_mem_locks = {}
_mem_rate_limits = {}

async def init_redis():
    global _redis_pool, _REDIS_ENABLED
    if not REDIS_URL:
        logger.warning("REDIS_URL is not set. Using in-memory fallbacks. (Multi-instance scaling and ARQ background worker will not work).")
        return
        
    if not (REDIS_URL.startswith("redis://") or REDIS_URL.startswith("rediss://") or REDIS_URL.startswith("unix://")):
        logger.error("REDIS_URL scheme is invalid (did you paste the Postgres URL?). Using in-memory fallbacks.")
        return

    try:
        _redis_pool = redis.from_url(REDIS_URL, decode_responses=True)
        # Test connection immediately to catch auth/network errors on startup
        await _redis_pool.ping()
        _REDIS_ENABLED = True
        logger.info("Redis connected successfully.")
    except Exception as e:
        logger.error("Redis connection failed (%s). Using in-memory fallbacks.", e)
        _redis_pool = None
        _REDIS_ENABLED = False

async def close_redis():
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.close()
        _redis_pool = None

async def check_rate_limit_redis(key: str, seconds: int) -> tuple[bool, int]:
    if not _REDIS_ENABLED:
        now = time.time()
        last_req = _mem_rate_limits.get(key)
        if last_req:
            elapsed = now - last_req
            if elapsed < seconds:
                return False, int(seconds - elapsed) + 1
        _mem_rate_limits[key] = now
        return True, 0

    current = time.time()
    redis_key = f"ratelimit:{key}"
    last_req = await _redis_pool.get(redis_key)
    if last_req:
        elapsed = current - float(last_req)
        if elapsed < seconds:
            return False, int(seconds - elapsed) + 1
    await _redis_pool.set(redis_key, str(current), ex=seconds)
    return True, 0

async def get_ai_cache_redis(key: str, ttl_seconds: int) -> dict | None:
    if not _REDIS_ENABLED:
        item = _mem_cache.get(key)
        if item and item["expires"] > time.time():
            return item["data"]
        return None

    redis_key = f"aicache:{key}"
    raw = await _redis_pool.get(redis_key)
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None

async def set_ai_cache_redis(key: str, payload: dict, ttl_seconds: int):
    if not _REDIS_ENABLED:
        _mem_cache[key] = {"data": payload, "expires": time.time() + ttl_seconds}
        return

    redis_key = f"aicache:{key}"
    await _redis_pool.set(redis_key, json.dumps(payload, ensure_ascii=False), ex=ttl_seconds)

class _DummyAsyncLock:
    """Fallback lock for in-memory single-flight when Redis is disabled."""
    def __init__(self):
        self._lock = asyncio.Lock()
    async def acquire(self): return await self._lock.acquire()
    async def release(self): self._lock.release()
    async def __aenter__(self): await self.acquire(); return self
    async def __aexit__(self, *args): self.release()

def get_single_flight_lock_redis(key: str):
    if not _REDIS_ENABLED:
        if key not in _mem_locks:
            _mem_locks[key] = _DummyAsyncLock()
        return _mem_locks[key]
    
    return Lock(_redis_pool, f"lock:{key}", timeout=60.0, blocking_timeout=5.0)
