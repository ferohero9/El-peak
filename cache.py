"""
El-Peak — Redis Cache & Rate Limiting Layer (P2.2)
Replaces in-process single-flight locks and DB-backed rate limits
to ensure correct behavior across multiple Railway instances.
"""
import os
import json
import time
import logging
import redis.asyncio as redis
from redis.asyncio.lock import Lock

logger = logging.getLogger("el-peak.cache")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_redis_pool: redis.Redis | None = None

async def init_redis():
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.from_url(REDIS_URL, decode_responses=True)
        logger.info("Redis connected.")

async def close_redis():
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.close()
        _redis_pool = None

def get_redis() -> redis.Redis:
    if _redis_pool is None:
        raise RuntimeError("Redis not initialized. Call init_redis() in lifespan.")
    return _redis_pool

async def check_rate_limit_redis(key: str, seconds: int) -> tuple[bool, int]:
    """
    Cross-instance interval rate limit.
    Uses a simple SET with TTL. If the key exists, the interval hasn't elapsed.
    """
    r = get_redis()
    current = time.time()
    redis_key = f"ratelimit:{key}"
    
    last_req = await r.get(redis_key)
    if last_req:
        elapsed = current - float(last_req)
        if elapsed < seconds:
            return False, int(seconds - elapsed) + 1
            
    await r.set(redis_key, str(current), ex=seconds)
    return True, 0

async def get_ai_cache_redis(key: str, ttl_seconds: int) -> dict | None:
    r = get_redis()
    redis_key = f"aicache:{key}"
    raw = await r.get(redis_key)
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None

async def set_ai_cache_redis(key: str, payload: dict, ttl_seconds: int):
    r = get_redis()
    redis_key = f"aicache:{key}"
    await r.set(redis_key, json.dumps(payload, ensure_ascii=False), ex=ttl_seconds)

def get_single_flight_lock_redis(key: str) -> Lock:
    """
    Cross-instance single-flight lock.
    timeout=60s prevents deadlocks if a worker crashes mid-Groq-call.
    blocking_timeout=5s makes concurrent requests wait briefly for the first one to finish and cache.
    """
    r = get_redis()
    return Lock(r, f"lock:{key}", timeout=60.0, blocking_timeout=5.0)
