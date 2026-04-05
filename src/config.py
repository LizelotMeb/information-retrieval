from __future__ import annotations


FAST_MODE = False

NUM_RESULTS = 200 if FAST_MODE else 1000
RERANK_K = 30 if FAST_MODE else 100
FB_TERMS = 10
FB_VALUES = [10, 30] if FAST_MODE else [5, 10, 20, 30, 50]
N_TOPICS = 10 if FAST_MODE else None

CACHE_PREFIX = "fast" if FAST_MODE else "full"
