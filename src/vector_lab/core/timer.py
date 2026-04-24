import time
from contextlib import contextmanager


@contextmanager
def elapsed_timer():
    start = time.perf_counter()
    result = {"elapsed": None}
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
