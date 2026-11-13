"""Exercise the dispatch guard when an older group is still in flight."""
import multiprocessing as mp
from pathlib import Path
import sys
from threading import Event, Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.kv_dispatch import _Dispatch


class _Versions:
    def __init__(self, ctx):
        self.version = ctx.Value("i", 0)

    def publish(self, _model):
        self.version.value += 1


def test_oldest_reservation_retains_its_last_version_credit():
    ctx = mp.get_context("spawn")
    versions = _Versions(ctx)
    dispatch = _Dispatch(ctx, versions, total=3, window=2, max_lag=1, version_aware=True)
    assert dispatch.acquire() == (0, 0, 0)
    assert dispatch.acquire() == (1, 1, 0)
    dispatch.finish(1, 0, object())
    blocked, completed = Event(), Event()
    reservations = []
    wait = dispatch.condition.wait

    def observe_wait():
        blocked.set()
        return wait()

    dispatch.condition.wait = observe_wait

    def acquire():
        reservations.append(dispatch.acquire())
        completed.set()

    actor = Thread(target=acquire, daemon=True)
    actor.start()
    assert blocked.wait(timeout=2)
    assert not completed.is_set()
    dispatch.finish(0, 0, object())
    assert completed.wait(timeout=2)
    actor.join()
    assert reservations == [(2, 0, 2)]
    assert dispatch.acquire() is None


if __name__ == "__main__":
    test_oldest_reservation_retains_its_last_version_credit()
    print("ok  an old in-flight group keeps its remaining lag credit")
