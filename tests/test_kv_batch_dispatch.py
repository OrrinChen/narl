"""Check that fixed two-group updates preserve reserved version deadlines."""

from itertools import combinations_with_replacement
import multiprocessing as mp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.kv_batch_dispatch import BatchDispatch, GROUPS_PER_UPDATE, _fits


class _Weights:
    def __init__(self, ctx):
        self.version = ctx.Value("i", 0)

    def publish(self, _model):
        self.version.value += 1


def test_ready_group_waits_only_until_old_reservation_uses_its_last_slot():
    ctx = mp.get_context("spawn")
    weights = _Weights(ctx)
    dispatch = BatchDispatch(ctx, weights, total=6, window=3, max_lag=1, version_aware=True)
    old, first, second = [dispatch.acquire() for _ in range(3)]
    assert dispatch.admit(*first) == 1
    assert dispatch.admit(*second) == 2
    dispatch.complete_update(object())

    younger, parked = [dispatch.acquire() for _ in range(2)]
    assert old == (0, 0, 0) and younger[2] == parked[2] == 1
    assert dispatch.may_admit(*younger)
    assert dispatch.admit(*younger) == 1
    assert not dispatch.may_admit(*parked)
    assert dispatch.may_admit(*old)
    assert dispatch.admit(*old) == 2
    dispatch.complete_update(object())

    assert dispatch.may_admit(*parked)
    assert dispatch.admit(*parked) == 1
    last = dispatch.acquire()
    assert last[0] == 5 and last[2] == 2
    assert dispatch.admit(*last) == 2
    dispatch.complete_update(object())
    assert weights.version.value == 3
    assert dispatch.acquire() is None


def test_feasible_reservations_always_have_an_admissible_oldest_group():
    current_version = 3
    for max_lag in range(3):
        versions = range(current_version - max_lag, current_version + 1)
        for size in range(1, 8):
            for active in combinations_with_replacement(versions, size):
                for pending in range(GROUPS_PER_UPDATE):
                    if _fits(active, current_version, pending, max_lag):
                        assert _fits(active[1:], current_version, pending + 1, max_lag)
                if _fits(active, current_version, GROUPS_PER_UPDATE, max_lag):
                    assert _fits(active, current_version + 1, 0, max_lag)


def test_fifo_can_retire_a_group_without_spending_a_learner_slot():
    ctx = mp.get_context("spawn")
    weights = _Weights(ctx)
    dispatch = BatchDispatch(ctx, weights, total=4, window=1, max_lag=1, version_aware=False)
    retired = dispatch.acquire()
    dispatch.retire(*retired)
    ready = dispatch.acquire()
    assert retired == (0, 0, 0) and ready == (1, 0, 0)
    try:
        dispatch.admit(*retired)
    except RuntimeError as exc:
        assert "does not own" in str(exc)
    else:
        raise AssertionError("retired group was admitted through a reused slot")
    assert dispatch.admit(*ready) == 1
    assert dispatch.admit(*dispatch.acquire()) == 2
    dispatch.complete_update(object())
    assert dispatch.admit(*dispatch.acquire()) == 1
    assert dispatch.acquire() is None
    try:
        dispatch.complete_update(object())
    except RuntimeError as exc:
        assert "complete learner batch" in str(exc)
    else:
        raise AssertionError("a one-group tail published weights")
    assert weights.version.value == 1


def _worker(dispatch, results):
    while (reservation := dispatch.acquire()) is not None:
        results.put(reservation)
    results.put(None)


def test_spawned_actors_fill_complete_updates_with_one_producer_slot():
    ctx = mp.get_context("spawn")
    for version_aware in (False, True):
        weights = _Weights(ctx)
        dispatch = BatchDispatch(ctx, weights, total=4, window=1, max_lag=1, version_aware=version_aware)
        results = ctx.Queue()
        actors = [ctx.Process(target=_worker, args=(dispatch, results)) for _ in range(2)]
        for actor in actors:
            actor.start()
        try:
            group_count = finished = 0
            while finished < len(actors):
                reservation = results.get(timeout=5)
                if reservation is None:
                    finished += 1
                    continue
                assert reservation[0] == group_count
                group_count += 1
                assert dispatch.may_admit(*reservation)
                if dispatch.admit(*reservation) == GROUPS_PER_UPDATE:
                    dispatch.complete_update(object())
            assert group_count == 4
            assert weights.version.value == 2
            for actor in actors:
                actor.join(timeout=5)
                assert actor.exitcode == 0
        finally:
            for actor in actors:
                if actor.is_alive():
                    actor.terminate()
                    actor.join(timeout=5)
                actor.close()
            results.close()
            results.join_thread()


if __name__ == "__main__":
    test_ready_group_waits_only_until_old_reservation_uses_its_last_slot()
    test_feasible_reservations_always_have_an_admissible_oldest_group()
    test_fifo_can_retire_a_group_without_spending_a_learner_slot()
    test_spawned_actors_fill_complete_updates_with_one_producer_slot()
    print("ok  batch admission, rollout deadlines, and spawned actor liveness")
