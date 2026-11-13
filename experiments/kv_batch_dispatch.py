"""Reserve learner slots for complete rollout groups before their version expires."""

GROUPS_PER_UPDATE = 2


def _fits(versions, current_version, pending, max_lag):
    # For each deadline, count all groups due by then against the remaining
    # learner slots, including the last version on which they are still valid.
    for due_count, version in enumerate(sorted(versions), start=1):
        deadline = version + max_lag
        slots = GROUPS_PER_UPDATE * (deadline - current_version + 1) - pending
        if due_count > slots:
            return False
    return True


class BatchDispatch:
    """Cap generating, queued, and parked groups at window, plus two learner slots."""

    def __init__(self, ctx, weights, total, window, max_lag, version_aware):
        if total < 0 or total % GROUPS_PER_UPDATE or window < 1 or max_lag < 0:
            raise ValueError("expected an even group total, positive window, and nonnegative max lag")
        self.condition = ctx.Condition()
        self.weights = weights
        self.total = total
        self.max_lag = max_lag
        self.version_aware = version_aware
        self._next_id = ctx.RawValue("i", 0)
        self._pending = ctx.RawValue("i", 0)
        self._group_ids = ctx.RawArray("i", [-1] * window)
        self._versions = ctx.RawArray("i", [-1] * window)

    def _active_versions(self, exclude_slot=None):
        return [version for slot, version in enumerate(self._versions)
                if version >= 0 and slot != exclude_slot]

    def acquire(self):
        with self.condition:
            while self._next_id.value < self.total:
                if -1 in self._group_ids:
                    version = self.weights.version.value
                    if not self.version_aware or _fits(
                        [*self._active_versions(), version], version, self._pending.value, self.max_lag,
                    ):
                        slot = list(self._group_ids).index(-1)
                        group_id = self._next_id.value
                        self._next_id.value += 1
                        self._group_ids[slot] = group_id
                        self._versions[slot] = version
                        if self._next_id.value == self.total:
                            self.condition.notify_all()
                        return group_id, slot, version
                self.condition.wait()
            return None

    def _check_reservation(self, group_id, slot, version):
        if (group_id < 0 or version < 0 or not 0 <= slot < len(self._group_ids)
                or self._group_ids[slot] != group_id or self._versions[slot] != version):
            raise RuntimeError("rollout returned a reservation it does not own")

    def _may_admit(self, slot):
        pending = self._pending.value
        return pending < GROUPS_PER_UPDATE and (
            not self.version_aware or _fits(
                self._active_versions(exclude_slot=slot), self.weights.version.value, pending + 1, self.max_lag,
            )
        )

    def may_admit(self, group_id, slot, reserved_version):
        with self.condition:
            self._check_reservation(group_id, slot, reserved_version)
            return self._may_admit(slot)

    def _release(self, slot):
        self._group_ids[slot] = -1
        self._versions[slot] = -1
        self.condition.notify_all()

    def admit(self, group_id, slot, reserved_version):
        """Transfer a ready group from its producer reservation to the current batch."""
        with self.condition:
            self._check_reservation(group_id, slot, reserved_version)
            if not self._may_admit(slot):
                raise RuntimeError("admission would exceed current batch or rollout deadlines")
            self._pending.value += 1
            self._release(slot)
            return self._pending.value

    def retire(self, group_id, slot, reserved_version):
        with self.condition:
            self._check_reservation(group_id, slot, reserved_version)
            self._release(slot)

    def complete_update(self, model):
        with self.condition:
            if self._pending.value != GROUPS_PER_UPDATE:
                raise RuntimeError("cannot publish without a complete learner batch")
            self.weights.publish(model)
            self._pending.value = 0
            self.condition.notify_all()
