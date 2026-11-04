"""Common declared-batch next-use eviction, preserving the frozen ledger.

Only the current submitted batch is visible. Real recency is restored before
commit; an ended batch leaves no rank or forecast in the shared ledger.
"""
from easygraph.gpu._arena.state_model import StateModel


class EpochModel(StateModel):
    def _planning_copy(self):
        work = super()._planning_copy()
        owner = self.epoch_owner()
        work.epoch_owner = self.epoch_owner
        if owner is not None and owner.batch_requests is not None:
            for name, component in work.components.items():
                component.last_access -= owner.next_use(name) * (1 << 40)
        return work

    def plan(self, required_specs, *args, **kwargs):
        real = {k: c.last_access for k, c in self.components.items()}
        result = super().plan(required_specs, *args, **kwargs)
        if result.feasible:
            touched = {s.key for s in required_specs}
            for name, component in result._state.components.items():
                if name not in touched and name in real:
                    component.last_access = real[name]
            result._state.epoch_owner = self.epoch_owner
        return result
