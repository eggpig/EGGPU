"""Additional, honest controls over the existing native ownership ABI.

Importing this file does not import CUDA or start a runtime.  All physical
releases use the existing exact receipt/inventory contract.  View-LRU ranks
individual views; shared bytes remain live until the last legal owner leaves.
"""
from time import perf_counter


def release_legal(inventory, evict, devices, *, require_empty=False):
    from easygraph.gpu._controller import (
        validate_native_row, release_closure, predict_release, _request_roots)
    from easygraph.gpu._receipts import merge_release_receipt, _state_signature
    before = inventory()
    roots, predictions = [], {}
    for row in before['per_device']:
        owners = validate_native_row(row)
        legal = set()
        for oid, owner in owners.items():
            if owner['kind'] == 'catalog':
                continue
            closure = release_closure(row, {oid})
            if closure is not None:
                legal.update(closure)
        roots.extend(_request_roots(row, legal))
        predictions[row['device_id']] = predict_release(row, legal)
    receipt = evict(roots, devices)
    after = inventory()
    merged = merge_release_receipt(before, receipt, devices, devices)
    if _state_signature(after) != _state_signature(merged):
        raise RuntimeError('Release receipt differs from fresh full inventory')
    prior = {r['device_id']: r for r in before['per_device']}
    actual_total = 0
    for row in after['per_device']:
        validate_native_row(row)
        old = prior[row['device_id']]
        prediction = predictions[row['device_id']]
        actual = old['managed_resident_bytes'] - row['managed_resident_bytes']
        expected_removed = set(prediction['predicted_allocation_ids'])
        removed = {a['allocation_id'] for a in old['allocations']} - {
            a['allocation_id'] for a in row['allocations']}
        old_owners = {o['owner_id'] for o in old['owners'] if o['kind'] != 'catalog'}
        new_owners = {o['owner_id'] for o in row['owners'] if o['kind'] != 'catalog'}
        if (actual != prediction['predicted_freed_bytes'] or removed != expected_removed
                or old_owners - new_owners != set(prediction['closure_owner_ids'])):
            raise RuntimeError('Exact legal release prediction failed')
        if require_empty and row['managed_resident_bytes'] != 0:
            raise RuntimeError('No-retention boundary still has physical storage')
        actual_total += actual
    if receipt['released_bytes'] != actual_total:
        raise RuntimeError('Aggregate release byte receipt differs')
    return dict(before_retention=before, after_retention=after, receipt=receipt,
        owner_ids=roots, devices=list(devices), predicted_freed_bytes=actual_total,
        freed_bytes=actual_total, prediction_error_bytes=0,
        capacity_progress=actual_total > 0, zero_release_is_capacity_progress=False,
        protected_owner_ids=[o['owner_id'] for r in after['per_device'] for o in r['owners']
                             if o['kind'] != 'catalog'])


def install_controller(session, policy):
    """Install only added policies; leave the original two strong arms intact."""
    if policy not in ('no_retention', 'view_lru'):
        return
    from easygraph.gpu._controller import StateSwitchController, _BoundaryModel, _request_roots

    class ViewLRU(StateSwitchController):
        def candidates(self, row, deficit):
            model = _BoundaryModel(row)
            rows = []
            for oid, owner in model.owners.items():
                if owner['kind'] == 'catalog':
                    continue
                closure = model.closure({oid})
                if closure is None:
                    continue
                prediction = model.prediction(closure)
                # Zero-byte release is allowed to remove one sharing holder,
                # but capacity is checked only from the next physical inventory.
                rows.append(dict(owner_ids=_request_roots(row, closure), **prediction,
                    score=self.last_use.get(oid, 0), last_use=self.last_use.get(oid, 0),
                    score_kind='individual_view_recency_then_workspace_fallback',
                    owner_kind=owner['kind'], remaining_deficit_bytes=deficit,
                    owner_referenced_bytes_sum=sum(model.referenced[i] for i in closure),
                    useful_freed_bytes=min(deficit, prediction['predicted_freed_bytes'])))
            return sorted(rows, key=lambda r: (r['owner_kind'] != 'view', r['score'], tuple(r['owner_ids'])))

    class NoRetention(StateSwitchController):
        def finish_query(self, graph_id, algorithm):
            begin = perf_counter()
            budget = self.budgets
            self.budgets = None
            try:
                result = super().finish_query(graph_id, algorithm)
            finally:
                self.budgets = budget
            action = release_legal(self.inventory, self.evict, self.devices, require_empty=True)
            self._last_live = set()
            self.last_use.clear()
            result.update(after_retention=action['after_retention'],
                actions=[dict(action, reason='no_retention')],
                controller_seconds=perf_counter() - begin, budgets=budget,
                state_switch_policy='no_retention', retention_policy='no_retention',
                effective_retained_bytes_target=0,
                boundary='After complete owning host output; all legal managed owners and idle catalog released')
            return result

    backend = session._backend
    prior = backend.controller
    cls = NoRetention if policy == 'no_retention' else ViewLRU
    replacement = cls(prior.cpp, prior.devices, prior.budgets, policy='closure',
        inventory_mode=prior.inventory_mode, batch_devices=prior.batch_devices,
        graph_bindings=prior.graph_bindings)
    replacement.policy = policy
    backend.controller = replacement


def retire_for_rebuild(session):
    """Keep actual leased old views alive, release every other legal owner."""
    controller = session._backend.controller
    return release_legal(controller.inventory, controller.evict, controller.devices)
