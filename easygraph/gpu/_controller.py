"""History-only joint view/workspace retention; no CUDA or future trace access.

The demand score is dimensionless, not a prediction of measured milliseconds.
Native receipts remain authoritative; any mismatch with allocation-closure
predictions aborts. Every policy exports a fresh full inventory per query.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

from easygraph.gpu._retention import algorithm_name
from easygraph.gpu._receipts import (
    ManagedNativeSession,
    ReceiptRetentionController,
    _state_signature,
    merge_release_receipt,
)

HISTORY_WINDOW = 32
GHOST_CAPACITY = 256
CURRENT_MULTIPLIER = 2.0
POLICIES = ("lru", "scratch_first", "gdsf", "closure", "closure_refault", "closure_lru")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_native_row(row):
    """Require the complete frozen ABI, including exact shared-ref accounting."""
    owners = {o["owner_id"]: o for o in row["owners"]}
    if len(owners) != len(row["owners"]):
        raise RuntimeError("Duplicate native owner")
    for owner in owners.values():
        if owner["kind"] not in ("view", "workspace", "catalog"):
            raise RuntimeError("Unsupported owner kind in closure contract")
        if int(owner["device_id"]) != int(row["device_id"]):
            raise RuntimeError("Owner is on a different device")
        if int(owner["active_leases"]) < 0:
            raise RuntimeError("Invalid native lease count")
        parent = owner.get("parent_owner_id", "")
        if parent and (parent not in owners or owners[parent]["kind"] != "view"
                       or owner["kind"] != "workspace"):
            raise RuntimeError("Unresolved per-view workspace parent")
    for allocation in row["allocations"]:
        refs = allocation["owners"]
        if len(refs) != len(set(refs)) or not refs or not set(refs) <= set(owners):
            raise RuntimeError("Invalid allocation holder set")
        required = ("catalog_retained", "catalog_external_references",
                    "inventoried_shared_references", "untracked_shared_references",
                    "catalog_only_idle")
        if any(k not in allocation for k in required):
            raise RuntimeError("Unknown native shared-reference accounting")
        ext = int(allocation["catalog_external_references"])
        inventoried = int(allocation["inventoried_shared_references"])
        untracked = int(allocation["untracked_shared_references"])
        if min(ext, inventoried, untracked) < 0 or untracked:
            raise RuntimeError("Untracked or invalid allocation references")
        catalogs = {i for i in refs if owners[i]["kind"] == "catalog"}
        catalogued = bool(allocation["catalog_retained"])
        if catalogued:
            if len(catalogs) != 1 or ext != inventoried:
                raise RuntimeError("Catalog references do not match inventoried references")
            if bool(allocation["catalog_only_idle"]) != (ext == 0):
                raise RuntimeError("Catalog idle flag contradicts native reference count")
            noncatalog = set(refs) - catalogs
            if (not noncatalog) != (ext == 0):
                raise RuntimeError("Catalog holders contradict external reference count")
        elif catalogs or allocation["catalog_only_idle"] or ext:
            raise RuntimeError("Noncatalog allocation has catalog references")
    return owners


def release_closure(row, requested):
    """Exact owner cascade of the frozen BFS/PR registries, or None if leased."""
    owners = validate_native_row(row)
    requested = set(requested)
    if not requested <= set(owners):
        raise RuntimeError("Requested closure contains unknown owners")
    if any(owners[i]["kind"] == "catalog" for i in requested):
        raise RuntimeError("Catalog lifetime is allocation-specific, not an owner action")
    closure = set(requested)
    for owner in owners.values():
        if owner.get("parent_owner_id") in requested:
            closure.add(owner["owner_id"])
    if any(owners[i]["active_leases"] for i in closure):
        return None
    return frozenset(closure)


def predict_release(row, requested, *, release_idle=True):
    """Return exact predicted freed allocations under the complete-boundary ABI."""
    owners = validate_native_row(row)
    closure = release_closure(row, requested)
    if closure is None:
        return None
    freed = []
    for allocation in row["allocations"]:
        holders = {i for i in allocation["owners"] if owners[i]["kind"] != "catalog"}
        if holders <= closure and (not allocation["catalog_retained"] or release_idle):
            freed.append(allocation)
    return {
        "closure_owner_ids": sorted(closure),
        "predicted_freed_bytes": sum(int(a["bytes"]) for a in freed),
        "predicted_allocation_ids": sorted(a["allocation_id"] for a in freed),
    }


def _request_roots(row, closure):
    """A parent view already requests its per-view workspace's deletion."""
    owners = {o["owner_id"]: o for o in row["owners"]}
    return sorted(i for i in closure if owners[i].get("parent_owner_id") not in closure)


class _BoundaryModel:
    """One validated incidence index, discarded after the boundary changes."""

    def __init__(self, row):
        self.row = row
        self.owners = validate_native_row(row)
        self.children = {}
        self.referenced = Counter()
        self.allocations = []
        for oid, owner in self.owners.items():
            parent = owner.get("parent_owner_id")
            if parent:
                self.children.setdefault(parent, set()).add(oid)
        for allocation in row["allocations"]:
            holders = frozenset(i for i in allocation["owners"]
                                if self.owners[i]["kind"] != "catalog")
            self.allocations.append((holders, allocation))
            for oid in holders:
                self.referenced[oid] += int(allocation["bytes"])

    def closure(self, requested):
        requested = frozenset(requested)
        if not requested <= set(self.owners):
            raise RuntimeError("Unknown candidate owner")
        if any(self.owners[i]["kind"] == "catalog" for i in requested):
            raise RuntimeError("Catalog owner cannot be requested")
        closure = set(requested)
        for oid in requested:
            closure.update(self.children.get(oid, ()))
        if any(self.owners[i]["active_leases"] for i in closure):
            return None
        return frozenset(closure)

    def prediction(self, closure):
        freed = [a for holders, a in self.allocations if holders <= closure]
        return dict(
            closure_owner_ids=sorted(closure),
            predicted_freed_bytes=sum(int(a["bytes"]) for a in freed),
            predicted_allocation_ids=sorted(a["allocation_id"] for a in freed))


class StateSwitchController(ReceiptRetentionController):
    """Fixed H=32 history, native owner closures, and bounded semantic ghosts.

    lru/scratch_first delegate to frozen v2 policy selection and actions.
    gdsf is a windowed frequency/size adaptation, without GreedyDual aging or
    measured recovery costs. It independently charges each removed owner and
    uses owner-referenced bytes. Closure policies charge each affected request
    signature once and divide by actual allocation release bytes.
    closure_lru uses the identical legal closure menu and release machinery as
    closure, ranking that menu by the most recent use among removed owners.
    The historical lru option remains owner-LRU and is not this strong control.
    """

    def __init__(self, cpp, devices, budgets=None, *, policy="closure",
                 inventory_mode="receipt", batch_devices=False,
                 graph_bindings=None):
        if policy not in POLICIES:
            raise ValueError(f"Unknown state-switch policy: {policy}")
        super().__init__(
            cpp, list(devices), budgets,
            "scratch_first" if policy == "scratch_first" else "lru",
            inventory_mode=inventory_mode, batch_devices=batch_devices)
        self.policy = policy
        self.graph_bindings = dict(graph_bindings or {})
        self.history = deque(maxlen=HISTORY_WINDOW)
        self.ghosts = OrderedDict()
        self.boosts = OrderedDict()
        self._pending_request = None
        self._frequencies = Counter()
        self._current_request = None
        self._refaults = []
        self._refault_total = 0
        self._ghost_evictions = 0
        self._last_live = set()
        self._request_calls = 0

    def set_request(self, graph_id, algorithm, parameters=None):
        """Set current request only; parameters are serialized, never executed."""
        self._pending_request = (
            str(graph_id), algorithm_name(algorithm), _json(parameters or {}),
            self.graph_bindings.get(str(graph_id), "unbound"))

    def _semantic_key(self, owner):
        graph = str(owner.get("graph_id", ""))
        return (
            int(owner["device_id"]), owner["kind"],
            algorithm_name(owner.get("algorithm")), graph,
            int(owner.get("generation", 0)), int(owner.get("partition_id", -1)),
            str(owner.get("policy", "")),
            self.graph_bindings.get(graph, "unbound") if graph else "worker_pool",
        )

    def _matching_requests(self, owner):
        algorithm = algorithm_name(owner.get("algorithm"))
        graph = str(owner.get("graph_id", ""))
        pool = owner["kind"] == "workspace" and not graph
        return {
            request for request in self._frequencies
            if request[1] == algorithm and (pool or request[0] == graph)
        }

    def _begin_history(self, state, graph_id, algorithm):
        request = self._pending_request or (
            str(graph_id), algorithm, "{}", self.graph_bindings.get(str(graph_id), "unbound"))
        self._pending_request = None
        if request[:2] != (str(graph_id), algorithm):
            raise RuntimeError("Pending current request does not match native completion")
        self._current_request = request
        self.history.append(request)
        self._request_calls += 1
        self._frequencies = Counter(self.history)
        self._refaults = []
        for key in list(self.ghosts):
            if self.clock - self.ghosts[key]["evicted_call"] > HISTORY_WINDOW:
                del self.ghosts[key]
        for key in list(self.boosts):
            if self.clock > self.boosts[key]["expires_call"]:
                del self.boosts[key]
        seen_semantic = set()
        for row in state["per_device"]:
            owners = validate_native_row(row)
            for owner in owners.values():
                if owner["kind"] == "catalog":
                    continue
                key = self._semantic_key(owner)
                if key in seen_semantic:
                    raise RuntimeError("Ambiguous live semantic owner identity")
                seen_semantic.add(key)
                oid = owner["owner_id"]
                if self._current_request in self._matching_requests(owner):
                    self.last_use[oid] = self.clock
                if oid in self._last_live or key not in self.ghosts:
                    continue
                ghost = self.ghosts.pop(key)
                age = self.clock - ghost["evicted_call"]
                if not 1 <= age <= HISTORY_WINDOW:
                    continue
                multiplier = 1.0 + (HISTORY_WINDOW - age) / HISTORY_WINDOW
                self.boosts[key] = dict(
                    multiplier=multiplier, expires_call=self.clock + HISTORY_WINDOW - 1)
                self.boosts.move_to_end(key)
                if len(self.boosts) > GHOST_CAPACITY:
                    self.boosts.popitem(last=False)
                self._refault_total += 1
                self._refaults.append(dict(
                    semantic_key=list(key), previous_owner_id=ghost["owner_id"],
                    rebuilt_owner_id=oid, distance_calls=age, multiplier=multiplier))

    def _boost(self, owner):
        if self.policy != "closure_refault":
            return 1.0
        return self.boosts.get(self._semantic_key(owner), {}).get("multiplier", 1.0)

    def _demand(self, row, closure, *, union):
        owners = {o["owner_id"]: o for o in row["owners"]}
        request_weights = {}
        unknown = {}
        separate = 0.0
        for oid in sorted(closure):
            owner = owners[oid]
            requests = self._matching_requests(owner)
            boost = self._boost(owner)
            if not requests:
                # Semantic unknown owners get the same unit prior in every new policy.
                unknown[self._semantic_key(owner)] = boost
                separate += boost
            for request in requests:
                current = CURRENT_MULTIPLIER if request == self._current_request else 1.0
                demand = self._frequencies[request] * current * boost
                request_weights[request] = max(request_weights.get(request, 0.0), demand)
                separate += demand
        amount = sum(request_weights.values()) + sum(unknown.values()) if union else separate
        return amount, {
            "affected_requests": [
                dict(signature=list(q), count=self._frequencies[q], demand=request_weights[q])
                for q in sorted(request_weights)],
            "unknown_semantic_owners": len(unknown),
            "demand_unit": "estimated_future_reacquire_demand_dimensionless",
            "request_union": union,
        }

    def _candidate_record(self, row, requested, *, union, deficit, model=None):
        model = model or _BoundaryModel(row)
        closure_set = model.closure(requested)
        if closure_set is None:
            return None
        prediction = model.prediction(closure_set)
        closure = prediction["closure_owner_ids"]
        demand, details = self._demand(row, closure, union=union)
        referenced = sum(model.referenced[i] for i in closure)
        raw_size = prediction["predicted_freed_bytes"] if union else referenced
        denominator = min(raw_size, deficit)
        if denominator <= 0:
            return None
        return dict(
            owner_ids=_request_roots(row, set(closure)), **prediction,
            estimated_reacquire_demand=demand, denominator_bytes=denominator,
            owner_referenced_bytes_sum=referenced,
            raw_size_bytes=raw_size, remaining_deficit_bytes=deficit,
            useful_freed_bytes=min(prediction["predicted_freed_bytes"], deficit),
            score=demand / denominator,
            last_use=max((self.last_use.get(i, 0) for i in closure), default=0),
            **details,
        )

    @staticmethod
    def _sort_key(candidate):
        return (candidate["score"], candidate["last_use"],
                candidate["estimated_reacquire_demand"],
                candidate["predicted_freed_bytes"], tuple(candidate["owner_ids"]))

    def candidates(self, row, deficit):
        model = _BoundaryModel(row)
        owners = model.owners
        union = self.policy != "gdsf"
        seeds = [{i} for i, o in owners.items() if o["kind"] != "catalog"]
        if union:
            seeds += [
                {i for i in a["owners"] if owners[i]["kind"] != "catalog"}
                for a in row["allocations"]]
        unique = {}
        legal_closures = set()
        for seed in set(map(frozenset, seeds)):
            if not seed:
                continue
            closure = model.closure(seed)
            if closure is None:
                continue
            if closure in legal_closures:
                continue
            legal_closures.add(closure)
            candidate = self._candidate_record(
                row, closure, union=union, deficit=deficit, model=model)
            if candidate is not None:
                unique[tuple(candidate["closure_owner_ids"])] = candidate
        if union:
            # A bounded greedy prefix over the native holder-generated menu.
            # This is a candidate generator, not a claim of global optimality.
            prefix = set()
            ordered = sorted(unique.values(), key=self._sort_key)
            for item in ordered:
                prefix.update(item["closure_owner_ids"])
                key = tuple(sorted(prefix))
                candidate = unique.get(key) or self._candidate_record(
                    row, prefix, union=True, deficit=deficit, model=model)
                if candidate is not None:
                    unique[tuple(candidate["closure_owner_ids"])] = candidate
                    if candidate["predicted_freed_bytes"] >= deficit:
                        break
            # Shared allocations with only zero-marginal singleton candidates
            # are admitted by the holder seed above. All-legal fallback handles
            # any remaining union structure without recursive subset search.
            all_legal = set().union(*legal_closures) if legal_closures else set()
            if all_legal:
                key = tuple(sorted(all_legal))
                candidate = unique.get(key) or self._candidate_record(
                    row, all_legal, union=True, deficit=deficit, model=model)
                if candidate is not None:
                    unique[tuple(candidate["closure_owner_ids"])] = candidate
        # Generate exactly the same bounded menu for closure and closure_lru.
        # The common prefix construction runs before the ranking intervention;
        # a policy cannot gain a different legal release menu.
        menu = list(unique.values())
        for candidate in menu:
            candidate["demand_score"] = candidate["score"]
            candidate["score_kind"] = "request_demand_per_useful_released_byte"
            if self.policy == "closure_lru":
                candidate["score"] = candidate["last_use"]
                candidate["score_kind"] = "most_recent_owner_use_in_closure"
        if self.policy in ("closure", "closure_lru"):
            # Both treatments have the same tie breakers; only score changes.
            # Recency ties use bytes and stable identities, never frequency.
            return sorted(menu, key=lambda c: (
                c["score"], c["last_use"], c["predicted_freed_bytes"], tuple(c["owner_ids"])))
        return sorted(menu, key=self._sort_key)

    def _remember_removed(self, before, after, selected):
        after_ids = {o["owner_id"] for row in after["per_device"] for o in row["owners"]}
        for row in before["per_device"]:
            if row["device_id"] not in selected:
                continue
            for owner in row["owners"]:
                if owner["kind"] == "catalog" or owner["owner_id"] in after_ids:
                    continue
                key = self._semantic_key(owner)
                self.ghosts[key] = dict(
                    owner_id=owner["owner_id"], evicted_call=self.clock)
                self.ghosts.move_to_end(key)
                while len(self.ghosts) > GHOST_CAPACITY:
                    self.ghosts.popitem(last=False)
                    self._ghost_evictions += 1

    def _history_stats(self):
        return dict(
            window_calls=HISTORY_WINDOW, observed_calls=self._request_calls,
            retained_call_count=len(self.history), distinct_requests=len(self._frequencies),
            current_multiplier=CURRENT_MULTIPLIER, unknown_owner_prior=1.0,
            ghost_capacity=GHOST_CAPACITY, ghost_entries=len(self.ghosts),
            ghost_capacity_evictions=self._ghost_evictions,
            refault_total=self._refault_total, current_refaults=self._refaults,
            boost_entries=len(self.boosts),
            cost_model="history_only_dimensionless_request_demand",
            measured_recovery_seconds_used=False,
            ghost_scoring_enabled=self.policy == "closure_refault",
        )

    def finish_query(self, graph_id, algorithm):
        # Preserve exact frozen v2 actions for the two historical baselines.
        if self.policy in ("lru", "scratch_first"):
            result = super().finish_query(graph_id, algorithm)
            self._pending_request = None
            result.update(state_switch_policy=self.policy, decisions=[], history_stats={
                "cost_model": "frozen_v2_owner_recency",
                "measured_recovery_seconds_used": False})
            return result
        begin = perf_counter()
        self.clock += 1
        algorithm = algorithm_name(algorithm)
        state = initial = self.inventory()
        self._begin_history(state, graph_id, algorithm)
        actions, decisions = [], []
        counts = dict(full_inventory_exports=1, eviction_calls=0,
                      receipt_device_rows=0, recheck_exports=0,
                      skipped_empty_catalog_calls=0)

        def apply(selected_candidates, reason):
            nonlocal state
            selected = sorted(selected_candidates)
            before = state
            rows = {int(r["device_id"]): r for r in before["per_device"]}
            requested = sorted({i for c in selected_candidates.values() for i in c["owner_ids"]})
            predicted = sum(c["predicted_freed_bytes"] for c in selected_candidates.values())
            receipt = self.evict(requested, selected)
            merged = merge_release_receipt(before, receipt, selected, self.devices)
            merged_rows = {int(r["device_id"]): r for r in merged["per_device"]}
            deltas = {}
            for device in selected:
                candidate = selected_candidates[device]
                old_allocs = {a["allocation_id"] for a in rows[device]["allocations"]}
                new_allocs = {a["allocation_id"] for a in merged_rows[device]["allocations"]}
                removed_allocs = old_allocs - new_allocs
                expected_allocs = set(candidate["predicted_allocation_ids"])
                old_owners = {o["owner_id"] for o in rows[device]["owners"]
                              if o["kind"] != "catalog"}
                new_owners = {o["owner_id"] for o in merged_rows[device]["owners"]
                              if o["kind"] != "catalog"}
                actual = rows[device]["managed_resident_bytes"] - merged_rows[device]["managed_resident_bytes"]
                difference = actual - candidate["predicted_freed_bytes"]
                deltas[str(device)] = difference
                if (difference or removed_allocs != expected_allocs
                        or old_owners - new_owners != set(candidate["closure_owner_ids"])):
                    raise RuntimeError(
                        f"Closure prediction/receipt mismatch on device {device}: "
                        f"bytes_delta={difference}, expected_allocs={sorted(expected_allocs)}, "
                        f"actual_allocs={sorted(removed_allocs)}")
                validate_native_row(merged_rows[device])
            if receipt["released_bytes"] != predicted:
                raise RuntimeError("Aggregate closure prediction differs from release receipt")
            counts["eviction_calls"] += 1
            counts["receipt_device_rows"] += len(selected)
            if self.inventory_mode == "recheck":
                checked = self.inventory()
                counts["full_inventory_exports"] += 1
                counts["recheck_exports"] += 1
                if _state_signature(merged) != _state_signature(checked):
                    raise RuntimeError("Release receipt differs from fresh full inventory")
                merged = checked
            self._remember_removed(before, merged, selected)
            actions.append(dict(
                reason=reason, devices=selected, owner_ids=requested, receipt=receipt,
                predicted_freed_bytes=predicted, freed_bytes=receipt["released_bytes"],
                prediction_error_bytes=receipt["released_bytes"] - predicted,
                per_device_prediction_error_bytes=deltas,
                chosen_by_device={str(d): c for d, c in selected_candidates.items()}))
            state = merged

        if self.budgets is not None:
            groups = [self.devices] if self.batch_devices else [[d] for d in self.devices]
            for group in groups:
                rows = {int(r["device_id"]): r for r in state["per_device"]}
                over = [d for d in group if rows[d]["managed_resident_bytes"] > self.budgets[d]]
                trim = {}
                for device in over:
                    prediction = predict_release(rows[device], [])
                    if prediction["predicted_freed_bytes"]:
                        trim[device] = dict(owner_ids=[], **prediction)
                    else:
                        counts["skipped_empty_catalog_calls"] += 1
                if trim:
                    apply(trim, "idle_catalog")
                while True:
                    rows = {int(r["device_id"]): r for r in state["per_device"]}
                    over = [d for d in group if rows[d]["managed_resident_bytes"] > self.budgets[d]]
                    if not over:
                        break
                    choices = {}
                    for device in over:
                        deficit = rows[device]["managed_resident_bytes"] - self.budgets[device]
                        menu = self.candidates(rows[device], deficit)
                        if not menu:
                            raise RuntimeError(
                                f"No legal owner release can satisfy device {device} "
                                f"retained budget {self.budgets[device]}")
                        choices[device] = menu[0]
                        decisions.append(dict(
                            device=device, call=self.clock, deficit_bytes=deficit,
                            policy=self.policy, chosen=menu[0], candidates=menu))
                    apply(choices, self.policy)
            if any(r["managed_resident_bytes"] > self.budgets[r["device_id"]]
                   for r in state["per_device"]):
                raise RuntimeError("Final per-device retention budget violated")
        self._last_live = {o["owner_id"] for r in state["per_device"] for o in r["owners"]}
        self.last_use = {i: t for i, t in self.last_use.items() if i in self._last_live}
        return dict(
            before_retention=initial, after_retention=state, actions=actions,
            decisions=decisions, history_stats=self._history_stats(),
            controller_seconds=perf_counter() - begin,
            retention_policy=self.policy, state_switch_policy=self.policy,
            budgets=self.budgets, controller_version="state_switch_v2_closure_lru",
            inventory_mode=self.inventory_mode, batch_devices=self.batch_devices,
            control_counts=counts,
            budget_scope="Completed-query retained BFS/PR views, workspaces and catalog; not transient or total CUDA peak",
            accounting_contract="Fresh full inventory per query; exact native receipts and closure predictions; exclusive session lock",
        )


class StateSwitchSession(ManagedNativeSession):
    """Same frozen native query/output path, with an independent new controller."""

    def __init__(self, cpp, manifests, *, devices=(0,), budgets=None,
                 policy="closure", inventory_mode="receipt", batch_devices=False):
        manifests = dict(manifests)
        bindings = {}
        for manifest in manifests.values():
            payload = Path(manifest).read_bytes()
            graph = json.loads(payload)["graph"]
            identity = str(graph["id"])
            digest = hashlib.sha256(payload).hexdigest()
            if identity in bindings and bindings[identity] != digest:
                raise ValueError("Different manifests share the same native graph identity")
            bindings[identity] = digest
        controller = StateSwitchController(
            cpp, devices, budgets, policy=policy, inventory_mode=inventory_mode,
            batch_devices=batch_devices, graph_bindings=bindings)
        super().__init__(
            cpp, manifests, devices=devices, budgets=budgets,
            retention="scratch_first" if policy == "scratch_first" else "lru",
            inventory_mode=inventory_mode, batch_devices=batch_devices)
        self.controller = controller

    def query(self, graph_id, algorithm, parameters=None):
        begin = perf_counter()
        with self._lock:
            if self.closed:
                raise RuntimeError("Session closed")
            self.controller.set_request(self.metadata[graph_id]["id"], algorithm, parameters)
            try:
                result = super().query(graph_id, algorithm, parameters)
            except BaseException:
                self.controller._pending_request = None
                raise
            return replace(result, seconds=perf_counter() - begin)

    def capability_descriptor(self):
        descriptor = super().capability_descriptor()
        return replace(
            descriptor,
            control_routes=descriptor.control_routes + (
                "allocation_holder_closure", "bounded_history_request_demand",
                "optional_semantic_refault_ghosts"),
            qualification="CPU closure/history semantics and each native GPU cohort must be reported separately")
