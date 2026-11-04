"""Exact request-boundary accounting without redundant all-device exports.

Each completed query obtains a fresh full inventory. A native release returns
the complete post-release inventory of every device it modified. Under the
session's exclusive request lock, those rows replace the corresponding rows in
the current snapshot; untouched devices cannot change. This is not a prediction,
a cross-query memory cache, or a relaxed audit.

``recheck`` adds a fresh all-device export after each release, compares it with
the receipt-derived state, and charges that cost inside the query. Use it for
qualification/ablation; use the same mode for competing workspace policies.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from easygraph.gpu._retention import (
    BoundaryRetentionController,
    ManagedNativeSession as V1ManagedNativeSession,
    algorithm_name,
    check_inventory,
)


def _state_signature(snapshot):
    """Semantic inventory projection, independent of row serialization order."""
    return {
        int(row["device_id"]): {
            "bytes": int(row["managed_resident_bytes"]),
            "owners": {o["owner_id"]: o for o in row["owners"]},
            "allocations": {
                a["allocation_id"]: a for a in row["allocations"]
            },
        }
        for row in snapshot["per_device"]
    }


def merge_release_receipt(state, receipt, selected, devices):
    """Validate and merge an exact native release receipt; never guess bytes."""
    selected = list(selected)
    patch = check_inventory(receipt["after_inventory"], selected)
    before = {int(row["device_id"]): row for row in state["per_device"]}
    changed = {int(row["device_id"]): row for row in patch["per_device"]}
    before_bytes = sum(before[d]["managed_resident_bytes"] for d in selected)
    after_bytes = sum(changed[d]["managed_resident_bytes"] for d in selected)
    if receipt.get("before_bytes", before_bytes) != before_bytes:
        raise RuntimeError("Release receipt does not start at the current inventory")
    if receipt.get("after_bytes", after_bytes) != after_bytes:
        raise RuntimeError("Release receipt after_bytes disagrees with allocation union")
    if receipt["released_bytes"] != before_bytes - after_bytes:
        raise RuntimeError("Release receipt freed bytes disagree with allocation union")
    if "per_device" in receipt:
        rows = {int(row["device_id"]): row for row in receipt["per_device"]}
        if set(rows) != set(selected) or len(rows) != len(receipt["per_device"]):
            raise RuntimeError("Release receipt device rows are inconsistent")
        for device, row in rows.items():
            if (row["before_bytes"] != before[device]["managed_resident_bytes"]
                    or row["after_bytes"] != changed[device]["managed_resident_bytes"]):
                raise RuntimeError("Per-device release receipt bytes are inconsistent")
    for device in selected:
        if changed[device]["managed_resident_bytes"] > before[device]["managed_resident_bytes"]:
            raise RuntimeError("Release increased a device's managed resident storage")
    merged = dict(state)
    merged["per_device"] = [changed.get(d, before[d]) for d in devices]
    merged["devices"] = list(devices)
    merged["managed_resident_bytes"] = sum(
        row["managed_resident_bytes"] for row in merged["per_device"])
    merged["has_active_leases"] = any(
        owner.get("active_leases", 0)
        for row in merged["per_device"] for owner in row["owners"])
    merged["has_untracked_shared_references"] = any(
        allocation.get("untracked_shared_references", 0)
        for row in merged["per_device"] for allocation in row["allocations"])
    return check_inventory(merged, devices)


class ReceiptRetentionController(BoundaryRetentionController):
    """The v1 policy, using exact release receipts and optional device batching.

    ``batch_devices=False`` retains v1's device/action order, which permits a
    strict state-equivalence ablation. ``True`` executes one independently
    chosen release per violating device in the same native gang operation.
    Devices do not physically share allocations, so choices commute.
    """

    def __init__(self, cpp, devices, budgets=None, retention="lru", *,
                 inventory_mode="receipt", batch_devices=False):
        devices = list(devices)
        if (not devices or len(set(devices)) != len(devices)
                or any(isinstance(d, bool) or not isinstance(d, int) or d < 0
                       for d in devices)):
            raise ValueError("Devices must be distinct nonnegative integer IDs")
        if inventory_mode not in ("receipt", "recheck"):
            raise ValueError("inventory_mode must be receipt or recheck")
        if not isinstance(batch_devices, bool):
            raise ValueError("batch_devices must be a boolean")
        super().__init__(cpp, devices, budgets, retention)
        self.inventory_mode = inventory_mode
        self.batch_devices = batch_devices

    def _candidate(self, row, graph_id, algorithm, attempted):
        candidates = []
        for owner in row["owners"]:
            identity = owner["owner_id"]
            if identity in attempted or owner.get("active_leases", 0):
                continue
            if owner["kind"] == "workspace":
                priority = 0 if self.retention == "scratch_first" else 2
            elif owner["kind"] == "view":
                current = (owner.get("graph_id") == graph_id
                           and algorithm_name(owner.get("algorithm")) == algorithm)
                priority = 3 if current else 1
            else:
                continue
            candidates.append((priority, self.last_use.get(identity, 0), identity))
        return None if not candidates else min(candidates)[2]

    def finish_query(self, graph_id, algorithm):
        begin = perf_counter()
        self.clock += 1
        initial = self.inventory()
        state = initial
        algorithm = algorithm_name(algorithm)
        actions = []
        counts = dict(full_inventory_exports=1, eviction_calls=0,
                      receipt_device_rows=0, recheck_exports=0,
                      skipped_empty_catalog_calls=0)
        for row in state["per_device"]:
            for owner in row["owners"]:
                same = algorithm_name(owner.get("algorithm")) == algorithm
                if same and (owner.get("graph_id") == graph_id or
                             (owner.get("kind") == "workspace" and not owner.get("graph_id"))):
                    self.last_use[owner["owner_id"]] = self.clock

        def apply_release(owner_ids, selected, reason):
            nonlocal state
            receipt = self.evict(owner_ids, selected)
            merged = merge_release_receipt(state, receipt, selected, self.devices)
            counts["eviction_calls"] += 1
            counts["receipt_device_rows"] += len(selected)
            if self.inventory_mode == "recheck":
                checked = self.inventory()
                counts["full_inventory_exports"] += 1
                counts["recheck_exports"] += 1
                if _state_signature(merged) != _state_signature(checked):
                    raise RuntimeError("Release receipt differs from fresh full inventory")
                merged = checked
            actions.append({
                "reason": reason, "devices": list(selected),
                "owner_ids": list(owner_ids), "receipt": receipt,
                "freed_bytes": receipt["released_bytes"],
            })
            state = merged

        if self.budgets is not None:
            # Missing catalog_idle_bytes means an older/testing ABI: retain the
            # safe cleanup call rather than assume that no idle bytes exist.
            def needs_catalog_trim(row):
                return row.get("catalog_idle_bytes", None) != 0

            groups = [self.devices] if self.batch_devices else [[d] for d in self.devices]
            for group in groups:
                rows = {int(row["device_id"]): row for row in state["per_device"]}
                over = [d for d in group
                        if rows[d]["managed_resident_bytes"] > self.budgets[d]]
                trim = [d for d in over if needs_catalog_trim(rows[d])]
                counts["skipped_empty_catalog_calls"] += len(over) - len(trim)
                if trim:
                    apply_release([], trim, "idle_catalog")
                attempted = {d: set() for d in group}
                while True:
                    rows = {int(row["device_id"]): row for row in state["per_device"]}
                    over = [d for d in group
                            if rows[d]["managed_resident_bytes"] > self.budgets[d]]
                    if not over:
                        break
                    candidates = []
                    for device in over:
                        identity = self._candidate(rows[device], graph_id, algorithm,
                                                   attempted[device])
                        if identity is None:
                            raise RuntimeError(
                                f"No legal owner release can satisfy device {device} "
                                f"retained budget {self.budgets[device]}")
                        attempted[device].add(identity)
                        candidates.append(identity)
                    apply_release(candidates, over, self.retention)
            for row in state["per_device"]:
                if row["managed_resident_bytes"] > self.budgets[row["device_id"]]:
                    raise RuntimeError("Final per-device retention budget violated")
        live = {owner["owner_id"] for row in state["per_device"] for owner in row["owners"]}
        self.last_use = {key: value for key, value in self.last_use.items() if key in live}
        return {
            "before_retention": initial, "after_retention": state, "actions": actions,
            "controller_seconds": perf_counter() - begin,
            "retention_policy": self.retention, "budgets": self.budgets,
            "controller_version": "native_view_management_v2",
            "inventory_mode": self.inventory_mode, "batch_devices": self.batch_devices,
            "control_counts": counts,
            "budget_scope": (
                "Completed-query retained BFS/PR views, workspaces and shared catalog "
                "only; not transient or total CUDA peak"),
            "accounting_contract": (
                "Fresh full inventory after every query; exact native receipt rows "
                "for modified devices under the exclusive session request lock"),
        }


class ManagedNativeSession(V1ManagedNativeSession):
    """The same P1/P2 query path and native DSO, with receipt-driven control."""

    def __init__(self, cpp, manifests, *, devices=(0,), budgets=None, retention="lru",
                 inventory_mode="receipt", batch_devices=False):
        # Validate every new option before starting the process-global runtime.
        devices = list(devices)
        controller = ReceiptRetentionController(
            cpp, devices, budgets, retention,
            inventory_mode=inventory_mode, batch_devices=batch_devices)
        super().__init__(cpp, manifests, devices=devices, budgets=budgets,
                         retention=retention)
        self.controller = controller

    def capability_descriptor(self):
        descriptor = super().capability_descriptor()
        return replace(
            descriptor,
            control_routes=descriptor.control_routes + (
                "exact_release_receipt_accounting",
                "optional_independent_device_release_batching"),
            qualification=(
                "CPU semantics and native GPU qualification must be reported "
                "separately for the v2 controller and each device count"))
