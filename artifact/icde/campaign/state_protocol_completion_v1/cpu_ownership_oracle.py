"""Independent CPU ownership registry, extracted unchanged from existing tests. No CUDA."""
from pathlib import Path
import copy, json
import numpy as np

class OwnerGraphCpp:
    """Independent CPU resource model, not a mock of the controller's ordering."""
    def __init__(self, devices=(0,)):
        self.devices = list(devices)
        self.owners = {}
        self.atoms = {}
        self.references = {}
        self.evictions = []
        self.running = False
        self.starts = 0
        self.buffer = np.array([.5, .3, .2])

    def add(self, identity, device, atoms, *, kind='view', graph='G',
            algorithm='bfs', active=0, parent=''):
        self.owners[identity] = dict(owner_id=identity, device_id=device,
            kind=kind, graph_id=graph, algorithm=algorithm, active_leases=active,
            parent_owner_id=parent)
        self.references[identity] = set()
        for atom, size in atoms.items():
            key = (device, atom)
            if key in self.atoms:
                assert self.atoms[key] == size
            self.atoms[key] = size
            self.references[identity].add(key)

    def cpp_egmgpu_managed_state_inventory(self, devices):
        rows = []
        for device in devices:
            owners = [copy.deepcopy(o) for o in self.owners.values()
                      if o['device_id'] == device]
            keys = set().union(*(self.references[o['owner_id']] for o in owners))
            allocations = [dict(allocation_id=f'{d}:{atom}', device_id=d,
                bytes=self.atoms[d, atom], owners=sorted(identity for identity in self.references
                if (d, atom) in self.references[identity])) for d, atom in sorted(keys)]
            rows.append(dict(device_id=device, owners=owners, allocations=allocations,
                             managed_resident_bytes=sum(a['bytes'] for a in allocations)))
        return dict(per_device=rows,
                    managed_resident_bytes=sum(r['managed_resident_bytes'] for r in rows))

    def cpp_egmgpu_managed_evict(self, owner_ids, devices, release_idle=True,
                                release_workspaces=False):
        self.evictions.append((tuple(owner_ids), tuple(devices)))
        before = self.cpp_egmgpu_managed_state_inventory(devices)
        eligible = {k for k, o in self.owners.items() if o['device_id'] in devices}
        unknown = [k for k in owner_ids if k not in eligible]
        selected = set(owner_ids) & eligible
        if release_workspaces:
            selected |= {k for k in eligible if self.owners[k]['kind'] == 'workspace'}
        removed, workspaces, skipped = [], [], []
        for identity in sorted(selected):
            if identity not in self.owners:
                continue
            owner = self.owners[identity]
            children = {k for k in eligible if k in self.owners and
                        self.owners[k]['parent_owner_id'] == identity}
            group = {identity} | children
            if any(self.owners[k]['active_leases'] for k in group):
                skipped.append(identity)
                continue
            for key in sorted(group):
                (workspaces if self.owners[key]['kind'] == 'workspace' else removed).append(key)
                del self.owners[key]
                del self.references[key]
        # Device destruction may overwrite a backend-owned staging buffer.
        if removed or workspaces:
            self.buffer[:] = -999
        after = self.cpp_egmgpu_managed_state_inventory(devices)
        return dict(unknown_owner_ids=unknown, removed_owner_ids=removed,
            released_workspace_owner_ids=workspaces, skipped_active_owner_ids=skipped,
            released_bytes=before['managed_resident_bytes'] - after['managed_resident_bytes'],
            after_inventory=after)

    def cpp_egmgpu_runtime_status(self):
        return {'running': self.running}

    def cpp_egmgpu_runtime_start(self, **options):
        self.running = True
        self.starts += 1

    def cpp_egmgpu_runtime_shutdown(self, **options):
        self.running = False
        self.owners.clear()
        self.references.clear()
        self.buffer[:] = -999

    def cpp_egmgpu_load_partition_manifest(self, path, **options):
        return json.loads(Path(path).read_text())['graph']

    def cpp_egmgpu_partitioned_pagerank(self, graph, alpha, rounds, tolerance, devices):
        self.buffer[:] = [.5, .3, .2]
        self.add('result-view', devices[0], {'topology': 20}, graph=graph['id'], algorithm='pr')
        return {'values_dense': self.buffer, 'converged': True}

    def cpp_egmgpu_partitioned_state_stats(self, devices):
        return {}

class NativeReferenceCpp(OwnerGraphCpp):
    """CPU registry model with separately retained shared-array catalog entries."""

    def __init__(self, devices=(0,)):
        super().__init__(devices)
        self.catalog = set()
        self.exports = 0
        self.mismatch = False
        self._query_serial = 0

    def add(self, identity, device, atoms, *, shared=(), policy="", generation=0, **kw):
        super().add(identity, device, atoms, **kw)
        self.owners[identity].update(
            policy=policy, generation=generation, partition_id=device)
        self.catalog.update((device, atom) for atom in shared)

    def _snapshot(self, devices):
        rows = []
        for device in devices:
            catalog_id = f"catalog:{device}"
            owners = [copy.deepcopy(o) for o in self.owners.values() if o["device_id"] == device]
            owners.append(dict(
                owner_id=catalog_id, device_id=device, kind="catalog", algorithm="shared",
                graph_id="", parent_owner_id="", policy="", generation=0,
                partition_id=-1, active_leases=0))
            keys = {a for oid, refs in self.references.items()
                    if self.owners[oid]["device_id"] == device for a in refs}
            keys |= {a for a in self.catalog if a[0] == device}
            allocations = []
            for key in sorted(keys):
                holders = sorted(oid for oid, refs in self.references.items() if key in refs)
                catalogued = key in self.catalog
                allocations.append(dict(
                    allocation_id=f"{device}:{key[1]}", pointer=key[1], device_id=device,
                    bytes=self.atoms[key], owners=holders + ([catalog_id] if catalogued else []),
                    components=[key[1]], catalog_retained=catalogued,
                    catalog_external_references=len(holders) if catalogued else 0,
                    inventoried_shared_references=len(holders) if catalogued else 0,
                    untracked_shared_references=0, catalog_only_idle=catalogued and not holders))
            for owner in owners:
                owner["referenced_bytes"] = sum(
                    a["bytes"] for a in allocations if owner["owner_id"] in a["owners"])
            rows.append(dict(
                device_id=device, owners=owners, allocations=allocations,
                managed_resident_bytes=sum(a["bytes"] for a in allocations),
                catalog_idle_bytes=sum(a["bytes"] for a in allocations if a["catalog_only_idle"])))
        return dict(
            devices=list(devices), per_device=rows,
            managed_resident_bytes=sum(r["managed_resident_bytes"] for r in rows),
            has_active_leases=any(o["active_leases"] for r in rows for o in r["owners"]),
            has_untracked_shared_references=False)

    def cpp_egmgpu_managed_state_inventory(self, devices):
        self.exports += 1
        return self._snapshot(devices)

    def cpp_egmgpu_managed_evict(self, owner_ids, devices, release_idle=True,
                                release_workspaces=False):
        self.evictions.append((tuple(owner_ids), tuple(devices)))
        before = self._snapshot(devices)
        selected = set(owner_ids)
        removed, workspaces, skipped = [], [], []
        # Native view deletion cascades to its per-view workspace first.
        views = [o["owner_id"] for o in self.owners.values()
                 if o["kind"] == "view" and o["device_id"] in devices and o["owner_id"] in selected]
        for identity in views:
            children = [i for i, o in self.owners.items() if o["parent_owner_id"] == identity]
            if any(self.owners[i]["active_leases"] for i in [identity] + children):
                skipped.append(identity)
                continue
            for child in children:
                del self.owners[child]
                del self.references[child]
                workspaces.append(child)
            del self.owners[identity]
            del self.references[identity]
            removed.append(identity)
        for identity, owner in list(self.owners.items()):
            if (owner["device_id"] not in devices or owner["kind"] != "workspace"
                    or not (identity in selected or release_workspaces)):
                continue
            if owner["active_leases"]:
                skipped.append(identity)
                continue
            del self.owners[identity]
            del self.references[identity]
            workspaces.append(identity)
        if release_idle:
            live_atoms = set().union(*self.references.values()) if self.references else set()
            self.catalog = {a for a in self.catalog if a[0] not in devices or a in live_atoms}
        if self.mismatch:
            # Still internally consistent; differs only from the predicted release.
            self.add("surprise", devices[0], {"unexpected": 1}, graph="surprise")
        after = self._snapshot(devices)
        if removed or workspaces:
            self.buffer[:] = -999
        return dict(
            removed_owner_ids=removed, released_workspace_owner_ids=workspaces,
            skipped_active_owner_ids=skipped,
            before_bytes=before["managed_resident_bytes"],
            after_bytes=after["managed_resident_bytes"],
            released_bytes=before["managed_resident_bytes"] - after["managed_resident_bytes"],
            per_device=[dict(device_id=d, before_bytes=before["per_device"][i]["managed_resident_bytes"],
                             after_bytes=after["per_device"][i]["managed_resident_bytes"])
                        for i, d in enumerate(devices)],
            after_inventory=after)

    def cpp_egmgpu_partitioned_pagerank(self, graph, alpha, rounds, tolerance, devices):
        self.buffer[:] = [.5, .3, .2]
        self._query_serial += 1
        for device in devices:
            self.add(f"result-view-{device}-{self._query_serial}", device, {"topology": 20},
                     graph=graph["id"], algorithm="pr", shared=["topology"])
        return {"values_dense": self.buffer, "converged": True}
