"""Portable, CPU-only reconstruction of a captured physical recovery problem.

Packet callbacks are frozen over the complete (at most 128) eligible menu.
Relocation is reconstructed from the actual physical slot offsets/capacities,
using the R6 growth rule. A capture is diagnostic and never timed as service.
"""
from dataclasses import asdict
from itertools import combinations

from easygraph.gpu._arena.round6_recovery.planner_v2 import Component, CostModel, Problem, Query


def encode(problem, arena_slots, **metadata):
    names = tuple(sorted(problem.eligible_names))
    packet = []
    for size in range(len(names) + 1):
        for selected in combinations(names, size):
            value = 0 if not selected or problem.packet_bytes is None else problem.packet_bytes(frozenset(selected))
            packet.append(dict(selected=list(selected), payload_bytes=int(value)))
    record = dict(schema='recovery_problem_v1', metadata=metadata,
        components=[asdict(c) for c in problem.components],
        queries=[asdict(q) for q in problem.queries],
        resident_names=sorted(problem.resident_names), eligible_names=sorted(problem.eligible_names),
        stale_names=sorted(problem.eligible_names if problem.stale_names is None else problem.stale_names),
        budget_bytes=problem.budget_bytes, fixed_bytes=problem.fixed_bytes,
        packet_alignment=problem.packet_alignment,
        cost_model=None if problem.cost_model is None else asdict(problem.cost_model),
        packet_table=packet, physical_slots=arena_slots)
    restored = decode(record)
    for row in packet:
        selected=frozenset(row['selected'])
        assert restored.packet_bytes(selected)==row['payload_bytes']
        # Every subset of current residents would be exponential in graph count.
        # Exact validation for actual evaluated retained sets is done by caller.
        assert restored.growth_d2d_bytes(selected, problem.resident_names) == problem.growth_d2d_bytes(selected, problem.resident_names)
    return record


def decode(record):
    if record['schema'] != 'recovery_problem_v1': raise ValueError('Unknown problem schema')
    components = tuple(Component(**c) for c in record['components'])
    by_name = {c.name:c for c in components}
    table = {frozenset(p['selected']):p['payload_bytes'] for p in record['packet_table']}
    slots = record['physical_slots']
    fixed = record['fixed_bytes']
    def packet(selected): return table[frozenset(selected)]
    def relocation(selected, retained):
        if not any(by_name[n].capacity_bytes > by_name[n].old_capacity for n in selected): return 0
        cursor = fixed; shifted = 0; moved = 0
        physical = {by_name[n].allocation:n for n in retained}
        for name in sorted(physical.values(), key=lambda n:slots[by_name[n].allocation]['offset']):
            slot = slots[by_name[name].allocation]
            if slot['offset'] != cursor: moved += 2*slot['capacity']
            if shifted: moved += 2*slot['capacity']
            cursor += slot['capacity']
            if name in selected: shifted += max(0, by_name[name].capacity_bytes-slot['capacity'])
        return int(moved)
    return Problem(components, frozenset(record['resident_names']), frozenset(record['eligible_names']),
        tuple(Query(tuple(q['required']), q['transient_bytes'], q['label']) for q in record['queries']),
        record['budget_bytes'], fixed_bytes=fixed, packet_bytes=packet,
        packet_alignment=record['packet_alignment'],
        cost_model=None if record['cost_model'] is None else CostModel(**record['cost_model']),
        stale_names=frozenset(record['stale_names']), growth_d2d_bytes=relocation)
