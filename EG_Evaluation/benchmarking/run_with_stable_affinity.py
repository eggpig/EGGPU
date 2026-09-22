#!/usr/bin/env python3
"""Pin CPU affinity and NUMA memory policy, then exec a benchmark command.

This fills the small deployment gap on hosts that provide libnuma but not the
``numactl`` CLI.  The policy survives ``exec`` and is independently verified
again by the controlled benchmark runner before any measured worker starts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from stable_timing_protocol import (
    apply_controlled_thread_environment,
    bind_memory_nodes,
    collect_execution_placement,
    format_linux_list,
    parse_linux_list,
    validate_controlled_execution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpus",
        required=True,
        help="Exact Linux CPU list, for example 32-39,96-103.",
    )
    parser.add_argument(
        "--membind",
        required=True,
        help="Exact NUMA memory-node list, for example 1.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to exec; prefix it with `--`.",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after `--`")
    return args


def launch(args: argparse.Namespace) -> None:
    cpus = parse_linux_list(args.cpus)
    nodes = parse_linux_list(args.membind)
    if not cpus:
        raise SystemExit("--cpus must contain at least one CPU")
    if not nodes:
        raise SystemExit("--membind must contain at least one NUMA node")

    os.sched_setaffinity(0, set(cpus))
    bind_memory_nodes(nodes)
    canonical_cpus = format_linux_list(cpus)
    canonical_nodes = format_linux_list(nodes)
    os.environ["EGGPU_EXPECTED_CPU_AFFINITY"] = canonical_cpus
    os.environ["EGGPU_EXPECTED_NUMA_NODES"] = canonical_nodes
    os.environ["EGGPU_AFFINITY_LAUNCHER"] = str(
        os.path.realpath(__file__)
    )
    os.environ["EGGPU_NUMA_BINDING_METHOD"] = (
        "libnuma_numa_set_membind_verified_by_numa_get_membind"
    )
    apply_controlled_thread_environment(os.environ)
    placement = collect_execution_placement(os.environ)
    validate_controlled_execution(
        placement,
        expected_cpu_affinity=canonical_cpus,
        expected_numa_nodes=canonical_nodes,
    )
    os.environ["EGGPU_AFFINITY_LAUNCH_RECORD"] = json.dumps(
        {
            "cpu_affinity": canonical_cpus,
            "numa_membind": canonical_nodes,
            "binding_method": os.environ["EGGPU_NUMA_BINDING_METHOD"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    os.execvpe(args.command[0], args.command, os.environ)


if __name__ == "__main__":
    launch(parse_args())
