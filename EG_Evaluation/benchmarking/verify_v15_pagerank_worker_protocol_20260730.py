#!/usr/bin/env python3
"""Statically attest the V15 EGGPU PageRank timing boundary.

The regular-matrix runner imports ``library_baselines.py`` in every child
process.  The V15 PageRank correction must only run after that worker has been
changed from the generic keyword adapter to one direct public invocation.
This fail-closed checker is intentionally independent of measured timings: it
looks only at the worker source and emits a deterministic, hash-bound
attestation that the correction launcher records beside every result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
PROTOCOL_NAME = "eggpu_pagerank_direct_public_call_fail_closed_v1"


class ProtocolError(ValueError):
    """Raised when the worker does not implement the correction contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def all_calls(node: ast.AST) -> list[ast.Call]:
    return [item for item in ast.walk(node) if isinstance(item, ast.Call)]


def is_pagerank_test(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
        return False
    values = (test.left, test.comparators[0])
    return any(
        isinstance(value, ast.Constant) and value.value == "PageRank"
        for value in values
    )


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ProtocolError(f"missing worker function {name}()")


def find_pagerank_branch(function: ast.FunctionDef) -> ast.If:
    matches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If) and is_pagerank_test(node.test)
    ]
    if len(matches) != 1:
        raise ProtocolError(
            f"expected one PageRank branch in {function.name}(), found {len(matches)}"
        )
    return matches[0]


def direct_pagerank_calls(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in all_calls(node)
        if call_name(call) == "eg.pagerank"
    ]


def local_callables(branch: ast.AST) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in ast.walk(branch):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = node
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node.value
    return result


def first_arg_name(call: ast.Call) -> str:
    if not call.args:
        return ""
    first = call.args[0]
    return first.id if isinstance(first, ast.Name) else ""


def is_name(node: ast.AST, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def require_direct_pagerank_call(call: ast.Call) -> None:
    """Require the one public call and its exact experiment arguments."""

    if len(call.args) != 1 or not is_name(call.args[0], "g"):
        raise ProtocolError(
            "direct eg.pagerank call must have exactly one positional graph argument g"
        )
    if any(keyword.arg is None for keyword in call.keywords):
        raise ProtocolError("direct eg.pagerank call may not expand **kwargs")
    keyword_map = {keyword.arg: keyword.value for keyword in call.keywords}
    expected_names = {"alpha", "max_iter", "tol", "weight"}
    if set(keyword_map) != expected_names:
        raise ProtocolError(
            "direct eg.pagerank keywords must be exactly "
            "alpha, max_iter, tol, and weight"
        )
    expected_values = {
        "alpha": "pr_alpha",
        "max_iter": "pr_max_iter",
        "tol": "pr_tol",
    }
    for keyword, expected_name in expected_values.items():
        if not is_name(keyword_map[keyword], expected_name):
            raise ProtocolError(
                f"direct eg.pagerank {keyword} must use {expected_name}"
            )
    if not is_none(keyword_map["weight"]):
        raise ProtocolError("direct eg.pagerank call must explicitly use weight=None")


def require_return_only_callable(
    callable_node: ast.AST, direct_call: ast.Call
) -> None:
    if isinstance(callable_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if (
            len(callable_node.body) != 1
            or not isinstance(callable_node.body[0], ast.Return)
            or callable_node.body[0].value is not direct_call
        ):
            raise ProtocolError(
                "the timed PageRank callable body must be exactly one direct return"
            )
    elif isinstance(callable_node, ast.Lambda):
        if callable_node.body is not direct_call:
            raise ProtocolError(
                "the timed PageRank lambda must be exactly the direct public call"
            )
    else:
        raise ProtocolError("unsupported timed PageRank callable")


def function_definitions(tree: ast.AST, name: str) -> list[ast.FunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]


def require_unique_helper(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = function_definitions(tree, name)
    if len(matches) != 1:
        raise ProtocolError(f"expected one {name}() helper, found {len(matches)}")
    return matches[0]


def contains_name(node: ast.AST, expected: str) -> bool:
    return any(is_name(item, expected) for item in ast.walk(node))


def contains_call(node: ast.AST, names: set[str]) -> bool:
    return any(
        call_name(item) in names
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
    )


def raising_guards(function: ast.FunctionDef) -> list[ast.If]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and any(isinstance(item, ast.Raise) for item in ast.walk(node))
    ]


def compare_has_nonpositive_boundary(test: ast.AST) -> bool:
    for node in ast.walk(test):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
            continue
        operands = [node.left, *node.comparators]
        for operand in operands:
            if isinstance(operand, ast.Constant) and isinstance(
                operand.value, (int, float)
            ):
                if float(operand.value) <= 0:
                    return True
            if (
                isinstance(operand, ast.UnaryOp)
                and isinstance(operand.op, ast.USub)
                and isinstance(operand.operand, ast.Constant)
                and isinstance(operand.operand.value, (int, float))
            ):
                return True
    return False


def require_validator_helper(tree: ast.Module) -> ast.FunctionDef:
    helper = require_unique_helper(tree, "validate_pagerank_result")
    guards = raising_guards(helper)
    has_shape_guard = any(
        contains_name(guard.test, "n")
        and (
            contains_call(guard.test, {"len"})
            or any(
                isinstance(node, ast.Attribute) and node.attr == "shape"
                for node in ast.walk(guard.test)
            )
        )
        for guard in guards
    )
    has_finite_guard = any(
        any(
            call_name(node).endswith("isfinite")
            for node in ast.walk(guard.test)
            if isinstance(node, ast.Call)
        )
        for guard in guards
    )
    has_nonnegative_guard = any(
        compare_has_nonpositive_boundary(guard.test) for guard in guards
    )
    has_sum_guard = any(
        any(
            call_name(node).endswith("isclose")
            for node in ast.walk(guard.test)
            if isinstance(node, ast.Call)
        )
        for guard in guards
    ) and any(
        call_name(node) in {"sum", "math.fsum"}
        or call_name(node).endswith(".sum")
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
    )
    missing = [
        label
        for label, present in (
            ("shape/cardinality", has_shape_guard),
            ("finite", has_finite_guard),
            ("nonnegative", has_nonnegative_guard),
            ("unit-sum", has_sum_guard),
        )
        if not present
    ]
    if missing:
        raise ProtocolError(
            "validate_pagerank_result is not fail-closed for: "
            + ", ".join(missing)
        )
    return helper


def assignment_target_for_call(
    function: ast.FunctionDef, target_call: ast.Call
) -> str:
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None or target_call not in ast.walk(value):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if len(names) == 1:
            return names[0]
    return ""


def require_kernel_helper(tree: ast.Module) -> ast.FunctionDef:
    helper = require_unique_helper(tree, "require_kernel_time")
    reads = [
        call
        for call in all_calls(helper)
        if call_name(call).endswith("get_last_kernel_time")
    ]
    if len(reads) != 1:
        raise ProtocolError(
            "require_kernel_time must call get_last_kernel_time exactly once"
        )
    read = reads[0]
    if (
        len(read.args) != 1
        or read.keywords
        or not (
            is_name(read.args[0], "kernel_key")
            or (
                isinstance(read.args[0], ast.Constant)
                and read.args[0].value == "pagerank"
            )
        )
    ):
        raise ProtocolError(
            "get_last_kernel_time must read the requested PageRank kernel key"
        )
    value_name = assignment_target_for_call(helper, read)
    if not value_name:
        raise ProtocolError(
            "require_kernel_time must bind the get_last_kernel_time result"
        )
    guards = raising_guards(helper)
    has_missing_guard = any(
        any(
            isinstance(node, ast.Compare)
            and contains_name(node, value_name)
            and any(isinstance(op, (ast.Is, ast.Eq)) for op in node.ops)
            and any(is_none(value) for value in [node.left, *node.comparators])
            for node in ast.walk(guard.test)
        )
        for guard in guards
    )
    has_finite_guard = any(
        contains_name(guard.test, value_name)
        and any(
            call_name(node).endswith("isfinite")
            for node in ast.walk(guard.test)
            if isinstance(node, ast.Call)
        )
        for guard in guards
    )
    has_positive_guard = any(
        contains_name(guard.test, value_name)
        and compare_has_nonpositive_boundary(guard.test)
        for guard in guards
    )
    if not (has_missing_guard and has_finite_guard and has_positive_guard):
        raise ProtocolError(
            "require_kernel_time must raise for missing, non-finite, and "
            "non-positive kernel timings"
        )
    if any(
        call_name(call) == "kernel_or_algo" for call in all_calls(helper)
    ) or contains_name(helper, "algo_seconds"):
        raise ProtocolError("require_kernel_time may not fall back to algorithm time")
    for handler in [
        node for node in ast.walk(helper) if isinstance(node, ast.ExceptHandler)
    ]:
        if not any(isinstance(node, ast.Raise) for node in ast.walk(handler)):
            raise ProtocolError(
                "require_kernel_time may not swallow kernel-timing exceptions"
            )
    returns = [
        node.value
        for node in ast.walk(helper)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    if len(returns) != 1:
        raise ProtocolError("require_kernel_time must have one value return")
    returned = returns[0]
    if not (
        isinstance(returned, ast.Call)
        and call_name(returned) == "float"
        and len(returned.args) == 1
        and is_name(returned.args[0], value_name)
        and not returned.keywords
    ):
        raise ProtocolError(
            "require_kernel_time must return float(the validated kernel timing)"
        )
    return helper


def exact_signature_preflight(
    worker: ast.FunctionDef, timed_callable: ast.AST, timed_line: int
) -> ast.Call:
    calls = [
        call
        for call in all_calls(worker)
        if call_name(call) == "inspect.signature"
    ]
    if len(calls) != 1:
        raise ProtocolError(
            f"expected one inspect.signature(eg.pagerank), found {len(calls)}"
        )
    call = calls[0]
    if (
        len(call.args) != 1
        or call.keywords
        or not isinstance(call.args[0], ast.Attribute)
        or not is_name(call.args[0].value, "eg")
        or call.args[0].attr != "pagerank"
    ):
        raise ProtocolError("signature preflight must be inspect.signature(eg.pagerank)")
    if call.lineno >= timed_line or call in ast.walk(timed_callable):
        raise ProtocolError(
            "inspect.signature(eg.pagerank) must be outside and before the timer"
        )
    return call


def attest(source: Path) -> dict[str, object]:
    source = source.resolve(strict=True)
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source))
    worker = find_function(tree, "bench_easygraph_mode")
    branch = find_pagerank_branch(worker)
    # In Python's AST an ``elif`` is represented as another ``If`` inside the
    # preceding node's ``orelse``. Walking the whole PageRank ``If`` would
    # therefore inspect every later function branch and could falsely attribute
    # their adapters or timer fallbacks to PageRank. The protocol scope is the
    # selected branch body only.
    branch_scope = ast.Module(body=branch.body, type_ignores=[])
    branch_calls = all_calls(branch_scope)
    branch_names = [call_name(call) for call in branch_calls]

    if "call_with_supported_kwargs" in branch_names:
        raise ProtocolError(
            "PageRank still calls call_with_supported_kwargs inside its branch"
        )
    if "kernel_or_algo" in branch_names:
        raise ProtocolError(
            "PageRank still permits the algorithm-time kernel fallback"
        )

    timed_calls = [
        call for call in branch_calls if call_name(call) == "timed_algorithm"
    ]
    if len(timed_calls) != 1:
        raise ProtocolError(
            f"expected one PageRank timed_algorithm call, found {len(timed_calls)}"
        )
    timed_call = timed_calls[0]
    callable_name = first_arg_name(timed_call)
    callables = local_callables(branch_scope)
    if not callable_name or callable_name not in callables:
        raise ProtocolError(
            "timed_algorithm must receive a named PageRank-only direct callable"
        )
    timed_callable = callables[callable_name]
    pagerank_calls = direct_pagerank_calls(timed_callable)
    if len(pagerank_calls) != 1:
        raise ProtocolError(
            "the timed PageRank callable must contain exactly one eg.pagerank call"
        )
    direct_call = pagerank_calls[0]
    require_direct_pagerank_call(direct_call)
    require_return_only_callable(timed_callable, direct_call)
    if [call for call in all_calls(timed_callable) if call is not direct_call]:
        raise ProtocolError(
            "the timed PageRank callable contains non-public-call work"
        )

    signature_call = exact_signature_preflight(
        worker, timed_callable, timed_call.lineno
    )
    validator_helper = require_validator_helper(tree)
    kernel_helper = require_kernel_helper(tree)

    validation_calls = [
        call
        for call in branch_calls
        if call_name(call) == "validate_pagerank_result"
        and call.lineno > timed_call.lineno
        and call not in ast.walk(validator_helper)
    ]
    if len(validation_calls) != 1:
        raise ProtocolError(
            "expected one validate_pagerank_result(ranks, n) after timed_algorithm"
        )
    validation_call = validation_calls[0]
    if (
        len(validation_call.args) != 2
        or validation_call.keywords
        or not is_name(validation_call.args[0], "ranks")
        or not is_name(validation_call.args[1], "n")
    ):
        raise ProtocolError(
            "post-timer validation must be validate_pagerank_result(ranks, n)"
        )

    kernel_calls = [
        call
        for call in branch_calls
        if call_name(call) == "require_kernel_time"
        and call.lineno > timed_call.lineno
        and call not in ast.walk(kernel_helper)
    ]
    if len(kernel_calls) != 1:
        raise ProtocolError(
            "expected one require_kernel_time('pagerank') after timed_algorithm"
        )
    kernel_call = kernel_calls[0]
    if (
        len(kernel_call.args) != 1
        or kernel_call.keywords
        or not isinstance(kernel_call.args[0], ast.Constant)
        or kernel_call.args[0].value != "pagerank"
    ):
        raise ProtocolError(
            "post-timer kernel read must be require_kernel_time('pagerank')"
        )
    kernel_value_name = assignment_target_for_call(worker, kernel_call)
    if not kernel_value_name:
        raise ProtocolError("strict PageRank kernel timing must be assigned")
    emit_calls = [
        call for call in branch_calls if call_name(call) == "emit_metrics"
    ]
    if len(emit_calls) != 1:
        raise ProtocolError(
            f"expected one PageRank emit_metrics call, found {len(emit_calls)}"
        )
    emit_call = emit_calls[0]
    if len(emit_call.args) < 5 or not is_name(
        emit_call.args[4], kernel_value_name
    ):
        raise ProtocolError(
            "emit_metrics must publish the validated strict kernel timing"
        )

    return {
        "status": "pass",
        "protocol": PROTOCOL_NAME,
        "source_file": str(source),
        "source_sha256": sha256(source),
        "worker_function": worker.name,
        "pagerank_branch_line": branch.lineno,
        "signature_preflight": {
            "status": "pass",
            "line": signature_call.lineno,
            "call": "inspect.signature(eg.pagerank)",
            "outside_timer": True,
        },
        "timed_public_call": {
            "status": "pass",
            "callable": callable_name,
            "timed_algorithm_line": timed_call.lineno,
            "direct_call_line": direct_call.lineno,
            "call": "eg.pagerank",
            "graph_argument": "g",
            "explicit_keywords": ["alpha", "max_iter", "tol", "weight"],
            "keyword_bindings": {
                "alpha": "pr_alpha",
                "max_iter": "pr_max_iter",
                "tol": "pr_tol",
            },
            "weight": None,
            "return_only_callable": True,
            "generic_adapter_absent": True,
        },
        "result_validation": {
            "status": "pass",
            "outside_timer": True,
            "call": "validate_pagerank_result",
            "line": validation_call.lineno,
            "helper_line": validator_helper.lineno,
            "checks": ["shape", "finite", "nonnegative", "unit_sum"],
        },
        "kernel_timing": {
            "status": "pass",
            "missing_value_policy": "fail_closed",
            "algorithm_time_fallback_absent": True,
            "call": "require_kernel_time",
            "line": kernel_call.lineno,
            "helper_line": kernel_helper.lineno,
            "published_variable": kernel_value_name,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expect-attestation",
        type=Path,
        help="Require byte-equivalent JSON semantics to a pre-run attestation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = attest(args.source)
    if args.expect_attestation:
        expected = json.loads(
            args.expect_attestation.read_text(encoding="utf-8")
        )
        if payload != expected:
            raise ProtocolError(
                "worker protocol/source changed between preflight and postflight"
            )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
