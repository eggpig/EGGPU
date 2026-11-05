"""CPU-only protocol metadata; importing this module never imports a GPU runner."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / 'campaign'
PROVENANCE = json.loads((HERE / 'upstream_provenance.json').read_text())['files']
STATE_PROTOCOL = 'state-complete-workflow-completion-v1'


def checked_source(relative):
    path = HERE.parent / relative.replace('paper_results_v1/', 'paper_results/') if relative.startswith(('presentation/', 'paper_results_v1/')) else CAMPAIGN / relative
    expected = PROVENANCE[relative]
    if path.stat().st_size != expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError('Frozen upstream protocol changed: ' + relative)
    return path


def load_cpu_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, checked_source(relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STATE = load_cpu_module('state_protocol_completion_v1/state_protocol.py', '_paper_completion_state_trace')
PLAN = load_cpu_module('state_protocol_completion_v1/state_plan.py', '_paper_completion_state_grid')

# Read only immutable constants from the actual E6 worker. Do not import its
# adapter modules, inspect devices, or initialize any execution backend.
tree = ast.parse(checked_source('e6_protocol_completion_v2/e6_worker.py').read_text())
names = {'PROTOCOL', 'FUNCTIONS', 'GROUPS', 'RTOL', 'ATOL', 'CONTRACT'}
assignments = [node for node in tree.body if isinstance(node, ast.Assign)
               and {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)} <= names]
scope = {'__builtins__': {}}
exec(compile(ast.Module(body=assignments, type_ignores=[]), '<frozen-E6-constants>', 'exec'), scope)
E6_PROTOCOL, E6_CONTRACT, E6_GROUPS = (scope[name] for name in ('PROTOCOL', 'CONTRACT', 'GROUPS'))


def e6_configurations():
    # Execute only the pure configurations function from the bound planner.
    source = ast.parse(checked_source('e6_protocol_completion_v2/build_plans.py').read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'configurations')
    context = {'__builtins__': {'dict': dict}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<frozen-E6-grid>', 'exec'), context)
    return context['configurations']()


def state_cells():
    return [dict(row, gpu_count=row['partitions']) for row in PLAN.grid()]


def e6_cells():
    return [dict(experiment='E6', dataset=graph, **row)
            for graph in ('EP', 'LJ', 'OR') for row in e6_configurations()]


def e6_expected_devices(cell, function):
    system = cell['system']
    fallback = function in E6_GROUPS.get(cell['ablate'], ())
    if system == 'igraph': return []
    if system == 'cugraph_igraph': return [] if function in ('constraint', 'closeness') else [0]
    if system == 'cugraph_composite': return [0]
    if fallback: return [] if cell['coverage_backend'] == 'igraph' else [0]
    return list(range(cell['gpu_count'])) if cell['gpu_count'] > 1 and function in ('effective_size', 'constraint', 'closeness') else [0]
