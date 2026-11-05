"""Explicitly selected, source-bound E1 statistics. Never searches for latest runs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

DATASETS = ('US', 'EP', 'LJ', 'OR', 'TW', 'IN', 'UK')
ALGORITHMS = ('bfs', 'sssp', 'wcc', 'pr')
METRICS = ('first', 'ready')
REPEATS = tuple(range(10))
QUALITY = {'pr_residual_l1': 1e-6, 'pr_mass_error': 1e-6, 'pr_reference_l1': 1e-5}
CALL_SCHEMAS = {'eggpu.e1.sg.job.v1', 'eggpu.native.job.v1',
                'eggpu.e1.sg.author.v1', 'eggpu.e1.c2graph_p.v1',
                'cugraph-mg-host-csr-complete-v1'}
GUM_SCHEMA = 'gum-canonical-complete-monotonic-v2'
CONTINUOUS_SCHEMAS = {'eggpu.e1.sg.author.v1', 'eggpu.e1.c2graph_p.v1'}
FAILURES = {'FAIL', 'FAILED', 'FAIL_QUALITY', 'QUALITY_FAIL', 'TIMEOUT', 'OOM',
            'INDEX_UNSUPPORTED', 'FAILED_PROCESS', 'FAILED_CLOSE', 'BUILD_FAILED'}


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class Evidence:
    """Only explicit bindings, with optional explicit relocation; hash each inode once."""
    def __init__(self, base, remaps=()):
        self.base = Path(base)
        self.remaps = [(Path(x['from']), Path(x['to'])) for x in remaps]
        self.cache = {}
        self.verified = {}

    def path(self, value):
        p = Path(value)
        for origin, target in self.remaps:
            try:
                p = target / p.relative_to(origin)
                break
            except ValueError:
                pass
        return p if p.is_absolute() else self.base / p

    def verify(self, item):
        require(isinstance(item, dict) and {'path', 'sha256'} <= item.keys(),
                'Evidence requires an explicit path and SHA256')
        p = self.path(item['path'])
        st = p.stat()
        require(p.is_file(), 'Evidence is not a regular file: ' + str(p))
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if key not in self.cache:
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b''):
                    h.update(chunk)
            self.cache[key] = h.hexdigest()
        require(self.cache[key] == item['sha256'], 'Evidence SHA changed: ' + str(p))
        require('bytes' not in item or item['bytes'] == st.st_size,
                'Evidence size changed: ' + str(p))
        self.verified[str(p)] = dict(sha256=self.cache[key], bytes=st.st_size)
        return p

    def read(self, item):
        return json.loads(self.verify(item).read_text())


def input_domains(evidence, registry_binding):
    registry = evidence.read(registry_binding)
    require(registry['status'] == 'ALL_SEVEN_CANONICAL_READY', 'Seven frozen inputs are not ready')
    require(set(registry['records']) == set(DATASETS), 'Exactly the seven frozen graphs are required')
    domains = {}
    for dataset in DATASETS:
        entry = registry['records'][dataset]
        receipt_binding = {'path': entry['receipt_path'], 'sha256': entry['receipt_sha256']}
        path = evidence.verify(receipt_binding)
        receipt = json.loads(path.read_text())
        require(receipt == entry['receipt'] and receipt['status'] == 'PASS_CANONICAL_INPUT',
                'Registry differs from complete input receipt')
        files = receipt['files']
        for item in files.values():
            evidence.verify(item)
        meta = evidence.read(files['canonical'])
        sources = evidence.read(files['sources'])['primary_sources']
        require(len(sources) == 3 and len(set(sources)) == 3, 'Three frozen sources required')
        identity = dict(receipt={'bytes': path.stat().st_size, 'sha256': entry['receipt_sha256']},
                        files={k: {x: v[x] for x in ('bytes', 'sha256')} for k, v in files.items()},
                        sources=sources, n=meta['num_nodes'], m=meta['num_entries'], directed=meta['directed'])
        domains[dataset] = dict(identity=identity, sha256=digest(identity), sources=sources,
                                n=meta['num_nodes'], directed=meta['directed'])
    return domains


def validate_epoch(evidence, binding, specification, freeze):
    epoch = evidence.read(binding)
    require(epoch.get('protocol') == specification['protocol'], 'Selected epoch protocol differs')
    require(epoch.get('system') == specification['system'], 'Epoch system differs')
    if 'algorithm' in epoch:
        require(epoch['algorithm'] == specification['algorithm'], 'Epoch algorithm differs')
    if 'variant' in epoch:
        require(epoch['variant'] == specification.get('variant'), 'Epoch author variant differs')
    if 'release_sources' in epoch:
        observed = epoch['release_sources']
        expected = freeze['source_files']
        for name, info in observed.items():
            require(expected.get(name) == info, 'Runtime source is not the selected final release: ' + name)
        if specification['role'] == 'eggpu':
            required = {name for name in expected if name.startswith(('easygraph/', 'cpp_easygraph/'))
                        and Path(name).suffix in ('.py', '.cu', '.cpp', '.c', '.h', '.hpp', '.cuh')}
            require(required <= observed.keys(), 'Final EGGPU runtime source coverage is incomplete')
    else:
        require(specification['role'] != 'eggpu', 'EGGPU epoch lacks actual runtime source hashes')
    if specification['schema'] == 'eggpu.e1.sg.author.v1':
        for key in ('worker', 'common_worker', 'clock_helper', 'launcher', 'libstdcxx', 'setup'):
            require(isinstance(epoch.get(key), dict) and {'bytes', 'sha256'} <= epoch[key].keys(),
                    'Author epoch lacks explicit runner/runtime binding: ' + key)
        require(bool(epoch.get('build', {}).get('binary')) and
                bool(epoch.get('runtime_linker', {}).get('libraries')),
                'Author epoch lacks executable or resolved runtime bindings')
    else:
        require(bool(epoch.get('files')), 'Epoch lacks runner/binary file bindings')
    # The selected epoch contains complete source/binary/package hashes. The
    # raw job must bind exactly this immutable JSON, including execution flags.
    return digest(epoch)


def verify_quality(quality, algorithm):
    require(quality.get('pass') is True and quality.get('complete') is True,
            'Complete output quality is not established')
    if algorithm == 'pr':
        require(quality.get('dtype') == 'float64', 'Common PR output contract is float64')
        for name, limit in QUALITY.items():
            value = quality.get(name.removeprefix('pr_'))
            require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= limit,
                    'PR external quality threshold failed: ' + name)
    elif algorithm == 'wcc':
        require(quality.get('partition_equivalent') is True, 'WCC partition equivalence missing')
    else:
        require(quality.get('exact_reference_match') is True and quality.get('unreachable_match') is True,
                'Exact distance/unreachable reference check missing')


def seconds(value):
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0, 'Complete time must be finite and positive')
    return float(value)


def source_result(evidence, item):
    value = item['result']
    # Some GUM snapshots embed the result; others bind the child JSON.
    return evidence.read(value) if 'path' in value and 'sha256' in value else value


def read_source(evidence, child, schema, algorithm):
    require(child.get('status') == 'PASS' and child.get('quality_pass') is True,
            'Successful job contains an incomplete source')
    if schema == GUM_SCHEMA:
        rounds = child['rounds']
        require(len(rounds) == 3 and [r['round'] for r in rounds] == [0, 1, 2],
                'GUM requires initial, warmup, ready complete rounds')
        for row in rounds:
            verify_quality(row['quality'], algorithm)
            evidence.verify(row['raw_output'])
        require(child.get('contiguous_canonical_to_first_result') is True and
                child['continuous_first']['status'] == 'PASS_CONTINUOUS_MONOTONIC_INTERVAL',
                'GUM T_first is not one continuous complete interval')
        return {m: seconds(child['T_' + m + '_seconds']) for m in METRICS}
    calls = child['calls']
    require(set(calls) == {'first', 'warmup', 'ready'}, 'Exactly one first/warmup/ready call required')
    for call in calls.values():
        verify_quality(call['quality'], algorithm)
        evidence.verify(call['output'])
    if schema in CONTINUOUS_SCHEMAS:
        require(calls['first']['continuous_interval']['status'] == 'PASS_CONTINUOUS_MONOTONIC_INTERVAL',
                'Author T_first must use the continuous clock, not a sum of phases')
    return {m: seconds(calls[m]['seconds']) for m in METRICS}


def read_run(evidence, reference, spec, epochs, domains, mode):
    raw = evidence.read(reference)
    require(raw.get('experiment', 'E1') == 'E1', 'An E1 selection cannot silently consume another experiment')
    dataset = raw['dataset']; algorithm = raw.get('algorithm', raw.get('function'))
    require(dataset in DATASETS and algorithm in spec['algorithms'], 'Run outside declared applicability')
    require(raw.get('system') == spec['system'] and raw.get('variant') == spec.get('variant'),
            'System/author variant differs from frozen selection')
    require(raw.get('gpu_count', raw.get('count')) == spec['gpu_count'], 'GPU resource panel differs')
    entry = spec['algorithms'][algorithm]
    require(raw.get('schema') == entry['schema'], 'Runner schema differs')
    require(raw.get('protocol', entry['protocol']) == entry['protocol'], 'Run protocol differs')
    repeat = raw['repeat']
    require(type(repeat) is int and repeat in REPEATS, 'Repeat ID must be 0..9')
    allowed = ('formal',) if mode == 'formal' else ('target', 'diagnostic')
    require(raw.get('phase') in allowed, 'Qualification/diagnostic cannot enter formal statistics')
    domain = domains[dataset]
    require(raw.get('input_sha256') == domain['sha256'] and raw.get('directed') == domain['directed'],
            'Canonical input identity or direction differs')
    expected_epoch = epochs[algorithm]
    require(raw.get('epoch_sha256') == expected_epoch, 'Mixed or unbound source/runner epoch')
    require(digest(evidence.read(raw['epoch'])) == expected_epoch, 'Raw epoch content differs')
    result = dict(dataset=dataset, algorithm=algorithm, repeat=repeat, raw_status=raw.get('status'),
                  reference=reference, valid=False, error=raw.get('error'), sources=[])
    if raw.get('status') in FAILURES:
        result['status'] = raw['status']
        return result
    good_status = ('PASS', 'PASS_FORMAL') if mode == 'formal' else ('PASS', 'PASS_QUALIFICATION')
    require(raw.get('status') in good_status and raw.get('quality_pass') is True, 'Invalid successful-job status')
    if mode == 'formal':
        if entry['schema'] in CALL_SCHEMAS - {'cugraph-mg-host-csr-complete-v1'}:
            require(raw.get('performance_eligible') is True, 'Formal job was not declared performance eligible')
        if entry['schema'] == GUM_SCHEMA:
            require(raw.get('formal') is True, 'GUM formal marker missing')
        proofs = raw.get('qualification', {})
        for kind in ('tiny', 'target'):
            proof = evidence.read(raw.get(kind + '_proof', proofs.get(kind)))
            require(proof.get('status') == 'PASS_QUALIFICATION' and proof.get('phase') == kind and proof.get('quality_pass') is True
                    and proof.get('epoch_sha256') == expected_epoch,
                    'Formal job lacks matching tiny/target evidence')
            require(proof.get('system') == spec['system'] and
                    proof.get('algorithm', proof.get('function')) == algorithm and
                    proof.get('gpu_count', proof.get('count')) == spec['gpu_count'] and
                    proof.get('directed') == domain['directed'],
                    'Qualification algorithm/system/resource/direction differs')
            require(digest(evidence.read(proof['epoch'])) == expected_epoch, 'Qualification epoch changed')
            require(proof.get('dataset') == (dataset if kind == 'target' else
                    'TINY_D' if domain['directed'] else 'TINY_U'), 'Qualification input scope differs')
            if kind == 'target':
                require(proof.get('input_sha256') == domain['sha256'], 'Target proof binds another input')
            proof_sources = proof['sources']
            require(len(proof_sources) == (3 if algorithm in ('bfs', 'sssp') else 1),
                    'Qualification source domain is incomplete')
            for source_proof in proof_sources:
                child_proof = source_result(evidence, source_proof)
                require(child_proof.get('source') == source_proof.get('source'),
                        'Qualification source child differs')
                read_source(evidence, child_proof, entry['schema'], algorithm)
    rows = raw['sources']
    expected_sources = domain['sources'] if algorithm in ('bfs', 'sssp') else [None]
    require(len(rows) == len(expected_sources), 'Source count does not match the common task')
    if len(expected_sources) == 3:
        require([x['source'] for x in rows] == expected_sources, 'Three source IDs/order differ')
    else:
        require(rows[0].get('source') in (None, 0), 'Unexpected source for source-free function')
    for item in rows:
        child = source_result(evidence, item)
        require(child.get('source') == item.get('source'), 'Source child differs from parent')
        if 'input_sha256' in child:
            require(child['input_sha256'] == domain['sha256'], 'Child input identity differs')
        timings = read_source(evidence, child, entry['schema'], algorithm)
        result['sources'].append(dict(source=item.get('source'), **timings))
    result.update(valid=True, status='QUALIFIED_FORMAL' if mode == 'formal' else 'QUALIFICATION_ONLY')
    for metric in METRICS:
        result[metric] = statistics.mean(row[metric] for row in result['sources'])
        if 'T_' + metric + '_seconds' in raw:
            require(math.isclose(result[metric], raw['T_' + metric + '_seconds'], rel_tol=1e-12),
                    'Stored mean does not equal the three-source arithmetic mean')
    return result


def mean_ci(values):
    from scipy.stats import t
    require(len(values) == 10, 'A formal cell needs exactly ten independent repeat means')
    mean = statistics.mean(values); variance = statistics.variance(values)
    radius = float(t.ppf(.975, 9)) * math.sqrt(variance / 10)
    return dict(n=10, mean=mean, sd=math.sqrt(variance), ci95=[mean - radius, mean + radius])


def welch_difference(baseline, eggpu):
    """Baseline minus EGGPU; exploratory after choosing the smallest baseline mean."""
    from scipy.stats import t
    a, b = statistics.variance(baseline) / 10, statistics.variance(eggpu) / 10
    delta = statistics.mean(baseline) - statistics.mean(eggpu)
    df = (a + b) ** 2 / (a * a / 9 + b * b / 9) if a + b else None
    radius = float(t.ppf(.975, df)) * math.sqrt(a + b) if df else 0.
    return dict(mean=delta, ci95=[delta - radius, delta + radius], degrees_of_freedom=df,
                interpretation='exploratory, conditional on selected fastest mean; not multiplicity-adjusted')


def summarize(selection_path, mode):
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text())
    require(selection['schema'] == 'eggpu.e1.selection.v1', 'Unknown selection schema')
    require(selection['status'] == 'FROZEN_EXPLICIT_SELECTION', 'Explicit frozen selection required')
    require(selection['mode'] == mode, 'Selection mode differs; do not promote qualification data')
    evidence = Evidence(selection_path.parent, selection.get('path_remap', []))
    freeze = evidence.read(selection['final_eggpu_release'])
    require(digest(freeze['source_files']) == freeze['source_tree_sha256'], 'Final release manifest digest differs')
    domains = input_domains(evidence, selection['input_registry'])
    systems = selection['systems']; ids = [x['id'] for x in systems]
    require(len(set(ids)) == len(ids) and bool(ids), 'Unique explicit system IDs required')
    panels = sorted({x['gpu_count'] for x in systems})
    require(set(panels) <= {1, 8}, 'E1 panels support SG and P8 only')
    for count in panels:
        own = [s for s in systems if s['gpu_count'] == count and s['role'] == 'eggpu']
        require(len(own) == 1 and set(own[0]['algorithms']) == set(ALGORITHMS),
                'Each panel needs one preselected EGGPU route for all four functions')
    cells = {}; run_index = {}; source_rows = []
    for spec in systems:
        require(spec['role'] in ('eggpu', 'external'), 'Unknown comparison role')
        require(set(spec['algorithms']) | set(spec['not_applicable']) == set(ALGORITHMS)
                and not set(spec['algorithms']) & set(spec['not_applicable']),
                'Every algorithm must have predeclared applicability or an explained exclusion')
        require(all(isinstance(reason, str) and reason.strip() for reason in spec['not_applicable'].values()),
                'Unsupported semantics need an explicit reason')
        epochs = {}
        for algorithm, entry in spec['algorithms'].items():
            require(entry['schema'] in CALL_SCHEMAS | {GUM_SCHEMA}, 'Unknown runner schema')
            epochs[algorithm] = validate_epoch(evidence, entry['epoch'], dict(spec, **entry, algorithm=algorithm), freeze)
        selected = {}
        for ref in spec['runs']:
            run = read_run(evidence, ref, spec, epochs, domains, mode)
            key = (run['dataset'], run['algorithm'], run['repeat'])
            require(key not in selected, 'Duplicate repeat or attempts; cannot select the faster run')
            require(ref['sha256'] not in run_index, 'The same raw run was selected twice')
            selected[key] = run; run_index[ref['sha256']] = run
            for source in run['sources']:
                row = dict(run)
                row.update(system_id=spec['id'], gpu_count=spec['gpu_count'], **source, sources=None)
                source_rows.append(row)
        terminals = {}
        for terminal in spec.get('terminal_cells', []):
            ref = terminal['evidence']; evidence.verify(ref)
            require(ref['sha256'] in run_index, 'Terminal evidence must be an explicitly selected failed run')
            run = run_index[ref['sha256']]
            require(not run['valid'] and run['status'] in FAILURES and terminal['reason'].strip(),
                    'A terminal cell needs a bound failure and explanation')
            key = (run['dataset'], run['algorithm'])
            require(key not in terminals, 'Duplicate terminal declaration')
            require(ref in spec['runs'], 'Terminal failure belongs to another system')
            terminals[key] = terminal
        for algorithm in ALGORITHMS:
            for dataset in DATASETS:
                key = (spec['id'], dataset, algorithm)
                cell = dict(system_id=spec['id'], label=spec['label'], role=spec['role'],
                            gpu_count=spec['gpu_count'], dataset=dataset, algorithm=algorithm)
                if algorithm not in spec['algorithms']:
                    cell.update(status='NOT_APPLICABLE', reason=spec['not_applicable'][algorithm],
                                resolved=True, qualified=False, runs=[])
                else:
                    runs = [selected[(dataset, algorithm, i)] for i in REPEATS if (dataset, algorithm, i) in selected]
                    valid = [r for r in runs if r['valid']]
                    full = len(valid) == 10 and mode == 'formal'
                    terminal = terminals.get((dataset, algorithm))
                    failed = [r['status'] for r in runs if not r['valid']]
                    cell.update(status='COMPLETE' if full else failed[0] if failed else
                                'QUALIFICATION_ONLY' if valid and mode != 'formal' else 'PENDING',
                                resolved=full or len(runs) == 10 or terminal is not None,
                                qualified=full, selected_repeats=len(runs), valid_repeats=len(valid),
                                terminal=terminal, runs=runs, failure_statuses=failed)
                    if full:
                        cell['metrics'] = {m: mean_ci([r[m] for r in runs]) for m in METRICS}
                cells[key] = cell
    comparisons = []; totals = []
    for count in panels:
        own = next(s for s in systems if s['gpu_count'] == count and s['role'] == 'eggpu')
        external = [s for s in systems if s['gpu_count'] == count and s['role'] == 'external']
        for metric in METRICS:
            for algorithm in ALGORITHMS:
                for dataset in DATASETS:
                    e = cells[(own['id'], dataset, algorithm)]
                    candidates = [cells[(s['id'], dataset, algorithm)] for s in external if algorithm in s['algorithms']]
                    valid = [x for x in candidates if x['qualified']]
                    row = dict(gpu_count=count, metric=metric, dataset=dataset, algorithm=algorithm,
                        applicable_external=len(candidates), qualified_external=len(valid),
                        strict_full_baseline_coverage=bool(candidates) and all(c['resolved'] for c in candidates),
                        all_external_qualified=bool(candidates) and len(valid) == len(candidates),
                        paired=e['qualified'] and bool(valid), mean_leads=False)
                    if row['paired']:
                        best = min(valid, key=lambda c: (c['metrics'][metric]['mean'], c['system_id']))
                        bm = best['metrics'][metric]['mean']; em = e['metrics'][metric]['mean']
                        delta = welch_difference([r[metric] for r in best['runs']], [r[metric] for r in e['runs']])
                        row.update(fastest_qualified_measured=best['system_id'],
                                   cofastest_qualified_measured=[c['system_id'] for c in valid if c['metrics'][metric]['mean']==bm], baseline_mean=bm,
                                   eggpu_mean=em, speedup=bm/em, mean_leads=em < bm,
                                   exploratory_difference=delta, difference_ci_above_zero=delta['ci95'][0] > 0)
                    comparisons.append(row)
            rows = [r for r in comparisons if r['gpu_count'] == count and r['metric'] == metric]
            wins = sum(r['mean_leads'] for r in rows); paired = sum(r['paired'] for r in rows)
            totals.append(dict(gpu_count=count, metric=metric, planned_cells=28, paired_cells=paired,
                mean_leads=wins, leads_over_paired=wins/paired if paired else None,
                leads_over_28=wins/28, valid_coverage=paired/28,
                strict_full_baseline_coverage=sum(r['strict_full_baseline_coverage'] for r in rows)/28,
                all_external_qualified_coverage=sum(r['all_external_qualified'] for r in rows)/28))
    breakdowns=[]
    for count in panels:
        for metric in METRICS:
            for dimension, names in [('dataset',DATASETS),('algorithm',ALGORITHMS)]:
                for name in names:
                    rows=[r for r in comparisons if r['gpu_count']==count and r['metric']==metric and r[dimension]==name]
                    wins=sum(r['mean_leads'] for r in rows);paired=sum(r['paired'] for r in rows)
                    breakdowns.append(dict(gpu_count=count,metric=metric,group=dimension,name=name,
                        planned_cells=len(rows),paired_cells=paired,mean_leads=wins,
                        leads_over_paired=wins/paired if paired else None,leads_over_planned=wins/len(rows),
                        valid_coverage=paired/len(rows),strict_full_baseline_coverage=sum(r['strict_full_baseline_coverage'] for r in rows)/len(rows)))
    paper_ready = (mode == 'formal' and all(c['resolved'] for c in cells.values())
                   and any(c['qualified'] for c in cells.values()))
    return dict(schema='eggpu.e1.summary.v1', mode=mode, selection_sha256=digest(selection),
        final_source_tree_sha256=freeze['source_tree_sha256'], paper_ready=paper_ready,
        watermark='QUALIFICATION — NOT PAPER RESULTS' if mode != 'formal' else
                  'INCOMPLETE — NOT PAPER RESULTS' if not paper_ready else None,
        fixed_denominator=28, datasets=list(DATASETS), algorithms=list(ALGORITHMS),
        ci_method='95% Student-t per-system mean; independent-group Welch difference; exploratory after baseline selection',
        systems=systems, cells=list(cells.values()), comparisons=comparisons, totals=totals, breakdowns=breakdowns,
        source_rows=source_rows, verified_evidence=evidence.verified)


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def latex(value):
    return str(value).replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')


def render(summary, output):
    output = Path(output)
    require(not (summary['mode'] != 'formal' and any(p.lower() in ('paper', 'current', 'formal') for p in output.parts)),
            'Qualification output may not enter paper/current/formal directories')
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'selection.sha256'
    if marker.exists():
        require(marker.read_text().strip() == summary['selection_sha256'], 'Output belongs to another selection; no silent replacement')
    else:
        require(not any(output.iterdir()), 'Output directory must be empty or bound to the identical selection')
        marker.write_text(summary['selection_sha256'] + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    write_csv(output/'coverage.csv', summary['totals'], list(summary['totals'][0]))
    write_csv(output/'breakdowns.csv', summary['breakdowns'], list(summary['breakdowns'][0]))
    write_csv(output/'comparisons.csv', summary['comparisons'],
        ['gpu_count','metric','dataset','algorithm','paired','applicable_external','qualified_external',
         'strict_full_baseline_coverage','fastest_qualified_measured','eggpu_mean','baseline_mean','speedup','mean_leads'])
    write_csv(output/'sources.csv', summary['source_rows'],
              ['system_id','gpu_count','dataset','algorithm','repeat','source','first','ready','status'])
    rows=[]
    for c in summary['cells']:
        for metric in METRICS:
            stats=c.get('metrics',{}).get(metric,{})
            rows.append(dict(c,metric=metric,**stats))
    write_csv(output/'cells.csv',rows,['system_id','gpu_count','dataset','algorithm','metric','status',
        'selected_repeats','valid_repeats','n','mean','sd','ci95'])
    title = summary['watermark'] or 'FORMAL E1 RESULTS'
    body = ['<!doctype html><meta charset="utf-8"><title>E1 checks</title>',
        '<style>body{font:14px sans-serif;color:#222}h1{font-size:32px;color:#9b1c1c}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px}.eggpu{background:#eaf0f5}</style>',
        '<h1>'+html.escape(title)+'</h1><p>Fixed denominator: 28 per panel and timing boundary.</p>',
        '<table><tr><th>GPUs</th><th>System</th><th>Function</th><th>Graph</th><th>Status</th><th>Valid repeats</th></tr>']
    for c in summary['cells']:
        body.append('<tr class="'+c['role']+'">'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            [c['gpu_count'],c['label'],c['algorithm'].upper(),c['dataset'],c['status'],c.get('valid_repeats',0)])+'</tr>')
    body.append('</table>');(output/'checks.html').write_text('\n'.join(body))
    if summary['mode'] != 'formal' or not summary['paper_ready']:
        return
    for metric in METRICS:
        lines=['% Generated only after explicit final selection and resolved ten-repeat cells.',
               '\\begin{tabular}{lll'+'r'*7+'}',
               'GPUs & Function & System & '+' & '.join(DATASETS)+' \\\\','\\hline']
        for spec in summary['systems']:
            for algorithm in ALGORITHMS:
                if algorithm not in spec['algorithms']:
                    continue
                values=[]
                for dataset in DATASETS:
                    c=next(x for x in summary['cells'] if x['system_id']==spec['id'] and x['algorithm']==algorithm and x['dataset']==dataset)
                    if not c['qualified']:
                        values.append(latex(c['status']));continue
                    mean=c['metrics'][metric]['mean']
                    competitors=[x['metrics'][metric]['mean'] for x in summary['cells'] if
                        x['gpu_count']==spec['gpu_count'] and x['algorithm']==algorithm and x['dataset']==dataset and x['qualified']]
                    value=f'{mean:.3g}'
                    if len(competitors)>1 and mean==min(competitors):value='\\textbf{'+value+'}'
                    values.append(value)
                lines.append(f"{spec['gpu_count']} & {algorithm.upper()} & {latex(spec['label'])} & "+' & '.join(values)+' \\\\')
        lines.extend(['\\end{tabular}',
            '% Time (s); arithmetic mean of ten independent repeat blocks. BFS/SSSP: three fixed sources averaged within each block.',
            '% Bold: lowest qualified measured mean. Mean CI and uncorrected exploratory Welch differences: cells.csv and summary.json.',
            '% NOT_APPLICABLE: different semantics; OOM: memory exhaustion; TIMEOUT: timeout; INDEX_UNSUPPORTED: index domain; FAIL_QUALITY: output quality; FAILED: retained execution error.'])
        (output/f'e1_overall_{metric}.tex').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--mode',choices=('qualification','formal'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=summarize(args.selection,args.mode);render(summary,args.output)
    print(json.dumps({k:summary[k] for k in ('mode','paper_ready','watermark','totals')},indent=2))


if __name__ == '__main__':
    main()
