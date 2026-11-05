"""Latest E6 presentation: external compositions, two real 2x2s, then scaling."""
import math


def draw(result, out, captions, r):
    rows = [x for x in result['cells'] if x['experiment'] == 'E6']
    if not rows: return
    plt = r.pyplot()
    names = {'structural': 'Structural holes', 'ranking': 'Ranking', 'closeness': 'Closeness',
             'bfs': 'BFS', 'wcc': 'WCC', 'preparation_and_handoff': 'Preparation / handoff'}
    colors = dict(structural='#B8D3E7', ranking='#8C79A6', closeness='#BED4AA',
                  bfs='#356A9A', wcc='#6B8E67', preparation_and_handoff='#E9BE96')
    for metric in ('first', 'ready'):
        for graph in ('EP', 'LJ', 'OR'):
            selected = [x for x in rows if x['definition']['dataset'] == graph and
                        x['definition']['gpu_count'] <= 1 and x['definition']['ablate'] == 'none']
            if not selected: continue
            selected.sort(key=lambda x: ('eggpu', 'cugraph_composite', 'cugraph_igraph', 'igraph').index(x['definition']['system']))
            fig, ax = plt.subplots(figsize=(3.39, 2.65), constrained_layout=True)
            handles = {}
            for index, row in enumerate(selected):
                if not row['complete_formal']:
                    ax.text(index, .02, row['terminal_status'] or 'INCOMPLETE', rotation=90, ha='center',
                            transform=ax.get_xaxis_transform(), fontsize=6, color='#9C2D21'); continue
                bottom = 0.
                for stage, label in names.items():
                    value = row['metrics'][metric + '__' + stage]['mean']
                    handles[label] = ax.bar(index, value, bottom=bottom, color=colors[stage], edgecolor='black', linewidth=.3, width=.68)
                    bottom += value
                total = r.stat(row, metric)
                r.n.need(math.isclose(bottom, total[0], rel_tol=1e-8, abs_tol=1e-8), 'E6 sequential stage means do not sum to continuous total')
                ax.errorbar(index, total[0], yerr=[[total[1][0]], [total[1][1]]], fmt='none', color='black', capsize=2, linewidth=.8)
            labels = {'eggpu': 'EGGPU', 'cugraph_composite': 'cuGraph composite', 'cugraph_igraph': 'cuGraph + igraph', 'igraph': 'igraph'}
            ax.set_xticks(range(len(selected)), [labels[x['definition']['system']] for x in selected], rotation=25, ha='right')
            ax.set_ylabel('Application time (s)'); ax.set_title(graph); ax.grid(axis='y'); ax.set_axisbelow(True)
            if handles: fig.legend(list(handles.values()), list(handles), loc='outside upper center', ncol=2, frameon=False)
            r.figure_save(fig, out, f'Fig9_E6_{graph}_{metric}',
                'Complete dependent structural-screening workflow on one fixed weak unweighted projection. Full-graph Effective Size and Constraint select 32 candidates; candidate WF Closeness selects the BFS source; WCC completes the task. All four systems execute every stage and return complete owning outputs. CPU and GPU compositions are disclosed separately. ' +
                ('First-complete time includes projection, backend construction and handoff. ' if metric == 'first' else 'Ready time follows one excluded warmup and recomputes both dependent selections. ') +
                'Stacked stages are sequential measured intervals; the residual is preparation, handoff and control overhead. Intervals apply to the continuous total. This controlled task is not presented as an observed production trace.',
                captions, result, selected)
        for backend, backend_name in (('igraph', 'igraph CPU'), ('cugraph_composite', 'cuGraph composite GPU')):
            groups = {}
            for graph in ('EP', 'LJ', 'OR'):
                selected = [x for x in rows if x['definition']['dataset'] == graph and x['definition']['system'] == 'eggpu' and
                            x['definition']['gpu_count'] == 1 and (x['definition']['ablate'] == 'none' or
                            (x['definition']['coverage_backend'] == backend and x['definition']['ablate'] in ('structural', 'closeness', 'structural_closeness')))]
                if selected:
                    selected.sort(key=lambda x: ('none', 'structural', 'closeness', 'structural_closeness').index(x['definition']['ablate']))
                    groups[graph] = selected
            if groups:
                r.bar_panel(groups, metric, out, f'Fig9S_E6_2x2_{backend}_{metric}',
                    f'Two-factor coverage intervention with {backend_name}: independently delegate structural holes (SH: Effective Size and Constraint) and candidate closeness (CL), retaining EGGPU BFS/WCC and all causal dependencies. Native is the same P1 control in both backend panels, not an additional repeat group. External-both closes each 2×2 grid. Total timings are measured independently; backend-specific linear interaction contrasts and uncertainty are reported without pairing repeat indices or assuming a favorable sign.',
                    result, captions, key='ablate', labels={'none': 'Native', 'structural': 'External SH', 'closeness': 'External CL', 'structural_closeness': 'External SH+CL'},
                    colors={'none': r.PALETTE['blue'], 'structural': r.PALETTE['orange'], 'closeness': r.PALETTE['green'], 'structural_closeness': r.PALETTE['purple']})
        legacy = {g: [x for x in rows if x['definition']['dataset'] == g and x['definition']['system'] == 'eggpu' and
                      x['definition']['gpu_count'] == 1 and x['definition']['ablate'] in ('none', 'bfs', 'wcc')] for g in ('EP', 'LJ', 'OR')}
        legacy = {g: sorted(v, key=lambda x: ('none', 'bfs', 'wcc').index(x['definition']['ablate'])) for g, v in legacy.items() if v}
        if legacy:
            r.bar_panel(legacy, metric, out, f'Fig9S_E6_traversal_{metric}',
                'Supplementary single-group CPU substitutions for BFS or WCC. These preserve the complete workflow and are separate from the two extension-group 2×2 interventions.',
                result, captions, key='ablate', labels={'none': 'Native', 'bfs': 'CPU BFS', 'wcc': 'CPU WCC'})
        scaling = {g: [x for x in rows if x['definition']['dataset'] == g and x['definition']['system'] == 'eggpu' and
                       x['definition']['ablate'] == 'none'] for g in ('EP', 'LJ', 'OR')}
        scaling = {g: v for g, v in scaling.items() if v}
        if scaling:
            r.panels(scaling, metric, 'gpu_count', 'dataset', out, f'Fig9S_E6_{metric}_resource',
                'Resource sensitivity of the same complete EGGPU workflow. Structural-hole work and selected-source closeness use replicated whole-graph data and sharded work; BFS/WCC use logical GPU0. This does not demonstrate aggregate-memory capacity scaling.',
                result, captions, 'GPUs', r.GRAPH_COLORS)
