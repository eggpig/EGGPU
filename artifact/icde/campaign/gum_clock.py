"""Observe existing GUM CLOCK_MONOTONIC host-complete markers without changing C++."""
from pathlib import Path
import json,os,time
import sg_worker as shared

SOURCES={
 'bfs':('adapters/bfs_wcc/gum_output.cu','017e9ef420290d0e3ab9d3a1401b7597e45974b370f5d1c7d6e0ddcb3a9862a6'),
 'wcc':('adapters/bfs_wcc/gum_output.cu','017e9ef420290d0e3ab9d3a1401b7597e45974b370f5d1c7d6e0ddcb3a9862a6'),
 'pr':('adapters/pr/gum_stage2.cu','7678128a2a3ceb03573235ab166a9945951f303e52353dea2de63f32fa667884'),
 'sssp':('adapters/sssp/gum_sssp_standard_output.cu','5fb050d17484469e503f7c68e1d713f447e08f01803824b3fe0f39c7733dd4ac')}
HEADER_SHA='275647ff48152b7bf5c41b4d5898cac3f4c16125bf94aef8bc9406be520a292f'


def clock_domain():
    # exec preserves the time namespace; same boot+namespace+offsets establish
    # that Python clock_gettime_ns and the existing C++ clock_gettime agree.
    offsets=Path('/proc/self/timens_offsets')
    return dict(clock='CLOCK_MONOTONIC',boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        time_namespace=os.readlink('/proc/self/ns/time'),
        time_namespace_for_children=os.readlink('/proc/self/ns/time_for_children'),
        time_offsets=offsets.read_text() if offsets.exists() else None,
        resolution_seconds=time.clock_getres(time.CLOCK_MONOTONIC))


def verify_source(artifact,function):
    name,expected=SOURCES[function];root=Path(artifact['source'])
    adapter=shared.binding(root/name);header=shared.binding((root/name).parent/'baseline_memory_clock.h')
    if adapter['sha256']!=expected or header['sha256']!=HEADER_SHA:
        raise ValueError('Unreviewed GUM complete-output monotonic marker source')
    return dict(adapter=adapter,clock_header=header,
        endpoint='Existing first query memory_query_end after complete dense host scatter and MPI completion barrier',
        cpp_changed=False,algorithm_changed=False,binary_rebuilt=False)


def first_result(folder,count,start_ns,parent_end_ns,parent_domain):
    if type(start_ns) is not int or type(parent_end_ns) is not int or not 0<start_ns<parent_end_ns:
        raise ValueError('Invalid parent monotonic measurement bracket')
    folder=Path(folder);ranks=[]
    for rank in range(count):
        launch_path=folder/f'launch-rank-{rank}.json';launch=json.loads(launch_path.read_text())
        if launch['rank']!=rank or launch['size']!=count or launch.get('clock_domain')!=parent_domain:
            raise ValueError('GUM rank does not share parent CLOCK_MONOTONIC domain')
        if launch.get('parent_monotonic_start_ns')!=start_ns or not start_ns<=launch['monotonic_launch_ns']<=parent_end_ns:
            raise ValueError('Rank launch is outside the parent clock bracket')
        marker=folder/f'memory-markers-rank-{rank}.jsonl'
        opened={};intervals=[];seen=set()
        for line in marker.read_text().splitlines():
            row=json.loads(line);identity=row['id']
            if row.get('pid')!=launch['pid'] or type(row.get('t_ns')) is not int:
                raise ValueError('Native marker PID/clock is not the exec-bound rank')
            if row['kind']=='begin':
                if identity in seen or identity in opened or row.get('rank')!=rank:raise ValueError('Invalid/duplicate marker begin')
                opened[identity]=row;seen.add(identity)
            elif row['kind']=='end':
                if identity not in opened:raise ValueError('Native marker end lacks unique begin')
                begin=opened.pop(identity)
                if not launch['monotonic_launch_ns']<=begin['t_ns']<=row['t_ns']<=parent_end_ns:
                    raise ValueError('Native monotonic interval outside its launch/parent bracket')
                intervals.append(dict(phase=begin['phase'],round=begin['query_id'],begin_ns=begin['t_ns'],end_ns=row['t_ns']))
            else:raise ValueError('Unknown monotonic marker event')
        if opened:raise ValueError('Unclosed native intervals')
        queries=[i for i in intervals if i['phase'] in ('warmup_query','measured_query')]
        if [(i['round'],i['phase']) for i in queries]!=[(0,'warmup_query'),(1,'warmup_query'),(2,'measured_query')]:
            raise ValueError('Missing ordered three complete native query markers')
        ranks.append(dict(rank=rank,launch=shared.binding(launch_path),markers=shared.binding(marker),
            first_query_begin_ns=queries[0]['begin_ns'],first_query_end_ns=queries[0]['end_ns']))
    # One interval from one parent origin to the latest required rank's complete
    # event, not a sum of durations or of phase maxima. Rank zero owns full n.
    end_ns=max(row['first_query_end_ns'] for row in ranks)
    return dict(status='PASS_CONTINUOUS_MONOTONIC_INTERVAL',clock_domain=parent_domain,
        start_ns=start_ns,end_ns=end_ns,parent_process_end_ns=parent_end_ns,seconds=(end_ns-start_ns)/1e9,
        rank_zero_complete_ns=ranks[0]['first_query_end_ns'],ranks=ranks,
        start_boundary='Before canonical weak/descriptor preparation and MPI process startup',
        end_boundary='Latest rank first-query completion event, after rank-zero full owned dense host output and MPI barrier',
        disk_output_and_python_audit_excluded=True,phase_durations_summed=False,
        contains_weak_descriptor_mpi_startup_native_preparation_and_full_host_output=True)
