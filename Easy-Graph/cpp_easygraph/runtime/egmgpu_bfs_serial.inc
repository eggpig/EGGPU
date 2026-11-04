// One physical GPU executes the original owner partitions in global BFS
// levels. Complete remote candidate exchange is retained within that device.
py::dict run_serial_partitioned_bfs(
    const std::shared_ptr<MultiPartitionGraph>& graph, std::uint64_t source,
    std::uint64_t max_iterations, const std::string& frontier_policy_text,
    int device, double dense_vertex_enter, double dense_vertex_exit,
    double dense_edge_enter, double dense_edge_exit, double dense_boundary_enter) {
    if (!graph || !graph->vertex_count())
        throw std::invalid_argument("serial-partition BFS requires a nonempty graph");
    const auto status = gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running || std::find(status.device_ids.begin(), status.device_ids.end(), device) == status.device_ids.end())
        throw std::invalid_argument("serial-partition BFS device outside active runtime");
    std::vector<int> virtual_devices;
    for (const auto& part : graph->partitions()) virtual_devices.push_back(part.device_id);
    // Upload ownership ranks are virtual. Their group describes the original
    // partition layout; device_id describes the single actual placement.
    if (std::find(virtual_devices.begin(), virtual_devices.end(), device) == virtual_devices.end())
        throw std::invalid_argument("serial BFS target must name one original logical partition device");
    const auto policy = gpu_easygraph::runtime::parse_bfs_frontier_policy(frontier_policy_text);
    BfsFrontierThresholds thresholds;
    thresholds.dense_vertex_enter=dense_vertex_enter; thresholds.dense_vertex_exit=dense_vertex_exit;
    thresholds.dense_edge_enter=dense_edge_enter; thresholds.dense_edge_exit=dense_edge_exit;
    thresholds.dense_boundary_enter=dense_boundary_enter;
    gpu_easygraph::runtime::validate_bfs_frontier_thresholds(thresholds);
    const char* edge_setting = std::getenv("EGMGPU_BFS_EDGE_SCHEDULE");
    const std::string edge_schedule = edge_setting ? edge_setting : "warp";
    if (edge_schedule != "warp" && edge_schedule != "thread")
        throw std::invalid_argument("EGMGPU_BFS_EDGE_SCHEDULE must be thread or warp");
    const int sparse_ctas_per_row = bfs_sparse_ctas_per_row(edge_schedule);
    const bool profile = bfs_profile_setting("EGMGPU_BFS_SERIAL_PROFILE",
        bfs_profile_setting("EGMGPU_BFS_PROFILE", false));
    const bool direct_assembly = bfs_direct_result_assembly();
    const BfsHostAssembly host_assembly = bfs_host_assembly(true, direct_assembly);
    const bool core_output_team16 = owner_output::enabled(true);
    if (core_output_team16 && host_assembly != BfsHostAssembly::Workers)
        throw std::invalid_argument("serial BFS team16 requires worker host assembly");
    // One actual GPU worker executes every virtual owner sequentially.
    const int core_output_width = core_output_team16 ? owner_output::requested_width(1U) : 1;
    const auto core_output_placement = std::make_shared<const owner_output::CpuPlacement>(
        owner_output::parse_placement(core_output_team16, 1U));
    const bool numa_local_output = bfs_numa_local_output();
    const BfsDenseDirection dense_direction = bfs_dense_direction(*graph);
    auto begin = SteadyClock::now();
    gpu_easygraph::runtime::PartitionedHostPlanCache<PartitionedBfsPlan>::AcquireResult acquired;
    {
        py::gil_scoped_release release;
        const auto key = gpu_easygraph::runtime::make_partitioned_host_plan_cache_key(
            gpu_easygraph::runtime::PartitionedHostPlanAlgorithm::Bfs,
            graph->graph_id(), graph->generation(), graph->view_signature(),
            graph->topology_signature(), graph->weighted(), graph->weight_signature(),
            graph->vertex_count(), graph->adjacency_count(), virtual_devices,
            "serial-direct-device" + std::to_string(device));
        acquired = gpu_easygraph::runtime::partitioned_bfs_host_plan_cache().acquire(key, [&]() {
            auto prepared = std::make_shared<PartitionedBfsPlan>(
                gpu_easygraph::runtime::build_partitioned_bfs_plan(host_partitions(*graph), graph->vertex_count()));
            for (std::size_t rank=0;rank<virtual_devices.size();++rank) {
                auto upload=make_upload(*graph,*prepared,virtual_devices,rank);
                upload.device_id=device; upload.view_signature += ":serial-colocated";
                prepared->prepared_uploads.push_back(std::make_shared<BfsPartitionUpload>(std::move(upload)));
            }
            return std::shared_ptr<const PartitionedBfsPlan>(prepared);
        });
    }
    const double preparation_wall=elapsed_seconds(begin,SteadyClock::now());
    const auto plan=acquired.plan;
    const std::size_t count=plan->partitions.size(), source_rank=plan->owner_rank(source), source_local=plan->owner_local(source);
    if (!max_iterations) max_iterations=graph->vertex_count();
    begin=SteadyClock::now();
    std::shared_ptr<BfsPinnedHostState> pinned;
    std::unique_lock<std::mutex> lease;
    bool pinned_hit=false;
    {
        py::gil_scoped_release release;
        pinned=std::static_pointer_cast<BfsPinnedHostState>(
            gpu_easygraph::runtime::runtime_manager().acquire_resident_resource(
                "bfs:serial:pinned:"+std::to_string(device)+(numa_local_output?":numa-local-output":":caller-output"),
                [](){return std::shared_ptr<void>(new BfsPinnedHostState());},&pinned_hit));
        lease=std::unique_lock<std::mutex>(pinned->lease_mutex);pinned->ensure(*plan,false,numa_local_output,device);
    }
    const double pinned_wall=elapsed_seconds(begin,SteadyClock::now());
    begin=SteadyClock::now();
    auto assembled_distances=allocate_bfs_host_output(plan->vertex_count,host_assembly);
    const double host_result_allocation_seconds=elapsed_seconds(begin,SteadyClock::now());
    const auto task_preparation_begin=SteadyClock::now();
    auto shared=std::make_shared<SharedBfsExecution>(*plan,pinned,policy,thresholds);
    shared->host_assembly=host_assembly;
    shared->core_output_team16=core_output_team16;
    shared->core_output_width=core_output_width;
    shared->core_output_workers=1U;
    shared->core_output_placement=core_output_placement;
    shared->host_usage_profile=bfs_profile_setting("EGMGPU_BFS_HOST_USAGE_PROFILE",false);
    shared->worker_gpu_numa=bfs_worker_gpu_numa();
    shared->assembled_distances=std::move(assembled_distances);
    auto wall=std::make_shared<std::map<std::string,double> >();
    auto events=std::make_shared<std::map<std::string,double> >();
    for(auto& part:shared->ranks){part.warp_edges=edge_schedule=="warp";part.sparse_ctas_per_row=sparse_ctas_per_row;part.direct_routes=true;part.move_output=true;part.trim_barriers=true;}
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.push_back([shared,plan,wall,events,count,source_rank,source_local,max_iterations,profile,dense_direction](int worker_device){
        if(shared->host_usage_profile)shared->serial_worker_usage.start();
        gpu_easygraph::runtime::run_with_gpu_worker_cpu_placement(
            worker_device,shared->worker_gpu_numa,shared->host_usage_profile,&shared->serial_cpu_placement,[&](){
        using gpu_easygraph::runtime::DevicePhaseTimer;
        auto& registry=gpu_easygraph::runtime::partitioned_bfs_state_registry();
        auto start=SteadyClock::now();
        for(std::size_t rank=0;rank<count;++rank){
            auto& part=shared->ranks[rank];
            part.view=registry.acquire(*plan->prepared_uploads[rank],&part.device_view_cache_hit,&part.shared_arrays);
            part.workspace=registry.workspace(part.view);
            if(part.sparse_ctas_per_row)part.workspace->ensure_high_degree(part.view->view());
            part.high_degree_workspace_bytes=part.workspace->high_degree_capacity_bytes();
            part.resident_bytes=part.view->resident_bytes()+part.workspace->capacity_bytes();
        }
        (*wall)["state_acquire"]=elapsed_seconds(start,SteadyClock::now());
        const auto stream=gpu_easygraph::runtime::worker_streams().compute();
        DevicePhaseTimer init_timer(profile),scan_timer(profile),combine_timer(profile),control_timer(profile),output_timer(profile);
        init_timer.begin(stream);start=SteadyClock::now();
        for(std::size_t rank=0;rank<count;++rank){
            auto& part=shared->ranks[rank];const auto& view=part.view->view();auto& work=*part.workspace;
            if(view.outgoing_route_offsets.back())check_cuda(cudaMemsetAsync(work.send_flags(),0,
                view.outgoing_route_offsets.back()*sizeof(int),stream),"clear serial BFS remote seen");
            gpu_easygraph::partitioned::initialize_bfs_frontiers(work.distances(),work.current_frontier(),
                work.current_bitmap(),view.owned_vertex_count,rank==source_rank?static_cast<int>(source_local):-1,stream);
            part.current_count=rank==source_rank?1U:0U;
            part.reached_count=part.current_count;
        }
        init_timer.end(stream);(*wall)["initialization_submission"]=elapsed_seconds(start,SteadyClock::now());
        for(std::uint64_t iteration=1;iteration<=max_iterations;++iteration){
            std::size_t active=0;double active_edges=0;
            const auto metrics=frontier_metrics(*plan,shared,&active,&active_edges);
            const auto kind=gpu_easygraph::runtime::select_bfs_frontier_kind(shared->policy,metrics,shared->thresholds,shared->previous_kind);
            shared->previous_kind=kind;
            SharedBfsExecution::FrontierDecision decision;decision.iteration=iteration;decision.kind=kind;decision.metrics=metrics;decision.active_vertices=active;decision.estimated_active_edges=active_edges;shared->decisions.push_back(decision);
            start=SteadyClock::now();scan_timer.begin(stream);
            // Convert every owner before any pull replica reads a peer's
            // current bitmap. All operations use this same ordered stream.
            for(auto& part:shared->ranks){
                const auto& view=part.view->view();auto& work=*part.workspace;
                if(part.have_last_kind&&part.last_kind!=kind&&kind==BfsFrontierKind::Dense)
                    gpu_easygraph::partitioned::build_bfs_frontier_bitmap(work.current_frontier(),part.current_count,work.current_bitmap(),view.owned_vertex_count,stream);
                part.last_kind=kind;part.have_last_kind=true;
            }
            if(kind==BfsFrontierKind::Dense && dense_direction.pull){
                for(std::size_t destination=0;destination<count;++destination){
                    auto& work=*shared->ranks[destination].workspace;
                    for(std::size_t source=0;source<count;++source){
                        const auto words=work.dense_send_word_count(source);
                        if(source==destination||!words)continue;
                        check_cuda(cudaMemcpyAsync(work.dense_send_bitmap(source),shared->ranks[source].workspace->current_bitmap(),
                            words*sizeof(std::uint32_t),cudaMemcpyDeviceToDevice,stream),"serial BFS immutable pull frontier replica");
                    }
                }
            }
            for(std::size_t rank=0;rank<count;++rank){
                auto& part=shared->ranks[rank];const auto& view=part.view->view();auto& work=*part.workspace;
                if(work.next_count())check_cuda(cudaMemsetAsync(work.next_count(),0,sizeof(int),stream),"clear serial BFS next count");
                if(kind==BfsFrontierKind::Sparse){
                    check_cuda(cudaMemsetAsync(work.send_counts(),0,count*sizeof(int),stream),"clear serial BFS send counts");
                    gpu_easygraph::partitioned::expand_bfs_sparse(view.offsets,view.indices,view.owners,view.halo_route_slots,
                        work.distances(),work.current_frontier(),part.current_count,work.next_frontier(),work.next_count(),work.send_flags(),
                        view.owned_vertex_count,view.partition_id,iteration,stream,part.warp_edges,view.route_destination_ranks,
                        view.route_owner_local_ids,view.outgoing_route_offsets_device,work.send_values(0),work.send_counts(),
                        part.sparse_ctas_per_row,work.high_degree_rows(),work.high_degree_control(),work.high_degree_grid_limit());
                }else{
                    const auto words=gpu_easygraph::runtime::bfs_bitmap_word_count(view.owned_vertex_count);
                    if(words)check_cuda(cudaMemsetAsync(work.next_bitmap(),0,words*sizeof(std::uint32_t),stream),"clear serial BFS next bitmap");
                    if(dense_direction.pull){
                        gpu_easygraph::partitioned::expand_bfs_pull(view.offsets,view.indices,view.owners,view.halo_route_slots,
                            view.route_destination_ranks,view.route_owner_local_ids,view.outgoing_dense_word_offsets_device,
                            work.distances(),work.current_bitmap(),work.dense_send_bitmap(0),work.next_bitmap(),work.next_frontier(),
                            work.next_count(),view.owned_vertex_count,view.partition_id,iteration,stream,dense_direction.lanes);
                        continue;
                    }
                    const auto outgoing=view.outgoing_dense_word_offsets.back();
                    if(outgoing)check_cuda(cudaMemsetAsync(work.dense_send_bitmap(0),0,outgoing*sizeof(std::uint32_t),stream),"clear serial BFS outgoing bitmap");
                    if(part.current_count)gpu_easygraph::partitioned::expand_bfs_dense(view.offsets,view.indices,view.owners,view.halo_route_slots,
                        view.route_destination_ranks,view.route_owner_local_ids,view.outgoing_dense_word_offsets_device,work.distances(),
                        work.current_bitmap(),work.next_bitmap(),work.next_frontier(),work.next_count(),work.dense_send_bitmap(0),
                        view.owned_vertex_count,view.partition_id,iteration,stream,part.warp_edges,work.send_flags());
                }
            }
            scan_timer.end(stream);(*wall)["expansion_submission"]+=elapsed_seconds(start,SteadyClock::now());
            start=SteadyClock::now();combine_timer.begin(stream);
            for(std::size_t destination=0;destination<count && !(kind==BfsFrontierKind::Dense && dense_direction.pull);++destination){
                auto& part=shared->ranks[destination];const auto& view=part.view->view();auto& work=*part.workspace;
                for(std::size_t source=0;source<count;++source){
                    auto& sender=*shared->ranks[source].workspace;
                    if(kind==BfsFrontierKind::Sparse){
                        gpu_easygraph::partitioned::apply_bfs_owner_candidates_device_count(sender.send_values(destination),
                            sender.send_counts()+destination,plan->route(source,destination).capacity(),work.distances(),
                            work.next_frontier(),work.next_count(),view.owned_vertex_count,iteration,stream);
                    }else{
                        gpu_easygraph::partitioned::apply_bfs_owner_bitmap(sender.dense_send_bitmap(destination),sender.dense_send_word_count(destination),
                            work.distances(),work.next_bitmap(),work.next_frontier(),work.next_count(),view.owned_vertex_count,iteration,stream);
                    }
                }
            }
            combine_timer.end(stream);(*wall)["owner_combine_submission"]+=elapsed_seconds(start,SteadyClock::now());
            start=SteadyClock::now();control_timer.begin(stream);
            for(std::size_t rank=0;rank<count;++rank){
                auto& part=shared->ranks[rank];int* target=static_cast<int*>(shared->host_counts[rank]->data());*target=0;
                if(part.workspace->next_count())check_cuda(cudaMemcpyAsync(target,part.workspace->next_count(),sizeof(int),cudaMemcpyDeviceToHost,stream),"serial BFS complete layer count");
            }
            control_timer.end(stream);check_cuda(cudaStreamSynchronize(stream),"serial BFS complete global layer");
            (*wall)["layer_count_and_preceding_work_completion"]+=elapsed_seconds(start,SteadyClock::now());
            if(iteration==1)(*events)["initialization"]=init_timer.seconds();
            (*events)["expansion"]+=scan_timer.seconds();(*events)["owner_combine"]+=combine_timer.seconds();(*events)["layer_count_d2h"]+=control_timer.seconds();
            start=SteadyClock::now();shared->global_active=0;
            for(std::size_t rank=0;rank<count;++rank){
                auto& part=shared->ranks[rank];part.next_count=*static_cast<const int*>(shared->host_counts[rank]->data());
                if(part.next_count<0||static_cast<std::size_t>(part.next_count)>part.view->view().owned_vertex_count)
                    throw std::runtime_error("invalid serial BFS next count");
                shared->global_active+=static_cast<std::size_t>(part.next_count);
                part.reached_count+=static_cast<std::uint64_t>(part.next_count);
            }
            shared->iterations=iteration;shared->converged=shared->global_active==0;
            for(auto& part:shared->ranks){part.current_count=static_cast<std::size_t>(part.next_count);part.workspace->advance_frontier();}
            (*wall)["global_termination"]+=elapsed_seconds(start,SteadyClock::now());
            if(shared->converged)break;
        }
        if(shared->host_usage_profile){
            shared->serial_before_output_usage.begin=shared->serial_worker_usage.begin;
            shared->serial_before_output_usage.finish();
            shared->serial_output_usage.begin=shared->serial_before_output_usage.end;
        }
        start=SteadyClock::now();output_timer.begin(stream);
        for(std::size_t rank=0;rank<count;++rank){
            auto& part=shared->ranks[rank];const auto owned=part.view->view().owned_vertex_count;
            if(owned)check_cuda(cudaMemcpyAsync(shared->output_buffers[rank]->data(),part.workspace->distances(),owned*sizeof(std::int64_t),cudaMemcpyDeviceToHost,stream),"serial BFS complete distances");
        }
        output_timer.end(stream);check_cuda(cudaStreamSynchronize(stream),"serial BFS complete output");
        (*wall)["output_d2h"]=elapsed_seconds(start,SteadyClock::now());(*events)["output_d2h"]=output_timer.seconds();
        if(shared->host_usage_profile)shared->serial_output_usage.finish();
        if(shared->host_assembly==BfsHostAssembly::Workers){
            start=SteadyClock::now();
            for(std::size_t rank=0;rank<count;++rank)assemble_bfs_rank_output(*plan,shared,rank);
            (*wall)["host_owner_assembly"]=elapsed_seconds(start,SteadyClock::now());
        }
        });
        if(shared->host_usage_profile)shared->serial_worker_usage.finish();
    });
    const double host_task_preparation_seconds=elapsed_seconds(task_preparation_begin,SteadyClock::now());
    begin=SteadyClock::now();
    {py::gil_scoped_release release;gpu_easygraph::runtime::runtime_manager().execute_gang(std::vector<int>(1U,device),std::move(tasks));}
    const double execution_wall=elapsed_seconds(begin,SteadyClock::now());
    if(!shared->converged)throw std::runtime_error("serial-partition BFS did not finish within max_iterations");
    NativeBfsResult native;
    {py::gil_scoped_release release;native=merge_bfs_result(*plan,shared,true,direct_assembly);}
    begin=SteadyClock::now();
    py::dict out;out["values_dense"]=move_bfs_result_array(&native);
    out["python_materialization_seconds"]=elapsed_seconds(begin,SteadyClock::now());out["native_merge_seconds"]=native.merge_seconds;
    out["state_preparation_seconds"]=preparation_wall;out["host_pinned_preparation_seconds"]=pinned_wall;out["host_pinned_cache_hit"]=pinned_hit;
    out["host_plan_cache_hit"]=acquired.cache_hit;out["device_processing_seconds"]=execution_wall;
    out["execution_mode"]="serial_partitions_one_gpu";out["partition_count"]=count;out["physical_gpu_count"]=1;out["devices_used"]=std::vector<int>(1U,device);
    out["source"]=source;out["iterations"]=shared->iterations;out["converged"]=shared->converged;out["reachable_vertex_count"]=native.reachable;
    out["unreachable_vertex_count"]=graph->vertex_count()-native.reachable;out["unreachable_distance"]=-1;
    out["route_mode"]="direct";out["output_mode"]="move";out["sync_mode"]="one_layer_completion";out["edge_schedule"]=edge_schedule;
    out["sparse_schedule"]=bfs_sparse_schedule_name(sparse_ctas_per_row);
    out["high_degree_threshold"]=gpu_easygraph::partitioned::kBfsHighDegreeThreshold;
    out["result_assembly"]=native.direct_assembly?"direct":"zeroed";
    out["host_assembly"]=bfs_host_assembly_name(host_assembly);
    out["core_output_team_version"]="stage4_core_owner_output_v1";
    out["core_output_team_policy"]=core_output_team16?"team16":"original";
    out["core_output_workers"]=1U;
    out["core_output_requested_per_worker"]=core_output_width;
    out["core_output_cpu_placement"]=owner_output::placement_to_python(*core_output_placement);
    py::list core_output_observations;
    for(const auto& part:shared->ranks)
        core_output_observations.append(owner_output::to_python(part.core_output_observation));
    out["core_output_observations"]=std::move(core_output_observations);
    out["host_result_allocation_seconds"]=host_result_allocation_seconds;
    out["host_task_preparation_seconds"]=host_task_preparation_seconds;
    out["host_assembly_timing_scope"]=host_assembly==BfsHostAssembly::Workers
        ?"one physical worker visits owners sequentially; selected CPU team copies/normalizes each owner including first-touch, nested in device_processing wall"
        :"caller sequential owner copy/normalization including first-touch, nested in native_merge wall";
    out["device_processing_timing_scope"]="runtime gang wall including GPU execution, D2H and selected worker CPU output assembly";
    out["output_pinned_placement"]=numa_local_output?"numa_local":"caller";
    out["dense_direction"]=dense_direction.pull?"pull":"push";out["pull_lanes"]=dense_direction.lanes;
    out["serial_expansion_timing_scope"]=dense_direction.pull?"frontier conversion, same-device frontier replication and local expansion":"frontier conversion and local expansion";
    out["frontier_policy"]=frontier_policy_text;out["communication_backend"]="same_device_owner_queues";out["inter_gpu_transfer_bytes"]=0;
    out["complete_output_d2h_bytes"]=graph->vertex_count()*sizeof(std::int64_t);
    out["device_view_cache_hit"]=native.device_view_cache_hit;out["aggregate_resident_device_bytes"]=native.resident_bytes;
    py::list cache_parts;for(std::size_t rank=0;rank<count;++rank){const auto& part=shared->ranks[rank];py::dict item;
        item["partition_rank"]=rank;item["physical_device"]=device;item["device_view_cache_hit"]=part.device_view_cache_hit;
        item["shared_array_hits"]=part.shared_arrays.hits;item["shared_array_new_bytes"]=part.shared_arrays.new_bytes;
        item["shared_array_retained_bytes"]=part.shared_arrays.retained_bytes;item["resident_device_bytes"]=part.resident_bytes;
        item["high_degree_workspace_bytes"]=part.high_degree_workspace_bytes;
        item["host_assembly_seconds"]=part.host_assembly_seconds;
        if(shared->host_usage_profile)item["host_owner_assembly_usage"]=bfs_usage_window_dict(part.host_assembly_usage);
        item["reachable_owned_vertex_count"]=part.reached_count;cache_parts.append(item);}
    out["partition_cache_statistics"]=cache_parts;
    out["host_usage_profile_enabled"]=shared->host_usage_profile;
    out["worker_cpu_placement_policy"]=shared->worker_gpu_numa?"gpu_numa":"none";
    out["serial_worker_cpu_placement"]=bfs_worker_cpu_placement_dict(shared->serial_cpu_placement);
    out["worker_cpu_placement_scope"]="one actual GPU worker task, current allowed mask intersect actual GPU NUMA CPUs; restore before task return; caller assembly is outside this policy";
    if(shared->host_usage_profile){
        py::dict usage;usage["worker_total"]=bfs_usage_window_dict(shared->serial_worker_usage);
        usage["before_output"]=bfs_usage_window_dict(shared->serial_before_output_usage);
        usage["output_gather"]=bfs_usage_window_dict(shared->serial_output_usage);
        out["serial_worker_thread_usage"]=std::move(usage);
        out["host_usage_scope"]="one actual worker, whole worker reported once; logical-rank host owner windows are sequential, nested only in workers mode; caller mode is on another thread after worker completion; wall minus CPU is unattributed blocking/wait/descheduling";
        out["diagnostic_performance_eligible"]=false;
    }
    out["serial_partition_wall_seconds"]=*wall;out["serial_partition_device_seconds"]=profile?py::object(py::cast(*events)):py::none();out["profile_enabled"]=profile;
    py::list trace;for(const auto& d:shared->decisions){py::dict t;t["iteration"]=d.iteration;t["frontier_kind"]=gpu_easygraph::runtime::bfs_frontier_kind_name(d.kind);t["active_vertices"]=d.active_vertices;trace.append(t);}out["frontier_trace"]=trace;
    return out;
}
