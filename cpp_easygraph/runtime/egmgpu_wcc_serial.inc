// Executes the SAME prepared partition CSR and local-forest kernels on a single
// worker/stream. Cross-partition equivalences are exchanged every round; these
// are never evaluated as disconnected induced-subgraph answers.
py::dict run_serial_partitioned_wcc(
    const std::shared_ptr<MultiPartitionGraph>& graph, int max_iterations,
    int device, bool dense_output) {
    if (!graph || !graph->vertex_count() || max_iterations <= 0)
        throw std::invalid_argument("invalid serial-partition WCC parameters");
    const auto status = gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running || std::find(status.device_ids.begin(), status.device_ids.end(), device) == status.device_ids.end())
        throw std::invalid_argument("serial-partition WCC device outside active runtime");
    std::vector<int> virtual_devices;
    const bool boundary_csr = wcc_boundary_csr(true);
    const WccOutputOptions output = wcc_output_options(true);
    const bool core_output_team16 = easygraph::egmgpu::owner_output::enabled(true);
    if (core_output_team16 && (!output.worker_assembly() || !output.pinned() || output.tiled()))
        throw std::invalid_argument("serial WCC team16 requires pinned owner-worker assembly; tiled serial is outside this candidate");
    const int core_output_width = core_output_team16
        ? easygraph::egmgpu::owner_output::requested_width(1U) : 1;
    const auto core_output_placement = std::make_shared<const easygraph::egmgpu::owner_output::CpuPlacement>(
        easygraph::egmgpu::owner_output::parse_placement(core_output_team16, 1U));
    for (const auto& part : graph->partitions()) virtual_devices.push_back(part.device_id);
    auto begin = SteadyClock::now();
    gpu_easygraph::runtime::PartitionedHostPlanCache<PartitionedWccPlan>::AcquireResult acquired;
    {
        py::gil_scoped_release release;
        const auto key = gpu_easygraph::runtime::make_partitioned_host_plan_cache_key(
            gpu_easygraph::runtime::PartitionedHostPlanAlgorithm::Wcc,
            graph->graph_id(), graph->generation(), graph->view_signature(),
            graph->topology_signature(), graph->weighted(), graph->weight_signature(),
            graph->vertex_count(), graph->adjacency_count(), virtual_devices,
            "serial-forest-device" + std::to_string(device) + (boundary_csr ? ":csr-boundary-v1" : "") +
                (output.tiled() ? ":host-output-tiles-v1" : ""));
        acquired = gpu_easygraph::runtime::partitioned_wcc_host_plan_cache().acquire(key, [&]() {
            auto prepared = std::make_shared<PartitionedWccPlan>(
                gpu_easygraph::runtime::build_partitioned_wcc_plan(host_partitions(*graph), graph->vertex_count()));
            if (output.tiled()) gpu_easygraph::runtime::prepare_wcc_output_gather_index(*prepared);
            for (std::size_t rank = 0; rank < virtual_devices.size(); ++rank) {
                auto upload = make_upload(*graph, *prepared, virtual_devices, rank, true, boundary_csr);
                upload.device_id = device;
                upload.device_group.assign(1U, device);
                upload.view_signature += ":serial";
                prepared->prepared_uploads.push_back(std::make_shared<WccPartitionUpload>(std::move(upload)));
            }
            return std::shared_ptr<const PartitionedWccPlan>(prepared);
        });
    }
    const double prepare_wall = elapsed_seconds(begin, SteadyClock::now());
    const auto plan = acquired.plan;
    const bool unique_local_edges = wcc_unique_local_edges(true);
    WccHostOutputLease output_lease;
    { py::gil_scoped_release release; output_lease.acquire(std::vector<int>(1U, device), output.pinned()); }
    const char* profile_setting = std::getenv("EGMGPU_WCC_SERIAL_PROFILE");
    const bool profile = profile_setting && std::string(profile_setting) == "1";
    const std::size_t count = plan->partitions.size();
    auto shared = std::make_shared<SharedWccExecution>(count, *plan, false);
    // P ownership ranks are sequential tasks of one actual worker, not P teams.
    shared->core_output_team16 = core_output_team16;
    shared->core_output_width = core_output_width;
    shared->core_output_workers = 1U;
    shared->core_output_placement = core_output_placement;
    if (output.worker_assembly()) { py::gil_scoped_release release; prepare_wcc_worker_result(*plan, *shared, output); }
    auto wall = std::make_shared<std::map<std::string, double> >();
    auto events = std::make_shared<std::map<std::string, double> >();
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.push_back([plan, shared, wall, events, count, max_iterations, profile, output, unique_local_edges](int worker_device) {
        auto& registry = gpu_easygraph::runtime::partitioned_wcc_state_registry();
        auto start = SteadyClock::now();
        for (std::size_t rank = 0; rank < count; ++rank) {
            auto& part = shared->ranks[rank];
            part.view = registry.acquire(*plan->prepared_uploads[rank], &part.device_view_cache_hit, &part.shared_arrays);
            part.workspace = registry.workspace(part.view);
            prepare_wcc_host_output(part, output);
            part.resident_bytes = part.view->resident_bytes() + part.workspace->capacity_bytes();
        }
        (*wall)["state_acquire"] = elapsed_seconds(start, SteadyClock::now());
        const auto stream = gpu_easygraph::runtime::worker_streams().compute();
        auto phase = [&](const std::string& name, const std::function<void()>& operation) {
            const auto phase_begin = SteadyClock::now();
            if (profile) {
                CudaEventTimer timer(name.c_str());
                timer.start(stream);
                operation();
                (*events)[name] += timer.finish(stream);
            } else {
                operation();
            }
            (*wall)[name] += elapsed_seconds(phase_begin, SteadyClock::now());
        };
        phase("initialization", [&]() {
            for (auto& part : shared->ranks) {
                const auto& view = part.view->view(); auto& work = *part.workspace;
                gpu_easygraph::partitioned::initialize_wcc_labels(view.local_to_global, work.labels(), view.local_vertex_count, stream);
                gpu_easygraph::partitioned::initialize_wcc_local_parents(work.local_parents(), view.owned_vertex_count, stream);
            }
        });
        phase("local_union", [&]() {
            for (auto& part : shared->ranks) {
                const auto& view = part.view->view();
                gpu_easygraph::partitioned::union_wcc_local_edges(view.offsets, view.indices,
                    part.workspace->local_parents(), view.owned_vertex_count, stream, unique_local_edges);
            }
        });
        phase("forest_compression", [&]() {
            for (auto& part : shared->ranks)
                gpu_easygraph::partitioned::compress_wcc_local_parents(part.workspace->local_parents(), part.view->view().owned_vertex_count, stream);
        });
        auto project = [&]() {
            for (auto& part : shared->ranks)
                gpu_easygraph::partitioned::project_wcc_component_labels(part.workspace->local_parents(),
                    part.workspace->labels(), part.view->view().owned_vertex_count, stream);
        };
        phase("component_projection", project);
        for (int iteration = 1; iteration <= max_iterations; ++iteration) {
            phase("changed_clear", [&]() {
                for (auto& part : shared->ranks) gpu_easygraph::partitioned::clear_wcc_changed(part.workspace->changed(), stream);
            });
            phase("owner_pack", [&]() {
                for (auto& part : shared->ranks) {
                    const auto& view = part.view->view(); auto& work = *part.workspace;
                    for (std::size_t destination = 0; destination < count; ++destination)
                        gpu_easygraph::partitioned::pack_wcc_owner_labels(work.labels(), view.outgoing_owner_local_ids[destination],
                            work.send_buffer(destination), view.outgoing_counts[destination], stream);
                }
            });
            phase("local_halo_scatter", [&]() {
                for (std::size_t destination = 0; destination < count; ++destination) {
                    auto& part = shared->ranks[destination]; const auto& view = part.view->view();
                    for (std::size_t source = 0; source < count; ++source)
                        gpu_easygraph::partitioned::scatter_wcc_halo_labels(shared->ranks[source].workspace->send_buffer(destination),
                            view.incoming_halo_local_ids[source], part.workspace->labels(), view.incoming_counts[source], stream);
                }
            });
            phase("boundary_combine", [&]() {
                for (auto& part : shared->ranks) {
                    const auto& view = part.view->view(); auto& work = *part.workspace;
                    combine_forest_boundary(view, work, stream);
                }
            });
            phase("component_projection", project);
            phase("changed_d2h", [&]() {
                for (auto& part : shared->ranks) {
                    part.host_changed = 0;
                    if (part.view->view().owned_vertex_count)
                        check_cuda(cudaMemcpyAsync(&part.host_changed, part.workspace->changed(), sizeof(int), cudaMemcpyDeviceToHost, stream),
                            "copy serial WCC changed flag");
                }
            });
            start = SteadyClock::now();
            check_cuda(cudaStreamSynchronize(stream), "serial WCC global changed ready");
            (*wall)["changed_completion"] += elapsed_seconds(start, SteadyClock::now());
            start = SteadyClock::now();
            shared->any_changed = false;
            for (const auto& part : shared->ranks) shared->any_changed |= part.host_changed != 0;
            shared->iterations = iteration; shared->converged = !shared->any_changed;
            (*wall)["changed_reduction"] += elapsed_seconds(start, SteadyClock::now());
            if (shared->converged) break;
        }
        phase("output_d2h", [&]() {
            for (auto& part : shared->ranks) {
                const auto size = part.view->view().owned_vertex_count;
                if (!size) continue;
                if (!output.pinned()) part.owned_labels.resize(size);
                void* target = output.pinned() ? part.pinned_labels->data() : part.owned_labels.data();
                check_cuda(cudaMemcpyAsync(target, part.workspace->labels(),
                    size * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream), "serial WCC complete output");
            }
        });
        start = SteadyClock::now();
        check_cuda(cudaStreamSynchronize(stream), "serial WCC complete output ready");
        (*wall)["output_completion"] += elapsed_seconds(start, SteadyClock::now());
        if (shared->worker_labels && shared->converged) {
            start = SteadyClock::now();
            // One actual worker executes all virtual partitions, after the
            // single stream's complete D2H boundary. No CUDA timing events.
            if (output.tiled()) {
                for (std::size_t rank = 0; rank < count; ++rank)
                    shared->output_sources[rank] = shared->ranks[rank].labels_data();
                // One physical worker writes the entire contiguous result
                // only after the single stream completed every rank D2H.
                assemble_wcc_tiled_result(*plan, *shared, shared->ranks[0], 0, 1);
            } else {
                for (std::size_t rank = 0; rank < count; ++rank)
                    assemble_wcc_worker_result(plan->partitions[rank], *shared, shared->ranks[rank]);
            }
            (*wall)["host_owner_assembly"] += elapsed_seconds(start, SteadyClock::now());
        }
    });
    begin = SteadyClock::now();
    { py::gil_scoped_release release;
      gpu_easygraph::runtime::runtime_manager().execute_gang(std::vector<int>(1U, device), std::move(tasks)); }
    const double execution_wall = elapsed_seconds(begin, SteadyClock::now());
    if (!shared->converged) throw std::runtime_error("serial-partition WCC did not converge within max_iterations");
    NativeWccResult native;
    { py::gil_scoped_release release;
      native = merge_wcc_result(*plan, shared, dense_output, output.canonical(), output.direct());
      output_lease.release(); }
    begin = SteadyClock::now();
    py::dict out;
    if (dense_output) {
        out["labels_dense"] = materialize_wcc_labels(native);
    } else {
        py::list components;
        for (const auto& values : native.components) {
            py::set component;
            for (auto vertex : values) component.add(py::int_(vertex));
            components.append(component);
        }
        out["components"] = components;
    }
    out["python_materialization_seconds"] = elapsed_seconds(begin, SteadyClock::now());
    out["state_preparation_seconds"] = prepare_wall; out["host_plan_cache_hit"] = acquired.cache_hit;
    out["device_processing_seconds"] = execution_wall; out["native_merge_seconds"] = native.merge_seconds;
    out["execution_mode"] = "serial_partitions_one_gpu"; out["wcc_algorithm"] = "forest";
    out["result_format"] = dense_output ? "dense_labels" : "components";
    out["host_output_mode"] = output.transfer;
    out["host_output_pinned_placement"] = output.pinned() ? output.placement : "not_applicable";
    out["result_merge_mode"] = output.merge;
    append_wcc_assembly_metadata(out, *shared, output, *plan, acquired.builder);
    out["host_output_tiled_readiness_barriers"] = 0;
    out["boundary_representation"] = boundary_csr ? "csr" : "coo";
    out["local_union_mode"] = unique_local_edges ? "unique_edges" : "all_arcs";
    out["result_ownership"] = "independent complete host labels; retained across queries and runtime shutdown";
    out["host_output_lease_wait_seconds"] = output_lease.wait_seconds;
    out["partition_count"] = count; out["physical_gpu_count"] = 1; out["devices_used"] = std::vector<int>(1U, device);
    out["iterations"] = shared->iterations; out["converged"] = shared->converged;
    out["serial_partition_host_phase_seconds"] = *wall;
    out["serial_partition_device_seconds"] = profile ? py::cast(*events) : py::none();
    out["profile_enabled"] = profile;
    out["phase_host_timing_semantics"] = profile
        ? "Host wall through event completion per phase; diagnostic mode adds intermediate synchronization."
        : "Host enqueue intervals; changed_completion/output_completion are explicit stream waits. Not GPU phase durations.";
    out["logical_communication_bytes"] = static_cast<std::uint64_t>(shared->iterations) * plan->halo_value_count * sizeof(std::uint64_t);
    out["physical_transfer_bytes"] = 0; out["communication_backend"] = "same_device_scatter";
    out["aggregate_resident_device_bytes"] = native.resident_bytes;
    py::list per_partition;
    for (std::size_t rank = 0; rank < count; ++rank) {
        const auto& part = shared->ranks[rank];
        py::dict item;
        item["partition_id"] = rank;
        append_wcc_output_metadata(item, part);
        item["physical_device_id"] = device;
        item["device_view_cache_hit"] = part.device_view_cache_hit;
        item["shared_array_hits"] = part.shared_arrays.hits;
        item["shared_array_new_bytes"] = part.shared_arrays.new_bytes;
        item["resident_device_bytes"] = part.resident_bytes;
        item["boundary_edge_count"] = part.view->view().boundary_edge_count;
        per_partition.append(item);
    }
    out["per_partition"] = per_partition;
    return out;
}
