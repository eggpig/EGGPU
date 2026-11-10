// Same destination partitions and CUDA operators, one physical GPU and one
// stream. Each iteration consumes the previous complete global rank vector;
// partitions are never treated as disconnected independent graphs.
py::dict run_serial_partitioned_pagerank(
    const std::shared_ptr<MultiPartitionGraph>& graph, double alpha,
    int max_iterations, double tolerance, int device) {
    if (!graph || graph->vertex_count() == 0 || !(alpha > 0.0 && alpha < 1.0) ||
        !std::isfinite(alpha) || max_iterations <= 0 || !(tolerance > 0.0) || !std::isfinite(tolerance))
        throw std::invalid_argument("invalid serial-partition PageRank parameters");
    const bool output_team16 = easygraph::egmgpu::owner_output::enabled();
    const int output_team_width = output_team16
        ? easygraph::egmgpu::owner_output::requested_width(1U) : 1;
    const auto status = gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running || std::find(status.device_ids.begin(), status.device_ids.end(), device) == status.device_ids.end())
        throw std::invalid_argument("serial-partition device outside active runtime");
    std::vector<int> virtual_devices;
    for (const auto& part : graph->partitions()) virtual_devices.push_back(part.device_id);
    bool host_hit = false;
    auto begin = SteadyClock::now();
    std::shared_ptr<PreparedPullState> prepared;
    {
        py::gil_scoped_release release;
        prepared = prepare_pull_state(graph, virtual_devices, &host_hit, device);
    }
    const double prepare_wall = elapsed_seconds(begin, SteadyClock::now());
    const char* setting = std::getenv("EGMGPU_PAGERANK_PROFILE");
    const bool profile = setting && std::string(setting) == "1";
    struct SerialState {
        std::vector<RankState> parts;
        std::vector<double> residuals, dangling;
        double state_wall = 0.0, init_wall = 0.0, contribution_wall = 0.0;
        double halo_wall = 0.0, pull_wall = 0.0, reduction_wall = 0.0, output_wall = 0.0;
        double init_gpu = 0.0, contribution_gpu = 0.0, halo_gpu = 0.0, pull_gpu = 0.0, output_gpu = 0.0;
        double scalar_d2h_wall = 0.0, scalar_d2h_gpu = 0.0;
        double residual = 0.0;
        int iterations = 0;
        bool converged = false;
    };
    auto state = std::make_shared<SerialState>();
    const std::size_t count = prepared->partitions.size();
    state->parts.resize(count); state->residuals.resize(count); state->dangling.resize(count);
    const std::string output_mode = pagerank_output_mode();
    const bool numpy_output_requested = easygraph::egmgpu::pr_numpy_output_requested();
    PageRankOutputLease output_lease;
    output_lease.acquire(output_mode, std::vector<int>(1U, device), prepared->partitions, true);
    if (output_lease.state)
        for (std::size_t rank = 0; rank < count; ++rank)
            state->parts[rank].pinned_result = output_lease.state->buffers[rank];
    const double n = static_cast<double>(graph->vertex_count());
    const auto output_allocation_begin = SteadyClock::now();
    const std::shared_ptr<PageRankHostResult> worker_result = output_team16
        ? std::make_shared<PageRankHostResult>(static_cast<std::size_t>(graph->vertex_count()), numpy_output_requested, output_mode != "copy")
        : std::shared_ptr<PageRankHostResult>();
    const double host_output_allocation_seconds = output_team16
        ? elapsed_seconds(output_allocation_begin, SteadyClock::now()) : 0.0;
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.push_back([state, prepared, profile, count, n, alpha, max_iterations, tolerance, worker_result, output_team_width](int worker_device) {
        using gpu_easygraph::runtime::DevicePhaseTimer;
        auto& registry = gpu_easygraph::runtime::partitioned_device_state_registry();
        auto start = SteadyClock::now();
        for (std::size_t rank = 0; rank < count; ++rank) {
            auto& part = state->parts[rank];
            gpu_easygraph::runtime::begin_device_state_profile(profile);
            part.view = registry.acquire(prepared->uploads[rank], &part.device_view_cache_hit, &part.shared_arrays);
            part.workspace = registry.workspace(part.view);
            part.device_state_profile = gpu_easygraph::runtime::end_device_state_profile();
        }
        state->state_wall = elapsed_seconds(start, SteadyClock::now());
        WorkerStreams& streams = gpu_easygraph::runtime::worker_streams();
        const cudaStream_t stream = streams.compute();
        DevicePhaseTimer timer(profile), contribution_timer(profile), halo_timer(profile), pull_timer(profile);
        start = SteadyClock::now(); timer.begin(stream);
        for (auto& part : state->parts)
            gpu_easygraph::partitioned::initialize_owned_rank(part.workspace->current_rank(),
                part.view->view().owned_vertex_count, 1.0 / n, stream);
        timer.end(stream); check_cuda(cudaStreamSynchronize(stream), "serial PR initialize");
        state->init_gpu = timer.seconds(); state->init_wall = elapsed_seconds(start, SteadyClock::now());
        for (int iteration = 1; iteration <= max_iterations; ++iteration) {
            start = SteadyClock::now(); contribution_timer.begin(stream);
            for (std::size_t rank = 0; rank < count; ++rank) {
                auto& part = state->parts[rank]; const auto& view = part.view->view();
                gpu_easygraph::partitioned::prepare_page_rank_contribution(
                    part.workspace->current_rank(), view.owned_out_weight,
                    part.workspace->local_contribution(), view.owned_vertex_count,
                    part.workspace->scalars(), stream);
            }
            contribution_timer.end(stream);
            state->contribution_wall += elapsed_seconds(start, SteadyClock::now());
            double total_dangling = 0.0;
            if (prepared->has_dangling) {
                start = SteadyClock::now(); timer.begin(stream);
                for (std::size_t rank = 0; rank < count; ++rank)
                    check_cuda(cudaMemcpyAsync(&state->dangling[rank], state->parts[rank].workspace->scalars(),
                        sizeof(double), cudaMemcpyDeviceToHost, stream), "serial PR dangling copy");
                timer.end(stream); check_cuda(cudaStreamSynchronize(stream), "serial PR dangling complete");
                state->scalar_d2h_gpu += timer.seconds(); state->scalar_d2h_wall += elapsed_seconds(start, SteadyClock::now());
                start = SteadyClock::now();
                for (double value : state->dangling) total_dangling += value;
                state->reduction_wall += elapsed_seconds(start, SteadyClock::now());
            }
            start = SteadyClock::now(); halo_timer.begin(stream);
            for (std::size_t rank = 0; rank < count; ++rank) {
                const auto& upload = prepared->uploads[rank];
                const auto& view = state->parts[rank].view->view();
                for (std::size_t owner = 0; owner < count; ++owner) {
                    const std::size_t first = upload.halo_owner_offsets[owner];
                    const std::size_t size = upload.halo_owner_offsets[owner + 1U] - first;
                    if (size) gpu_easygraph::partitioned::gather_peer_page_rank_halo(
                        state->parts[owner].workspace->local_contribution(), view.halo_source_local + first,
                        view.halo_destination_local + first, size,
                        state->parts[rank].workspace->local_contribution(), stream);
                }
            }
            halo_timer.end(stream);
            state->halo_wall += elapsed_seconds(start, SteadyClock::now());
            start = SteadyClock::now(); pull_timer.begin(stream);
            for (auto& part : state->parts) {
                const auto& view = part.view->view(); auto& work = *part.workspace;
                gpu_easygraph::partitioned::pull_page_rank_rows(view.offsets, view.indices, view.weights,
                    work.local_contribution(), work.current_rank(), work.next_rank(), view.owned_vertex_count,
                    alpha, (1.0 - alpha + alpha * total_dangling) / n, work.scalars() + 1U, stream,
                    view.pagerank_heavy_rows, view.pagerank_heavy_row_count);
            }
            pull_timer.end(stream);
            state->pull_wall += elapsed_seconds(start, SteadyClock::now());
            start = SteadyClock::now(); timer.begin(stream);
            for (std::size_t rank = 0; rank < count; ++rank)
                check_cuda(cudaMemcpyAsync(&state->residuals[rank], state->parts[rank].workspace->scalars() + 1U,
                    sizeof(double), cudaMemcpyDeviceToHost, stream), "serial PR residual copy");
            timer.end(stream); check_cuda(cudaStreamSynchronize(stream), "serial PR residual complete");
            // One stream already orders the operators. Synchronize only for
            // the global convergence decision (and dangling mass when needed).
            state->contribution_gpu += contribution_timer.seconds();
            state->halo_gpu += halo_timer.seconds();
            state->pull_gpu += pull_timer.seconds();
            state->scalar_d2h_gpu += timer.seconds(); state->scalar_d2h_wall += elapsed_seconds(start, SteadyClock::now());
            start = SteadyClock::now(); state->residual = 0.0;
            for (std::size_t rank = 0; rank < count; ++rank) {
                state->parts[rank].workspace->advance_rank(); state->residual += state->residuals[rank];
            }
            state->iterations = iteration; state->converged = state->residual <= n * tolerance;
            state->reduction_wall += elapsed_seconds(start, SteadyClock::now());
            if (state->converged) break;
        }
        start = SteadyClock::now(); timer.begin(stream);
        for (auto& part : state->parts) {
            const std::size_t size = part.view->view().owned_vertex_count;
            if (!part.pinned_result) part.owned_result.resize(size);
            if (size) check_cuda(cudaMemcpyAsync(part.result_data(),
                part.workspace->current_rank(), size * sizeof(double),
                cudaMemcpyDeviceToHost, stream), "serial PR complete output");
        }
        timer.end(stream); check_cuda(cudaStreamSynchronize(stream), "serial PR output complete");
        state->output_gpu = timer.seconds(); state->output_wall = elapsed_seconds(start, SteadyClock::now());
        if (worker_result) {
            // The complete single-stream D2H boundary precedes every owner copy.
            // Owners execute sequentially; each width16 team joins before reuse.
            for (std::size_t rank = 0; rank < count; ++rank) {
                auto& part = state->parts[rank];
                const auto output_begin = SteadyClock::now();
                worker_result->assign_team(prepared->partitions[rank], part.result_data(),
                    output_team_width, part.host_output_threads);
                part.host_output_assembly_seconds = elapsed_seconds(output_begin, SteadyClock::now());
            }
        }
    });
    begin = SteadyClock::now();
    { py::gil_scoped_release release;
      gpu_easygraph::runtime::runtime_manager().execute_gang(std::vector<int>(1U, device), std::move(tasks)); }
    const double execution_wall = elapsed_seconds(begin, SteadyClock::now());
    begin = SteadyClock::now();
    std::shared_ptr<PageRankHostResult> result = worker_result;
    if (!result) {
        result = std::make_shared<PageRankHostResult>(static_cast<std::size_t>(graph->vertex_count()), numpy_output_requested, output_mode != "copy");
        for (std::size_t rank = 0; rank < count; ++rank) {
            auto& part = state->parts[rank];
            const auto output_begin = SteadyClock::now();
            easygraph::egmgpu::owner_output::observe_original(part.host_output_threads);
            result->assign(prepared->partitions[rank], part.result_data());
            part.host_output_assembly_seconds = elapsed_seconds(output_begin, SteadyClock::now());
        }
    }
    const double merge_wall = elapsed_seconds(begin, SteadyClock::now());
    begin = SteadyClock::now();
    py::array_t<double> dense = result->materialize(output_mode != "copy");
    const double output_wall = elapsed_seconds(begin, SteadyClock::now());
    py::dict out;
    out["values_dense"] = std::move(dense); out["converged"] = state->converged;
    out["output_mode"] = output_mode;
    out["result_assembly"] = result->direct_assembly ? "direct" : "zeroed";
    out["pr_native_output_ownership"] = easygraph::egmgpu::pr_host_output_proof(
        numpy_output_requested, result->direct_assembly, output_mode != "copy", graph->vertex_count());
    out["host_pinned_output_prepare_seconds"] = output_lease.preparation_wall;
    out["host_pinned_output_cache_hit"] = output_lease.cache_hit;
    out["host_pinned_output_new_bytes"] = output_lease.new_bytes;
    out["host_pinned_output_placement"] = output_lease.placement;
    out["iterations"] = state->iterations; out["residual"] = state->residual;
    out["execution_mode"] = "serial_partitions_one_gpu"; out["partition_count"] = count;
    out["physical_gpu_count"] = 1; out["host_view_cache_hit"] = host_hit;
    add_host_plan_metrics(out, *prepared);
    add_device_projection_metrics(out, state->parts);
    py::list cache_hits;
    bool all_device_hits = true;
    std::uint64_t new_shared_bytes = 0;
    for (const auto& part : state->parts) {
        cache_hits.append(part.device_view_cache_hit);
        all_device_hits = all_device_hits && part.device_view_cache_hit;
        new_shared_bytes += part.shared_arrays.new_bytes;
    }
    out["partition_device_view_cache_hits"] = cache_hits;
    out["device_view_cache_hit"] = all_device_hits;
    out["shared_array_new_bytes"] = new_shared_bytes;
    out["state_preparation_seconds"] = prepare_wall; out["serial_execution_seconds"] = execution_wall;
    out["native_merge_seconds"] = merge_wall; out["python_array_construction_seconds"] = output_wall;
    py::dict wall, events;
    wall["state_acquire"] = state->state_wall; wall["initialization"] = state->init_wall;
    wall["contribution_submission"] = state->contribution_wall; wall["local_halo_submission"] = state->halo_wall;
    wall["pull_submission"] = state->pull_wall; wall["scalar_copy_and_preceding_work_completion"] = state->scalar_d2h_wall;
    wall["scalar_reduction"] = state->reduction_wall; wall["output_d2h"] = state->output_wall;
    events["initialization"] = state->init_gpu; events["contribution"] = state->contribution_gpu;
    events["local_halo"] = state->halo_gpu; events["pull"] = state->pull_gpu;
    events["scalar_d2h"] = state->scalar_d2h_gpu; events["output_d2h"] = state->output_gpu;
    out["serial_partition_wall_seconds"] = wall;
    out["serial_partition_device_seconds"] = profile ? py::object(events) : py::none();
    out["profile_enabled"] = profile;
    out["core_output_team_version"] = "stage4_core_owner_output_v1";
    out["core_output_team_policy"] = output_team16 ? "team16" : "original";
    out["core_output_workers"] = 1U;
    out["core_output_requested_per_worker"] = output_team_width;
    out["core_output_thread_budget"] = output_team_width;
    out["host_output_allocation_seconds"] = output_team16
        ? py::object(py::float_(host_output_allocation_seconds)) : py::none();
    out["host_output_timing_scope"] = output_team16
        ? "one GPU worker visits owners sequentially after complete D2H; each CPU team joins before next owner, nested in serial_execution_seconds"
        : "caller owner copies and allocation after GPU worker completion, nested in native_merge_seconds";
    py::list output_observations, output_parts;
    for (std::size_t rank = 0; rank < count; ++rank) {
        const auto& part = state->parts[rank];
        output_observations.append(easygraph::egmgpu::owner_output::to_python(part.host_output_threads));
        py::dict item;
        item["partition_rank"] = rank;
        item["physical_device"] = device;
        item["host_output_assembly_seconds"] = part.host_output_assembly_seconds;
        item["host_output_within_worker"] = output_team16;
        output_parts.append(item);
    }
    out["core_output_observations"] = output_observations;
    out["serial_partition_output_seconds"] = output_parts;
    return out;
}
