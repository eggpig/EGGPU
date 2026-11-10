// Same partition plan and kernels as cooperative SSSP. A single fixed worker
// executes each complete rank phase before advancing to the next phase.
void exchange_serial_relaxations(
    const std::shared_ptr<SharedSsspExecution>& shared,
    const PartitionedSsspPlan& plan,
    std::uint64_t bucket,
    double delta,
    bool enqueue,
    WorkerStreams& streams) {
    const std::size_t count = plan.partitions.size();
    // Every sender finishes producing its messages before any receiver applies
    // them. Applying rank 0 early would change the light-round propagation.
    for (std::size_t rank = 0; rank < count; ++rank) {
        RankState& state = shared->ranks[rank];
        const auto begin = SteadyClock::now();
        compact_routes(state.view->view(), *state.workspace, streams, &state.timing, state.kernel_policy.control);
        copy_send_counts(shared, rank, *state.workspace, streams, &state);
        state.timing.communication.add(elapsed_seconds(begin, SteadyClock::now()));
    }
    for (std::size_t destination = 0; destination < count; ++destination) {
        RankState& state = shared->ranks[destination];
        const auto begin = SteadyClock::now();
        bool copied = false;
        CudaEventTimer timer("measure same-device SSSP route copy", state.timing.profile);
        for (std::size_t source = 0; source < count; ++source) {
            const int messages = shared->ranks[source].send_counts[destination];
            if (!messages) continue;
            if (!copied) {
                timer.start(streams.transfer());
                copied = true;
            }
            check_cuda(cudaMemcpyAsync(
                state.workspace->receive_values(source),
                shared->ranks[source].workspace->send_values(destination),
                static_cast<std::size_t>(messages) * sssp_wire_bytes(plan.wire_encoding),
                cudaMemcpyDeviceToDevice, streams.transfer()),
                "copy SSSP relaxations between partitions on one device");
        }
        if (copied) state.timing.transfer.add(timer.finish(streams.transfer()));
        state.timing.communication.add(elapsed_seconds(begin, SteadyClock::now()));
    }
    for (std::size_t rank = 0; rank < count; ++rank) {
        RankState& state = shared->ranks[rank];
        const auto begin = SteadyClock::now();
        apply_owner_relaxations(shared, plan, rank, state.view->view(),
            *state.workspace, bucket, delta, enqueue, streams, &state.timing);
        state.timing.communication.add(elapsed_seconds(begin, SteadyClock::now()));
    }
}

void run_serial_sssp_gang(
    const std::shared_ptr<SharedSsspExecution>& shared,
    const PartitionedSsspPlan& plan,
    const std::vector<SsspUploadHolder>& uploads,
    std::size_t source_rank,
    std::size_t source_local,
    double delta,
    std::uint64_t max_bucket_rounds,
    std::uint64_t max_light_rounds,
    int worker_device,
    bool profile) {
    const std::size_t count = plan.partitions.size();
    WorkerStreams& streams = gpu_easygraph::runtime::worker_streams();
    auto& registry = gpu_easygraph::runtime::partitioned_sssp_state_registry();
    for (std::size_t rank = 0; rank < count; ++rank) {
        const auto& upload = uploads[rank].get();
        if (upload.device_id != worker_device) {
            throw std::runtime_error("serial SSSP ran on wrong worker");
        }
        RankState& state = shared->ranks[rank];
        state.timing.profile = profile;
        // No cooperative-rank barriers exist in this execution mode.
        state.timing.wait.add(0.0);
        state.send_counts.assign(count, 0);
        shared->phases[rank] = "acquire_serial_resident_state";
        const auto begin = SteadyClock::now();
        state.view = registry.acquire(upload, &state.device_view_cache_hit,
            &state.shared_arrays);
        state.workspace = registry.workspace(state.view);
        state.state_upload_seconds = elapsed_seconds(begin, SteadyClock::now());
        state.resident_bytes = state.view->resident_bytes() +
            state.workspace->capacity_bytes();
    }
    for (std::size_t rank = 0; rank < count; ++rank) {
        RankState& state = shared->ranks[rank];
        const auto& view = state.view->view();
        auto& workspace = *state.workspace;
        shared->phases[rank] = "serial_initialize_delta_state";
        if (!view.owned_vertex_count) continue;
        CudaEventTimer timer("measure serial SSSP initialization", profile);
        timer.start(streams.compute());
        gpu_easygraph::partitioned::initialize_sssp_delta(
            workspace.distances(), workspace.settled(), workspace.r_flags(),
            view.owned_vertex_count,
            rank == source_rank ? static_cast<int>(source_local) : -1,
            streams.compute());
        state.timing.kernel.add(timer.finish(streams.compute()));
    }
    for (std::uint64_t round = 0; round < max_bucket_rounds; ++round) {
        const std::uint64_t bucket = shared->global_bucket;
        shared->global_frontier = 0;
        for (std::size_t rank = 0; rank < count; ++rank) {
            RankState& state = shared->ranks[rank];
            const auto& view = state.view->view();
            auto& workspace = *state.workspace;
            shared->phases[rank] = "serial_select_delta_bucket";
            if (view.owned_vertex_count) {
                check_cuda(cudaMemsetAsync(workspace.frontier_count(), 0,
                    sizeof(int), streams.compute()), "clear serial SSSP selected count");
                check_cuda(cudaMemsetAsync(workspace.r_flags(), 0,
                    view.owned_vertex_count * sizeof(int), streams.compute()),
                    "clear serial SSSP R flags");
                CudaEventTimer timer("measure serial SSSP bucket selection", profile);
                timer.start(streams.compute());
                gpu_easygraph::partitioned::select_sssp_bucket_frontier(
                    workspace.distances(), workspace.settled(),
                    workspace.current_frontier(), workspace.frontier_count(),
                    view.owned_vertex_count, bucket, delta, streams.compute(), state.kernel_policy.control);
                state.timing.kernel.add(timer.finish(streams.compute()));
                copy_frontier_count(shared, rank, workspace, streams, &state);
            } else {
                state.next_count = 0;
            }
            state.current_count = state.next_count;
            shared->global_frontier += state.current_count;
        }
        if (!shared->global_frontier) {
            throw std::runtime_error("selected serial SSSP bucket is globally empty");
        }
        std::uint64_t light_round = 0;
        while (shared->global_frontier) {
            if (++light_round > max_light_rounds) {
                throw std::runtime_error("serial SSSP light-edge closure exceeded the correctness guard");
            }
            for (std::size_t rank = 0; rank < count; ++rank) {
                RankState& state = shared->ranks[rank];
                shared->phases[rank] = "serial_light_relaxation_and_precombine";
                reset_relaxation_round(state.view->view(), *state.workspace,
                    streams, &state.timing);
                relax_edges(state.view->view(), *state.workspace, &state,
                    bucket, delta, true, streams);
            }
            exchange_serial_relaxations(shared, plan, bucket, delta, true, streams);
            shared->global_frontier = 0;
            for (std::size_t rank = 0; rank < count; ++rank) {
                RankState& state = shared->ranks[rank];
                auto& workspace = *state.workspace;
                compact_vertex_flags(workspace.next_flags(),
                    workspace.next_frontier(), state.view->view().owned_vertex_count,
                    workspace, streams, &state.timing, state.kernel_policy.control);
                copy_frontier_count(shared, rank, workspace, streams, &state);
                shared->global_frontier += state.next_count;
            }
            ++shared->light_rounds;
            for (auto& state : shared->ranks) {
                state.current_count = static_cast<std::size_t>(state.next_count);
                state.workspace->advance_frontier();
            }
        }
        for (std::size_t rank = 0; rank < count; ++rank) {
            RankState& state = shared->ranks[rank];
            const auto& view = state.view->view();
            auto& workspace = *state.workspace;
            shared->phases[rank] = "serial_heavy_relaxation";
            compact_vertex_flags(workspace.r_flags(), workspace.current_frontier(),
                view.owned_vertex_count, workspace, streams, &state.timing, state.kernel_policy.control);
            copy_frontier_count(shared, rank, workspace, streams, &state);
            state.current_count = state.next_count;
            reset_relaxation_round(view, workspace, streams, &state.timing);
            relax_edges(view, workspace, &state, bucket, delta, false, streams);
        }
        exchange_serial_relaxations(shared, plan, bucket, delta, false, streams);
        for (std::size_t rank = 0; rank < count; ++rank) {
            RankState& state = shared->ranks[rank];
            const auto& view = state.view->view();
            if (!view.owned_vertex_count) continue;
            CudaEventTimer timer("measure serial SSSP settle bucket", profile);
            timer.start(streams.compute());
            gpu_easygraph::partitioned::settle_sssp_r_vertices(
                state.workspace->r_flags(), state.workspace->settled(),
                view.owned_vertex_count, streams.compute());
            state.timing.kernel.add(timer.finish(streams.compute()));
        }
        shared->global_bucket = gpu_easygraph::partitioned::kSsspNoBucket;
        for (std::size_t rank = 0; rank < count; ++rank) {
            RankState& state = shared->ranks[rank];
            shared->phases[rank] = "serial_global_empty_bucket";
            find_local_minimum_bucket(shared, rank, state.view->view(),
                *state.workspace, delta, streams, &state);
            shared->global_bucket = std::min(shared->global_bucket,
                state.local_minimum_bucket);
        }
        ++shared->bucket_rounds;
        shared->converged = shared->global_bucket == gpu_easygraph::partitioned::kSsspNoBucket;
        if (shared->converged) break;
    }
    if (!shared->converged) {
        throw std::runtime_error("serial partitioned SSSP exceeded max_bucket_rounds");
    }
    for (std::size_t rank = 0; rank < count; ++rank) {
        RankState& state = shared->ranks[rank];
        const std::size_t owned = state.view->view().owned_vertex_count;
        shared->phases[rank] = "serial_output_gather";
        if (owned) {
            CudaEventTimer timer("measure serial SSSP output gather", profile);
            timer.start(streams.transfer());
            check_cuda(cudaMemcpyAsync(shared->output_buffers[rank]->data(),
                state.workspace->distances(), owned * sizeof(double),
                cudaMemcpyDeviceToHost, streams.transfer()),
                "gather serial partitioned SSSP distances");
            state.timing.output_gather.add(timer.finish(streams.transfer()));
            const auto* observed = static_cast<const double*>(shared->output_buffers[rank]->data());
            assemble_sssp_owner_output(plan.partitions[rank], *shared, state, observed);
        }
        shared->phases[rank] = "complete";
    }
}
