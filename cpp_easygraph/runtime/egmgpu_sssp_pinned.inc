// Runtime-owned writable scratch. A calling thread holds lease_mutex until
// every accepted worker has finished D2H and copied its result into query-owned
// vectors. No Python result aliases this state.
struct SsspPinnedHostState {
    std::mutex lease_mutex;
    std::vector<std::shared_ptr<PinnedHostBuffer> > host_payloads;
    std::vector<std::shared_ptr<PinnedHostBuffer> > host_counts;
    std::vector<std::shared_ptr<PinnedHostBuffer> > host_minimum_buckets;
    std::vector<std::shared_ptr<PinnedHostBuffer> > output_buffers;
    std::vector<gpu_easygraph::runtime::GpuLocalPinnedAllocationInfo> output_placement;
    std::size_t allocation_count = 0;
    std::size_t retained_bytes = 0;

    void grow(std::shared_ptr<PinnedHostBuffer>& buffer, std::size_t bytes) {
        if (!bytes || (buffer && buffer->size() >= bytes)) return;
        const std::size_t previous = buffer ? buffer->size() : 0U;
        auto replacement = std::make_shared<PinnedHostBuffer>(bytes);
        buffer = std::move(replacement);
        retained_bytes += bytes - previous;
        ++allocation_count;
    }

    void ensure(const PartitionedSsspPlan& plan, bool staging,
        bool numa_local_output = false, int colocated_device = -1) {
        const std::size_t count = plan.partitions.size();
        // Fixed count/backend in the cache key: resize cannot discard a live
        // buffer whose bytes are still included in retained_bytes.
        host_counts.resize(count);
        host_minimum_buckets.resize(count);
        output_buffers.resize(count);
        output_placement.resize(count);
        host_payloads.resize(count * count);
        for (std::size_t rank = 0; rank < count; ++rank) {
            grow(host_counts[rank], (count + 1U) * sizeof(int));
            grow(host_minimum_buckets[rank], sizeof(std::uint64_t));
            const std::size_t bytes = plan.partitions[rank].owned_vertex_count * sizeof(double);
            const bool needed = bytes && (!output_buffers[rank] || output_buffers[rank]->size() < bytes);
            if (needed && numa_local_output) {
                const std::size_t previous = output_buffers[rank] ? output_buffers[rank]->size() : 0U;
                output_buffers[rank] = gpu_easygraph::runtime::make_gpu_local_pinned_host_buffer(
                    bytes, colocated_device >= 0 ? colocated_device : plan.partitions[rank].device_id,
                    &output_placement[rank]);
                retained_bytes += bytes - previous;
                ++allocation_count;
            } else if (!numa_local_output) {
                grow(output_buffers[rank], bytes);
                if (needed) output_placement[rank].allocation_cpu = sched_getcpu();
            }
        }
        if (staging) for (std::size_t src = 0; src < count; ++src)
            for (std::size_t dst = 0; dst < count; ++dst)
                grow(host_payloads[src * count + dst],
                    plan.route(src, dst).capacity() * sssp_wire_bytes(plan.wire_encoding));
    }
};

py::array_t<double> sssp_owning_host_array(std::vector<double>& values) {
    std::unique_ptr<std::vector<double> > storage(new std::vector<double>(std::move(values)));
    double* data = storage->data();
    const py::ssize_t count = static_cast<py::ssize_t>(storage->size());
    py::capsule owner(storage.get(), [](void* pointer) {
        delete static_cast<std::vector<double>*>(pointer);
    });
    storage.release();
    return py::array_t<double>({count}, {sizeof(double)}, data, owner);
}
