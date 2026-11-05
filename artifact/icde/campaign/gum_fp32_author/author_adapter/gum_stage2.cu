#include "baseline_memory_clock.h"
// Original author FP32 PageRank; only complete host output is normalized to FP64 within the query clock.
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grape/grape.h>
#include <grape/util.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <type_traits>
#include <limits>
#include <memory>
#include <cstring>
#include "numa_pinned_host_standalone.h"
#include "examples/flags.h"
#include "pagerank_author.h"
#include "fp32_owned_output.h"
#include <thrust/sequence.h>
#include "grape_gpu/fragment/host_fragment.h"
#include "grape_gpu/fragment/loader.h"
#include "grape_gpu/worker/gpu_worker.h"

DEFINE_int32(stage2_rounds, 3, "Two warm queries and one measured query by default");
DEFINE_int64(stage2_vertices, 0, "Dense input vertex count");

// The official EmptyType loader ignores AddVertex and derives its domain only
// from edge endpoints. An unused nonempty payload selects the official vertex
// loader, so isolated IDs from vfile become real graph vertices before execution.
using FragmentBase = grape_gpu::HostFragment<int64_t, uint32_t, int32_t,
                                         grape::EmptyType, grape::LoadStrategy::kOnlyOut>;
class Fragment : public FragmentBase {
 public:
  explicit Fragment(std::shared_ptr<vertex_map_t> map)
      : FragmentBase(map), device_(map->GetCommSpec().local_id()) {}
  using FragmentBase::SetupMirrorInfo;
  // The official generic loader still passes the CPU grape range type. Adapt
  // its unchanged bounds to the GPU fragment's range on the owning CUDA device.
  void SetupMirrorInfo(grape::fid_t fid, const grape::VertexRange<uint32_t>& range,
                       const std::vector<uint32_t>& gids) {
    CHECK_CUDA(cudaSetDevice(device_));
    FragmentBase::SetupMirrorInfo(fid, vertex_range_t(range.begin().GetValue(),
        range.end().GetValue()), gids);
  }
 private:
  int device_;
};
class DenseVertexParser
    : public grape::TSVLineParser<int64_t, int32_t, grape::EmptyType> {
 public:
  void LineParserForVFile(const std::string& line, int64_t& id,
                         int32_t& unused_payload) override {
    id = std::stoll(line);
    unused_payload = 0;
  }
};

template<class Partitioner>
std::shared_ptr<Fragment> load_complete_graph(
    const grape::CommSpec& comm, const grape_gpu::LoadGraphSpec& spec,
    std::string& load_mode, std::string& cache_prefix) {
  grape_gpu::EVFragmentLoader<Fragment, Partitioner, grape::LocalIOAdaptor,
                             DenseVertexParser> loader(comm);
  auto serialized_spec = spec;
  grape_gpu::SetSerialize<Fragment, Partitioner>(comm, FLAGS_serialization_prefix,
      FLAGS_efile, FLAGS_vfile, serialized_spec);
  load_mode = serialized_spec.deserialize ? "serialized-fragment" :
      (serialized_spec.serialize ? "text-and-write-fragment" : "text");
  cache_prefix = serialized_spec.deserialize ? serialized_spec.deserialization_prefix :
      serialized_spec.serialization_prefix;
  return loader.LoadFragment(FLAGS_efile, FLAGS_vfile, serialized_spec);
}
template<class App>
void run(const grape::CommSpec& comm) {
  const auto memory_setup_begin = baseline_memory::monotonic_ns();
  const double setup_begin = MPI_Wtime();
  auto spec = grape_gpu::DefaultLoadGraphSpec();
  spec.set_directed(FLAGS_directed);
  spec.set_rebalance(false, 0);
  spec.set_rm_self_cycle(false);
  MPI_Barrier(comm.comm());
  const double load_begin = MPI_Wtime();
  std::shared_ptr<Fragment> frag;
  std::string load_mode, cache_prefix;
  if (FLAGS_partitioner == "seg")
    frag = load_complete_graph<grape::SegmentedPartitioner<int64_t>>(comm, spec, load_mode, cache_prefix);
  else if (FLAGS_partitioner == "hash")
    frag = load_complete_graph<grape::HashPartitioner<int64_t>>(comm, spec, load_mode, cache_prefix);
  else throw std::runtime_error("PR adapter requires seg or hash partitioner");
  MPI_Barrier(comm.comm());
  const double load_complete = MPI_Wtime();
  const double load_seconds = load_complete - load_begin;
  if (frag->InnerVertices().size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("GUM output MPI count exceeds int range");
  const int count = frag->InnerVertices().size();
  std::vector<int> counts(comm.worker_num()), offsets(comm.worker_num());
  MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm.comm());
  int total = 0;
  for (int i = 0; i < comm.worker_num(); ++i) {
    offsets[i] = total;
    if (counts[i] < 0 || total > std::numeric_limits<int>::max() - counts[i])
      throw std::runtime_error("GUM output MPI displacement exceeds int range");
    total += counts[i];
  }
  if (total != FLAGS_stage2_vertices)
    throw std::runtime_error("GUM graph loader omitted original vertices");
  std::vector<int64_t> ids(count), all_ids(comm.worker_id() == 0 ? total : 0);
  int position = 0;
  for (auto v : frag->InnerVertices()) ids[position++] = frag->GetId(v);
  MPI_Gatherv(ids.data(), count, MPI_INT64_T, all_ids.data(), counts.data(),
              offsets.data(), MPI_INT64_T, 0, comm.comm());
  // Cache and validate the exact gathered OID permutation once. Segmented
  // partitioning alone is not evidence that concatenation follows OID order.
  int identity_order = 1, valid_permutation = 1;
  if (comm.worker_id() == 0) {
    std::vector<unsigned char> seen(total, 0);
    for (int i = 0; i < total; ++i) {
      const int64_t oid = all_ids[i];
      if (oid < 0 || oid >= total || seen[oid]) valid_permutation = 0;
      else seen[oid] = 1;
      if (oid != i) identity_order = 0;
    }
  }
  MPI_Bcast(&valid_permutation, 1, MPI_INT, 0, comm.comm());
  MPI_Bcast(&identity_order, 1, MPI_INT, 0, comm.comm());
  if (!valid_permutation) throw std::runtime_error("GUM output OIDs are not a complete dense permutation");
  const double immutable_ready = MPI_Wtime();
  using gpu_easygraph::runtime::GpuLocalPinnedAllocationInfo;
  using gpu_easygraph::runtime::make_gpu_local_pinned_host_buffer;
  using gpu_easygraph::runtime::json_quoted;
  GpuLocalPinnedAllocationInfo pinned_info;
  const size_t local_bytes = static_cast<size_t>(count) * sizeof(float);
  auto local_pinned = make_gpu_local_pinned_host_buffer(local_bytes, comm.local_id(), &pinned_info);
  float* const local_output = static_cast<float*>(local_pinned->data());
  // Multi-rank gather preserves author FP32 values; conversion to each new
  // owned final FP64 output occurs inside the timed dense scatter below.
  std::vector<float> gathered(comm.worker_id() == 0 && comm.worker_num() > 1 ? total : 0);
  std::vector<std::unique_ptr<double[]>> retained_outputs;
  retained_outputs.reserve(FLAGS_stage2_rounds);
  const double output_static_ready = MPI_Wtime();
  const double runtime_begin = output_static_ready;
  grape_gpu::AppConfig config;
  config.lb = grape_gpu::ParseLoadBalancing(FLAGS_lb);
  config.work_stealing = false;
  auto app = std::make_shared<App>();
  auto worker = std::make_shared<typename App::worker_t>(app, frag);
  worker->Init(comm, config, FLAGS_pr_d, FLAGS_pr_mr);
  CHECK_CUDA(cudaDeviceSynchronize());
  MPI_Barrier(comm.comm());
  const double runtime_ready = MPI_Wtime();
  static_assert(std::is_same<typename App::rank_t, float>::value, "Author PR compute must be FP32");
  static_assert(sizeof(thrust::pair<uint32_t,typename App::rank_t>) == 8, "Author PR message must remain 8 bytes");
  const auto context = worker->GetContext();
  const auto& rank_range = context->rank.GetVertexRange();
  const auto& next_range = context->next_rank.GetVertexRange();
  const auto owned_range = frag->InnerVertices();
  if (owned_range.begin().GetValue() < rank_range.begin().GetValue() ||
      owned_range.end().GetValue() > rank_range.end().GetValue() ||
      next_range.begin().GetValue() != rank_range.begin().GetValue() ||
      next_range.end().GetValue() != rank_range.end().GetValue())
    throw std::runtime_error("GUM owned vertices are not a contained contiguous FP32 range");
  const size_t owned_offset = owned_range.begin().GetValue() - rank_range.begin().GetValue();
  const double output_plan_ready = MPI_Wtime();
  std::ofstream detail(FLAGS_out_prefix + "/timings-rank-" + std::to_string(comm.worker_id()) + ".jsonl");
  detail << std::setprecision(17);
  detail << "{\"phase\":\"output_static_prepare\",\"rank\":" << comm.worker_id()
         << ",\"seconds\":" << output_static_ready-immutable_ready << ",\"owned_vertices\":" << count
         << ",\"vertices\":" << total << ",\"element_bytes\":" << sizeof(float)
         << ",\"identity_oid_concatenation\":" << (identity_order ? "true" : "false")
         << ",\"pinned_bytes\":" << local_bytes << ",\"pinned_address\":"
         << json_quoted(std::to_string(reinterpret_cast<std::uintptr_t>(local_output)))
         << ",\"logical_device\":" << pinned_info.logical_device
         << ",\"numa_node\":" << pinned_info.numa_node
         << ",\"allocation_cpu\":" << pinned_info.allocation_cpu
         << ",\"pci_bus_id\":" << json_quoted(pinned_info.pci_bus_id)
         << ",\"numa_maps\":" << json_quoted(gpu_easygraph::runtime::pinned_numa_mapping(local_output))
         << ",\"allowed_node_cpus\":[";
  for (size_t i = 0; i < pinned_info.allowed_node_cpus.size(); ++i) {
    if (i) detail << ',';
    detail << pinned_info.allowed_node_cpus[i];
  }
  detail << "]}\n";
  std::ofstream metrics;
  if (comm.worker_id() == 0) {
    metrics.open(FLAGS_out_prefix + "/timings.jsonl");
    metrics << std::setprecision(17);
    metrics << "{\"phase\":\"graph_load_and_partition\",\"seconds\":" << load_seconds
            << ",\"load_mode\":\"" << load_mode
            << "\",\"cache_prefix\":\"" << cache_prefix << "\"}\n";
    metrics << "{\"phase\":\"input_vertex_domain\",\"vertices\":" << total << "}\n";
  }
  if (comm.worker_id() == 0)
    metrics << "{\"phase\":\"device_state_and_communicator_construct\",\"seconds\":"
            << runtime_ready-runtime_begin << "}\n";
  detail << "{\"phase\":\"setup_wall\",\"rank\":" << comm.worker_id()
         << ",\"seconds\":" << output_plan_ready-setup_begin
         << ",\"pre_load_configuration_and_sync_seconds\":" << load_begin-setup_begin
         << ",\"file_csr_load_and_partition_seconds\":" << load_complete-load_begin
         << ",\"immutable_ids_counts_permutation_seconds\":" << immutable_ready-load_complete
         << ",\"output_static_pinned_and_buffers_seconds\":" << output_static_ready-immutable_ready
         << ",\"worker_gpu_and_communicator_construct_seconds\":" << runtime_ready-output_static_ready
         << ",\"owned_range_validation_seconds\":" << output_plan_ready-runtime_ready
         << ",\"file_read_seconds\":null,\"host_csr_build_seconds\":null,\"partition_seconds\":null"
         << ",\"standalone_h2h_seconds\":null,\"standalone_h2d_seconds\":null"
         << ",\"load_scope\":\"Author LoadFragment combines file/serialized-CSR loading, partitioning, host construction and possible device work; no internal pure-stage clock.\""
         << ",\"worker_scope\":\"Author worker Init combines device state and communicator construction; no isolated H2D clock.\""
         << ",\"boundaries_seconds\":[" << setup_begin << ',' << load_begin << ',' << load_complete
         << ',' << immutable_ready << ',' << output_static_ready << ',' << runtime_ready
         << ',' << output_plan_ready << "]}\n";
  if (comm.worker_id() == 0) {
    metrics << "{\"phase\":\"immutable_output_map_construct\",\"seconds\":"
            << immutable_ready-load_complete << "}\n";
    metrics << "{\"phase\":\"output_static_prepare\",\"seconds\":"
            << output_static_ready-immutable_ready << "}\n";
    metrics << "{\"phase\":\"setup_wall\",\"seconds\":"
            << output_plan_ready-setup_begin
            << ",\"file_read_seconds\":null,\"host_csr_build_seconds\":null,\"partition_seconds\":null"
            << ",\"standalone_h2h_seconds\":null,\"standalone_h2d_seconds\":null}\n";
  }
  detail << "{\"phase\":\"owned_device_range\",\"rank\":" << comm.worker_id()
         << ",\"owned_offset_elements\":" << owned_offset
         << ",\"owned_vertices\":" << count
         << ",\"rank_array_vertices\":" << rank_range.size() << "}\n";
  const auto memory_setup_end = baseline_memory::monotonic_ns();
  std::ofstream memory_markers(FLAGS_out_prefix + "/memory-markers-rank-" + std::to_string(comm.worker_id()) + ".jsonl");
  baseline_memory::interval(memory_markers, comm.worker_id(), "setup_graph_to_resident_output_ready", -1, memory_setup_begin, memory_setup_end);
  for (int round = 0; round < FLAGS_stage2_rounds; ++round) {
    MPI_Barrier(comm.comm());
    const auto memory_query_begin = baseline_memory::monotonic_ns();
    const double query_begin = MPI_Wtime();
    context->rank.SetValue(0.0);
    context->rank.H2D();
    context->next_rank.SetValue(0.0);
    context->next_rank.H2D();
    context->curr_iter = 0;
    context->dangling_sum = 0;
    CHECK_CUDA(cudaDeviceSynchronize());
    MPI_Barrier(comm.comm());
    const double initialized = MPI_Wtime();
    worker->Query();
    CHECK_CUDA(cudaDeviceSynchronize());
    MPI_Barrier(comm.comm());
    const double computed = MPI_Wtime();
    // Every query allocates a fresh independent complete final double array.
    // All earlier outputs remain alive until process exit, so pointer reuse or
    // overwriting prior returned data is impossible. No initialization pass is
    // needed because the setup permutation check proves every slot is written.
    std::unique_ptr<double[]> dense;
    if (comm.worker_id() == 0) dense.reset(new double[static_cast<size_t>(total)]);
    const double final_allocated = MPI_Wtime();
    if (count) {
      auto device_rank = context->rank.DeviceObject();
      CHECK_CUDA(cudaMemcpy(local_output, device_rank.data() + owned_offset,
                            local_bytes, cudaMemcpyDeviceToHost));
    }
    const double d2h_complete = MPI_Wtime();
    if (comm.worker_num() > 1) {
      float* destination = comm.worker_id() == 0 ? gathered.data() : nullptr;
      MPI_Gatherv(local_output, count, MPI_FLOAT, destination, counts.data(),
                  offsets.data(), MPI_FLOAT, 0, comm.comm());
    }
    const double gather_complete = MPI_Wtime();
    if (comm.worker_id() == 0) {
      const float* values = comm.worker_num() == 1 ? local_output : gathered.data();
      gum_fp32_dense_output(dense.get(), values, all_ids.data(),
                            static_cast<size_t>(total), identity_order);
    }
    const double scatter_complete = MPI_Wtime();
    MPI_Barrier(comm.comm());
    const double complete = MPI_Wtime();
    const auto memory_query_end = baseline_memory::monotonic_ns();
    baseline_memory::interval(memory_markers, comm.worker_id(), round<2 ? "warmup_query" : "measured_query", round, memory_query_begin, memory_query_end);
    // Adjacent boundaries partition the original query wall time exactly.
    const double segment_sum = (initialized-query_begin) + (computed-initialized) +
        (final_allocated-computed) + (d2h_complete-final_allocated) +
        (gather_complete-d2h_complete) + (scatter_complete-gather_complete) +
        (complete-scatter_complete);
    detail << "{\"round\":" << round << ",\"role\":\""
           << (round < 2 ? "warmup" : "measurement") << "\",\"rank\":" << comm.worker_id()
           << ",\"reset_seconds\":" << initialized-query_begin
           << ",\"algorithm_and_communication_seconds\":" << computed-initialized
           << ",\"final_allocation_seconds\":" << final_allocated-computed
           << ",\"d2h_seconds\":" << d2h_complete-final_allocated
           << ",\"mpi_gather_seconds\":" << gather_complete-d2h_complete
           << ",\"dense_scatter_seconds\":" << scatter_complete-gather_complete
           << ",\"completion_barrier_seconds\":" << complete-scatter_complete
           << ",\"warm_query_seconds\":" << complete-query_begin
           << ",\"segment_sum_seconds\":" << segment_sum
           << ",\"closure_error_seconds\":" << segment_sum-(complete-query_begin)
           << ",\"d2h_bytes\":" << local_bytes
           << ",\"final_elements\":" << (comm.worker_id() == 0 ? total : 0)
           << ",\"final_element_bytes\":" << sizeof(double)
           << ",\"final_address\":" << json_quoted(std::to_string(reinterpret_cast<std::uintptr_t>(dense.get())))
           << ",\"prior_live_outputs\":" << retained_outputs.size()
           << ",\"identity_oid_concatenation\":" << (identity_order ? "true" : "false")
           << ",\"query_boundaries_seconds\":[" << query_begin << ',' << initialized << ',' << computed
           << ',' << final_allocated << ',' << d2h_complete << ',' << gather_complete
           << ',' << scatter_complete << ',' << complete << "]}\n";
    if (comm.worker_id() == 0) {
      metrics << "{\"round\":" << round << ",\"role\":\""
              << (round < 2 ? "warmup" : "measurement")
              << "\",\"query_initialize_seconds\":" << initialized-query_begin
              << ",\"algorithm_and_communication_seconds\":" << computed-initialized
              << ",\"d2h_gather_dense_seconds\":" << complete-computed
              << ",\"warm_query_seconds\":" << complete-query_begin
              << ",\"iterations\":" << FLAGS_pr_mr
              << ",\"alpha\":" << context->damping_factor
              << ",\"rank_bytes\":" << sizeof(typename App::rank_t)
              << ",\"message_bytes\":" << sizeof(thrust::pair<uint32_t,typename App::rank_t>)
              << "}\n";
      std::ofstream out(FLAGS_out_prefix + "/values-" + std::to_string(round) + ".f64", std::ios::binary);
      out.write(reinterpret_cast<const char*>(dense.get()), static_cast<size_t>(total)*sizeof(double));
      if (!out) throw std::runtime_error("failed to write complete FP64 PR output");
      retained_outputs.emplace_back(std::move(dense));
    }
    MPI_Barrier(comm.comm());
  }
}

int main(int argc, char** argv) {
  grape::gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  grape::InitMPIComm();
  {
    grape::CommSpec comm;
    comm.Init(MPI_COMM_WORLD);
    CHECK_CUDA(cudaSetDevice(comm.local_id()));
    run<grape_gpu::Pagerank<Fragment>>(comm);
  }
  grape::FinalizeMPIComm();
  google::ShutdownGoogleLogging();
}
