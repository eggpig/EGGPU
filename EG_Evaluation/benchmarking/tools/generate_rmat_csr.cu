#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t kSignedInt32Limit = 1ULL << 31;

struct Options {
    int scale = -1;
    int edge_factor = 16;
    std::uint64_t seed = 20260716ULL;
    fs::path output_dir;
    std::string name;
    bool force = false;
};

struct IsSelfLoop {
    __host__ __device__ bool operator()(std::uint64_t edge) const {
        return static_cast<std::uint32_t>(edge >> 32) ==
               static_cast<std::uint32_t>(edge);
    }
};

struct IsNotSelfLoop {
    __host__ __device__ bool operator()(std::uint64_t edge) const {
        return !IsSelfLoop{}(edge);
    }
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ std::uint32_t permute_vertex(
    std::uint32_t value,
    std::uint32_t mask,
    std::uint64_t seed) {
    // Every operation is bijective modulo 2^scale: xor with a constant,
    // multiplication by an odd integer, and reversible xor-shifts.
    value = (value ^ static_cast<std::uint32_t>(seed)) & mask;
    value = (value * 0x9e3779b1U) & mask;
    value ^= value >> 13;
    value &= mask;
    value = (value * 0x85ebca6bU) & mask;
    value ^= value >> 7;
    return value & mask;
}

__global__ void generate_edges(
    std::uint64_t* edges,
    std::uint64_t edge_count,
    int scale,
    std::uint64_t seed,
    std::uint64_t threshold_a,
    std::uint64_t threshold_ab,
    std::uint64_t threshold_abc) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= edge_count) return;

    std::uint64_t state = splitmix64(seed ^ (index * 0xd1342543de82ef95ULL));
    std::uint32_t source = 0;
    std::uint32_t destination = 0;
    for (int bit = scale - 1; bit >= 0; --bit) {
        state = splitmix64(state);
        const std::uint64_t sample = state >> 32;
        if (sample >= threshold_a && sample < threshold_ab) {
            destination |= 1U << bit;
        } else if (sample >= threshold_ab && sample < threshold_abc) {
            source |= 1U << bit;
        } else if (sample >= threshold_abc) {
            source |= 1U << bit;
            destination |= 1U << bit;
        }
    }
    const std::uint32_t mask = (1U << scale) - 1U;
    source = permute_vertex(source, mask, seed ^ 0x243f6a8885a308d3ULL);
    destination = permute_vertex(
        destination, mask, seed ^ 0x243f6a8885a308d3ULL);
    edges[index] = (static_cast<std::uint64_t>(source) << 32) | destination;
}

__global__ void mark_source_offsets(
    const std::uint64_t* edges,
    std::int32_t* offsets,
    std::int32_t edge_count) {
    const std::int32_t index =
        static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= edge_count) return;
    const std::uint32_t source = static_cast<std::uint32_t>(edges[index] >> 32);
    if (index == 0 || static_cast<std::uint32_t>(edges[index - 1] >> 32) != source) {
        offsets[source] = index;
    }
}

__global__ void extract_destinations(
    const std::uint64_t* edges,
    std::int32_t* indices,
    std::int32_t edge_count) {
    const std::int32_t index =
        static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= edge_count) return;
    indices[index] = static_cast<std::int32_t>(edges[index]);
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char character : value) {
        switch (character) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << character;
        }
    }
    return output.str();
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            return argv[index];
        };
        if (argument == "--scale") options.scale = std::stoi(value());
        else if (argument == "--edge-factor") options.edge_factor = std::stoi(value());
        else if (argument == "--seed") options.seed = std::stoull(value());
        else if (argument == "--output-dir") options.output_dir = value();
        else if (argument == "--name") options.name = value();
        else if (argument == "--force") options.force = true;
        else throw std::runtime_error("unknown option: " + argument);
    }
    if (options.scale < 1 || options.scale > 30) {
        throw std::runtime_error("--scale must be in [1, 30]");
    }
    if (options.edge_factor <= 0) {
        throw std::runtime_error("--edge-factor must be positive");
    }
    if (options.output_dir.empty() || options.name.empty()) {
        throw std::runtime_error("--output-dir and --name are required");
    }
    return options;
}

template <typename T>
void write_host_binary(const fs::path& path, const std::vector<T>& values) {
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create " + temporary.string());
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!output) throw std::runtime_error("failed writing " + temporary.string());
    output.close();
    fs::rename(temporary, path);
}

void write_device_i32(
    const fs::path& path,
    const std::int32_t* device,
    std::uint64_t count) {
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create " + temporary.string());
    constexpr std::uint64_t kChunkElements = 64ULL * 1024ULL * 1024ULL;
    std::vector<std::int32_t> host(
        static_cast<std::size_t>(std::min(count, kChunkElements)));
    for (std::uint64_t start = 0; start < count; start += kChunkElements) {
        const std::uint64_t length = std::min(kChunkElements, count - start);
        check_cuda(
            cudaMemcpy(
                host.data(), device + start, length * sizeof(std::int32_t),
                cudaMemcpyDeviceToHost),
            "copying CSR indices to host");
        output.write(
            reinterpret_cast<const char*>(host.data()),
            static_cast<std::streamsize>(length * sizeof(std::int32_t)));
        if (!output) throw std::runtime_error("failed writing " + temporary.string());
    }
    output.close();
    fs::rename(temporary, path);
}

std::uint64_t host_splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::vector<std::uint32_t> choose_sources(
    const std::vector<std::int32_t>& offsets,
    std::uint64_t seed,
    std::size_t count) {
    using Candidate = std::pair<std::uint64_t, std::uint32_t>;
    std::priority_queue<Candidate> selected;
    for (std::uint32_t node = 0; node + 1 < offsets.size(); ++node) {
        if (offsets[node] == offsets[node + 1]) continue;
        const Candidate candidate{host_splitmix64(seed ^ node), node};
        if (selected.size() < count) {
            selected.push(candidate);
        } else if (candidate < selected.top()) {
            selected.pop();
            selected.push(candidate);
        }
    }
    std::vector<Candidate> ordered;
    while (!selected.empty()) {
        ordered.push_back(selected.top());
        selected.pop();
    }
    std::sort(ordered.begin(), ordered.end());
    std::vector<std::uint32_t> sources;
    sources.reserve(ordered.size());
    for (const auto& item : ordered) sources.push_back(item.second);
    return sources;
}

double elapsed_seconds(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - started)
        .count();
}

}  // namespace

int main(int argc, char** argv) {
    std::uint64_t* raw_edges = nullptr;
    std::uint64_t* simple_edges = nullptr;
    std::int32_t* device_offsets = nullptr;
    std::int32_t* device_indices = nullptr;
    try {
        const Options options = parse_options(argc, argv);
        const std::uint64_t num_nodes = 1ULL << options.scale;
        const std::uint64_t raw_edge_count =
            num_nodes * static_cast<std::uint64_t>(options.edge_factor);
        if (num_nodes >= kSignedInt32Limit || raw_edge_count >= kSignedInt32Limit) {
            throw std::runtime_error(
                "nodes and raw directed edges must fit the signed 32-bit CSR ABI");
        }

        fs::create_directories(options.output_dir);
        const fs::path offsets_path =
            options.output_dir / (options.name + ".offsets.i32");
        const fs::path indices_path =
            options.output_dir / (options.name + ".indices.i32");
        const fs::path manifest_path = options.output_dir / (options.name + ".json");
        if (!options.force &&
            (fs::exists(offsets_path) || fs::exists(indices_path) ||
             fs::exists(manifest_path))) {
            throw std::runtime_error(
                "output already exists; pass --force to replace it");
        }
        if (options.force) {
            fs::remove(offsets_path);
            fs::remove(indices_path);
            fs::remove(manifest_path);
        }
        fs::remove(offsets_path.string() + ".tmp");
        fs::remove(indices_path.string() + ".tmp");

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "querying device memory");
        const std::uint64_t raw_bytes = raw_edge_count * sizeof(std::uint64_t);
        if (raw_bytes * 4 > static_cast<std::uint64_t>(free_bytes) * 9 / 10) {
            throw std::runtime_error(
                "insufficient free GPU memory for R-MAT generation and sort workspace");
        }

        const auto total_started = std::chrono::steady_clock::now();
        auto stage_started = total_started;
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&raw_edges), raw_bytes),
            "allocating raw R-MAT edges");
        constexpr int kBlockSize = 256;
        const int raw_grid = static_cast<int>(
            (raw_edge_count + kBlockSize - 1) / kBlockSize);
        constexpr double kProbabilityScale = 4294967296.0;
        const std::uint64_t threshold_a =
            static_cast<std::uint64_t>(std::llround(0.57 * kProbabilityScale));
        const std::uint64_t threshold_ab =
            static_cast<std::uint64_t>(std::llround(0.76 * kProbabilityScale));
        const std::uint64_t threshold_abc =
            static_cast<std::uint64_t>(std::llround(0.95 * kProbabilityScale));
        generate_edges<<<raw_grid, kBlockSize>>>(
            raw_edges, raw_edge_count, options.scale, options.seed,
            threshold_a, threshold_ab, threshold_abc);
        check_cuda(cudaGetLastError(), "launching R-MAT generation kernel");
        check_cuda(cudaDeviceSynchronize(), "synchronizing R-MAT generation");
        const double generation_seconds = elapsed_seconds(stage_started);

        thrust::device_ptr<std::uint64_t> raw_begin(raw_edges);
        thrust::device_ptr<std::uint64_t> raw_end(raw_edges + raw_edge_count);
        const std::uint64_t self_loops = static_cast<std::uint64_t>(
            thrust::count_if(raw_begin, raw_end, IsSelfLoop{}));
        stage_started = std::chrono::steady_clock::now();
        thrust::sort(raw_begin, raw_end);
        check_cuda(cudaDeviceSynchronize(), "synchronizing R-MAT sort");
        const double sort_seconds = elapsed_seconds(stage_started);

        stage_started = std::chrono::steady_clock::now();
        const auto unique_end = thrust::unique(raw_begin, raw_end);
        const std::uint64_t unique_with_loops =
            static_cast<std::uint64_t>(unique_end - raw_begin);
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&simple_edges),
                unique_with_loops * sizeof(std::uint64_t)),
            "allocating normalized R-MAT edges");
        thrust::device_ptr<std::uint64_t> simple_begin(simple_edges);
        const auto simple_end = thrust::copy_if(
            raw_begin, unique_end, simple_begin, IsNotSelfLoop{});
        const std::uint64_t simple_edge_count =
            static_cast<std::uint64_t>(simple_end - simple_begin);
        check_cuda(cudaDeviceSynchronize(), "synchronizing R-MAT normalization");
        const double normalization_seconds = elapsed_seconds(stage_started);
        check_cuda(cudaFree(raw_edges), "freeing raw R-MAT edges");
        raw_edges = nullptr;
        if (simple_edge_count >= kSignedInt32Limit) {
            throw std::runtime_error(
                "normalized directed CSR exceeds the signed 32-bit entry limit");
        }
        const std::uint64_t duplicate_non_loop_edges =
            raw_edge_count - self_loops - simple_edge_count;

        stage_started = std::chrono::steady_clock::now();
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_offsets),
                (num_nodes + 1) * sizeof(std::int32_t)),
            "allocating R-MAT offsets");
        thrust::device_ptr<std::int32_t> offsets_begin(device_offsets);
        thrust::fill(
            offsets_begin, offsets_begin + num_nodes + 1,
            static_cast<std::int32_t>(simple_edge_count));
        const int simple_grid = static_cast<int>(
            (simple_edge_count + kBlockSize - 1) / kBlockSize);
        mark_source_offsets<<<simple_grid, kBlockSize>>>(
            simple_edges, device_offsets,
            static_cast<std::int32_t>(simple_edge_count));
        check_cuda(cudaGetLastError(), "launching source-offset kernel");
        auto reverse_begin = thrust::make_reverse_iterator(
            offsets_begin + num_nodes + 1);
        auto reverse_end = thrust::make_reverse_iterator(offsets_begin);
        thrust::inclusive_scan(
            reverse_begin, reverse_end, reverse_begin,
            thrust::minimum<std::int32_t>());

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_indices),
                simple_edge_count * sizeof(std::int32_t)),
            "allocating R-MAT indices");
        extract_destinations<<<simple_grid, kBlockSize>>>(
            simple_edges, device_indices,
            static_cast<std::int32_t>(simple_edge_count));
        check_cuda(cudaGetLastError(), "launching destination extraction kernel");
        check_cuda(cudaDeviceSynchronize(), "synchronizing CSR construction");
        const double csr_seconds = elapsed_seconds(stage_started);
        check_cuda(cudaFree(simple_edges), "freeing normalized R-MAT edges");
        simple_edges = nullptr;

        stage_started = std::chrono::steady_clock::now();
        std::vector<std::int32_t> host_offsets(num_nodes + 1);
        check_cuda(
            cudaMemcpy(
                host_offsets.data(), device_offsets,
                host_offsets.size() * sizeof(std::int32_t),
                cudaMemcpyDeviceToHost),
            "copying R-MAT offsets to host");
        std::uint64_t max_degree = 0;
        std::uint64_t nonzero_nodes = 0;
        for (std::uint64_t node = 0; node < num_nodes; ++node) {
            const std::uint64_t degree = static_cast<std::uint64_t>(
                host_offsets[node + 1] - host_offsets[node]);
            max_degree = std::max(max_degree, degree);
            if (degree != 0) ++nonzero_nodes;
        }
        const auto benchmark_sources =
            choose_sources(host_offsets, options.seed ^ 0xa4093822299f31d0ULL, 64);
        write_host_binary(offsets_path, host_offsets);
        write_device_i32(indices_path, device_indices, simple_edge_count);
        const double write_seconds = elapsed_seconds(stage_started);
        check_cuda(cudaFree(device_offsets), "freeing R-MAT offsets");
        device_offsets = nullptr;
        check_cuda(cudaFree(device_indices), "freeing R-MAT indices");
        device_indices = nullptr;

        std::ofstream manifest(manifest_path, std::ios::trunc);
        if (!manifest) {
            throw std::runtime_error("cannot create " + manifest_path.string());
        }
        manifest << std::setprecision(17);
        manifest << "{\n";
        manifest << "  \"format\": \"eggpu-csr-v1\",\n";
        manifest << "  \"name\": \"" << json_escape(options.name) << "\",\n";
        manifest << "  \"source\": \"controlled Graph500-parameter R-MAT\",\n";
        manifest << "  \"source_url\": \"https://graph500.org/?page_id=12\",\n";
        manifest << "  \"offset_dtype\": \"int32\",\n";
        manifest << "  \"index_dtype\": \"int32\",\n";
        manifest << "  \"node_labels\": \"zero_based_contiguous\",\n";
        manifest << "  \"offsets_path\": \"" << offsets_path.filename().string()
                 << "\",\n";
        manifest << "  \"indices_path\": \"" << indices_path.filename().string()
                 << "\",\n";
        manifest << "  \"num_nodes\": " << num_nodes << ",\n";
        manifest << "  \"num_edges\": " << simple_edge_count << ",\n";
        manifest << "  \"num_entries\": " << simple_edge_count << ",\n";
        manifest << "  \"directed\": true,\n";
        manifest << "  \"generation\": 0,\n";
        manifest << "  \"nonzero_nodes\": " << nonzero_nodes << ",\n";
        manifest << "  \"max_degree\": " << max_degree << ",\n";
        manifest << "  \"self_loops_removed\": " << self_loops << ",\n";
        manifest << "  \"duplicates_removed\": " << duplicate_non_loop_edges << ",\n";
        manifest << "  \"normalization\": "
                 << "\"directed edge stream; self-loops removed; duplicate directed "
                    "pairs removed; CSR destinations sorted\",\n";
        manifest << "  \"edge_counts\": {\n";
        manifest << "    \"raw_edge_records\": " << raw_edge_count << ",\n";
        manifest << "    \"simple_directed_edges\": " << simple_edge_count << ",\n";
        manifest << "    \"unique_undirected_edges\": null,\n";
        manifest << "    \"csr_entries\": " << simple_edge_count << ",\n";
        manifest << "    \"self_loops_removed\": " << self_loops << ",\n";
        manifest << "    \"duplicates_removed\": " << duplicate_non_loop_edges << "\n";
        manifest << "  },\n";
        manifest << "  \"rmat\": {\n";
        manifest << "    \"scale\": " << options.scale << ",\n";
        manifest << "    \"edge_factor\": " << options.edge_factor << ",\n";
        manifest << "    \"seed\": " << options.seed << ",\n";
        manifest << "    \"a\": 0.57,\n";
        manifest << "    \"b\": 0.19,\n";
        manifest << "    \"c\": 0.19,\n";
        manifest << "    \"d\": 0.05,\n";
        manifest << "    \"rng\": \"counter-based SplitMix64\",\n";
        manifest << "    \"vertex_scrambling\": \"deterministic bijection over 2^scale labels\",\n";
        manifest << "    \"graph500_parameter_compatible\": true,\n";
        manifest << "    \"bit_identical_to_graph500_reference\": false\n";
        manifest << "  },\n";
        manifest << "  \"benchmark_sources_zero_based\": [";
        for (std::size_t index = 0; index < benchmark_sources.size(); ++index) {
            if (index) manifest << ", ";
            manifest << benchmark_sources[index];
        }
        manifest << "],\n";
        manifest << "  \"preprocessing\": {\n";
        manifest << "    \"generation_seconds\": " << generation_seconds << ",\n";
        manifest << "    \"sort_seconds\": " << sort_seconds << ",\n";
        manifest << "    \"normalization_seconds\": " << normalization_seconds << ",\n";
        manifest << "    \"csr_seconds\": " << csr_seconds << ",\n";
        manifest << "    \"write_seconds\": " << write_seconds << ",\n";
        manifest << "    \"total_seconds\": " << elapsed_seconds(total_started) << "\n";
        manifest << "  }\n";
        manifest << "}\n";
        manifest.close();

        std::cout << "Prepared " << manifest_path << " with " << num_nodes
                  << " nodes and " << simple_edge_count << " normalized edges"
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        if (raw_edges) cudaFree(raw_edges);
        if (simple_edges) cudaFree(simple_edges);
        if (device_offsets) cudaFree(device_offsets);
        if (device_indices) cudaFree(device_indices);
        std::cerr << "generate_rmat_csr: " << error.what() << std::endl;
        return 2;
    }
}
