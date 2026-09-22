#include <openssl/sha.h>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    fs::path offsets;
    fs::path indices;
    std::uint32_t nodes = 0;
    std::uint32_t entries = 0;
    bool directed = false;
    std::vector<std::uint32_t> sources;
};

static std::string json_escape(const std::string& value) {
    std::string output;
    output.reserve(value.size() + 8);
    for (char ch : value) {
        if (ch == '\\' || ch == '"') output.push_back('\\');
        if (ch == '\n') {
            output += "\\n";
        } else if (ch != '\r') {
            output.push_back(ch);
        }
    }
    return output;
}

static std::vector<std::uint32_t> parse_sources(const std::string& text) {
    std::vector<std::uint32_t> output;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) output.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    }
    return output;
}

static Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("missing value after " + arg);
            return argv[index];
        };
        if (arg == "--offsets") options.offsets = next();
        else if (arg == "--indices") options.indices = next();
        else if (arg == "--nodes") options.nodes = static_cast<std::uint32_t>(std::stoull(next()));
        else if (arg == "--entries") options.entries = static_cast<std::uint32_t>(std::stoull(next()));
        else if (arg == "--directed") options.directed = std::stoi(next()) != 0;
        else if (arg == "--sources") options.sources = parse_sources(next());
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (options.offsets.empty() || options.indices.empty() || options.nodes == 0 || options.sources.empty()) {
        throw std::runtime_error("--offsets, --indices, --nodes, --entries, and --sources are required");
    }
    for (std::uint32_t source : options.sources) {
        if (source >= options.nodes) throw std::runtime_error("source is outside the graph");
    }
    return options;
}

template <typename T>
static std::vector<T> read_raw(const fs::path& path, std::size_t count) {
    const std::uintmax_t expected = count * sizeof(T);
    if (fs::file_size(path) != expected) {
        throw std::runtime_error("binary size mismatch for " + path.string());
    }
    std::vector<T> values(count);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    constexpr std::size_t chunk_bytes = 1ULL << 30;
    std::size_t offset = 0;
    const std::size_t total = count * sizeof(T);
    char* data = reinterpret_cast<char*>(values.data());
    while (offset < total) {
        const std::size_t chunk = std::min(chunk_bytes, total - offset);
        input.read(data + offset, static_cast<std::streamsize>(chunk));
        if (!input) throw std::runtime_error("failed reading " + path.string());
        offset += chunk;
    }
    return values;
}

static std::string sha256(const std::vector<double>& values) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(
        reinterpret_cast<const unsigned char*>(values.data()),
        values.size() * sizeof(double),
        digest);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char byte : digest) output << std::setw(2) << static_cast<unsigned int>(byte);
    return output.str();
}

template <typename Clock>
static double seconds(typename Clock::time_point start, typename Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;
    try {
        const Options options = parse_options(argc, argv);
        const auto build_started = clock::now();
        auto offsets = read_raw<std::uint32_t>(options.offsets, static_cast<std::size_t>(options.nodes) + 1);
        auto indices = read_raw<std::uint32_t>(options.indices, options.entries);
        auto values = std::vector<std::uint8_t>(options.entries, 1);
        const auto host_loaded = clock::now();

        sycl::queue queue{sycl::gpu_selector_v};
        sygraph::graph::Properties properties;
        properties.directed = options.directed;
        properties.weighted = false;
        sygraph::formats::CSR<std::uint8_t, std::uint32_t, std::uint32_t> csr{
            std::move(offsets), std::move(indices), std::move(values)};
        auto graph = sygraph::graph::build::fromCSR<sygraph::memory::space::device>(
            queue, std::move(csr), properties);
        queue.wait_and_throw();
        const auto graph_built = clock::now();

        const auto e2e_started = clock::now();
        std::vector<double> result;
        result.resize(static_cast<std::size_t>(options.nodes) * options.sources.size());
        sygraph::algorithms::BFS bfs{graph};
        double kernel_seconds = 0.0;
        std::uint64_t finite_count = 0;
        std::uint64_t inf_count = 0;
        long double finite_sum = 0.0L;
        double maximum = 0.0;
        std::vector<std::size_t> iteration_counts;
        std::vector<std::size_t> source_degrees;
        iteration_counts.reserve(options.sources.size());
        source_degrees.reserve(options.sources.size());
        for (std::size_t source_index = 0; source_index < options.sources.size(); ++source_index) {
            std::uint32_t source = options.sources[source_index];
            source_degrees.push_back(graph.getDegree(source));
            bfs.init(source);
            const auto kernel_started = clock::now();
            const auto details = bfs.run(sygraph::algorithms::bfs_direction::push);
            queue.wait_and_throw();
            const auto kernel_finished = clock::now();
            kernel_seconds += seconds<clock>(kernel_started, kernel_finished);
            iteration_counts.push_back(details.iterations);
            const auto distances = bfs.getDistances();
            const std::size_t row_offset = source_index * static_cast<std::size_t>(options.nodes);
            const std::uint32_t unreachable = options.nodes + 1;
            for (std::size_t vertex = 0; vertex < distances.size(); ++vertex) {
                if (distances[vertex] == unreachable) {
                    result[row_offset + vertex] = std::numeric_limits<double>::infinity();
                    ++inf_count;
                } else {
                    const double value = static_cast<double>(distances[vertex]);
                    result[row_offset + vertex] = value;
                    finite_sum += value;
                    maximum = std::max(maximum, value);
                    ++finite_count;
                }
            }
        }
        const auto e2e_finished = clock::now();
        const std::string result_sha256 = sha256(result);
        const std::string device = queue.get_device().get_info<sycl::info::device::name>();

        std::cout << "RESULT_JSON {"
                  << "\"status\":\"ok\","
                  << "\"device\":\"" << json_escape(device) << "\","
                  << "\"nodes\":" << options.nodes << ','
                  << "\"entries\":" << options.entries << ','
                  << "\"directed\":" << (options.directed ? "true" : "false") << ','
                  << "\"source_count\":" << options.sources.size() << ','
                  << "\"sources\":[";
        for (std::size_t index = 0; index < options.sources.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << options.sources[index];
        }
        std::cout << "],"
                  << "\"source_degrees\":[";
        for (std::size_t index = 0; index < source_degrees.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << source_degrees[index];
        }
        std::cout << "],"
                  << "\"iteration_counts\":[";
        for (std::size_t index = 0; index < iteration_counts.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << iteration_counts[index];
        }
        std::cout << "],"
                  << "\"host_csr_load_seconds\":" << seconds<clock>(build_started, host_loaded) << ','
                  << "\"device_graph_build_seconds\":" << seconds<clock>(host_loaded, graph_built) << ','
                  << "\"build_seconds\":" << seconds<clock>(build_started, graph_built) << ','
                  << "\"e2e_seconds\":" << seconds<clock>(e2e_started, e2e_finished) << ','
                  << "\"kernel_seconds\":" << kernel_seconds << ','
                  << "\"result_sha256\":\"" << result_sha256 << "\","
                  << "\"result_shape\":[" << options.sources.size() << ',' << options.nodes << "],"
                  << "\"finite_count\":" << finite_count << ','
                  << "\"inf_count\":" << inf_count << ','
                  << "\"finite_sum\":" << static_cast<double>(finite_sum) << ','
                  << "\"maximum\":" << maximum
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "RESULT_JSON {\"status\":\"failed\",\"error\":\""
                  << json_escape(error.what()) << "\"}\n";
        return 2;
    }
}
