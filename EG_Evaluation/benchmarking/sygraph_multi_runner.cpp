#include <openssl/sha.h>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
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

#if !defined(SYGRAPH_RUNNER_SSSP) && !defined(SYGRAPH_RUNNER_BC)
#error "Compile sygraph_multi_runner.cpp for exactly one algorithm"
#endif
#if defined(SYGRAPH_RUNNER_SSSP) && defined(SYGRAPH_RUNNER_BC)
#error "SSSP and BC must use separate translation units"
#endif

struct Options {
    std::string function;
    fs::path offsets;
    fs::path indices;
    fs::path weights;
    fs::path result_output;
    std::uint32_t nodes = 0;
    std::uint64_t entries = 0;
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
        if (arg == "--function") options.function = next();
        else if (arg == "--offsets") options.offsets = next();
        else if (arg == "--indices") options.indices = next();
        else if (arg == "--weights") options.weights = next();
        else if (arg == "--result-output") options.result_output = next();
        else if (arg == "--nodes") options.nodes = static_cast<std::uint32_t>(std::stoull(next()));
        else if (arg == "--entries") options.entries = std::stoull(next());
        else if (arg == "--directed") options.directed = std::stoi(next()) != 0;
        else if (arg == "--sources") options.sources = parse_sources(next());
        else throw std::runtime_error("unknown option: " + arg);
    }
#if defined(SYGRAPH_RUNNER_SSSP)
    if (options.function != "sssp") throw std::runtime_error("this binary requires --function sssp");
#else
    if (options.function != "bc") throw std::runtime_error("this binary requires --function bc");
#endif
    if (options.offsets.empty() || options.indices.empty() || options.nodes == 0 || options.sources.empty()) {
        throw std::runtime_error("--offsets, --indices, --nodes, --entries, and --sources are required");
    }
    if (options.function == "sssp" && options.weights.empty()) {
        throw std::runtime_error("SSSP requires --weights");
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

static void write_raw(const fs::path& path, const std::vector<double>& values) {
    if (path.empty()) return;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!output) throw std::runtime_error("failed writing " + path.string());
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

struct Measurement {
    std::vector<double> values;
    double kernel_seconds = 0.0;
    std::size_t rows = 0;
    std::size_t columns = 0;
};

#if defined(SYGRAPH_RUNNER_SSSP)
template <typename Graph>
static Measurement run_sssp(Graph& graph, const Options& options) {
    using clock = std::chrono::steady_clock;
    Measurement measured;
    measured.rows = options.sources.size();
    measured.columns = options.nodes;
    measured.values.resize(measured.rows * measured.columns);
    sygraph::algorithms::SSSP sssp{graph};
    const double unreachable = std::numeric_limits<double>::max();
    for (std::size_t source_index = 0; source_index < options.sources.size(); ++source_index) {
        auto source = options.sources[source_index];
        sssp.init(source);
        const auto started = clock::now();
        sssp.run();
        graph.getQueue().wait_and_throw();
        const auto finished = clock::now();
        measured.kernel_seconds += seconds<clock>(started, finished);
        const auto distances = sssp.getDistances();
        const std::size_t row_offset = source_index * measured.columns;
        for (std::size_t vertex = 0; vertex < distances.size(); ++vertex) {
            measured.values[row_offset + vertex] = distances[vertex] == unreachable
                ? std::numeric_limits<double>::infinity()
                : static_cast<double>(distances[vertex]);
        }
    }
    return measured;
}
#else
template <typename Graph>
static Measurement run_bc(Graph& graph, const Options& options) {
    using clock = std::chrono::steady_clock;
    Measurement measured;
    measured.rows = 1;
    measured.columns = options.nodes;
    measured.values.assign(options.nodes, 0.0);
    for (std::uint32_t source : options.sources) {
        sygraph::algorithms::BC bc{graph};
        bc.init(source);
        const auto started = clock::now();
        bc.run();
        graph.getQueue().wait_and_throw();
        const auto finished = clock::now();
        measured.kernel_seconds += seconds<clock>(started, finished);
        const auto values = bc.getValues();
        for (std::size_t vertex = 0; vertex < values.size(); ++vertex) {
            measured.values[vertex] += static_cast<double>(values[vertex]);
        }
    }
    if (!options.directed) {
        for (double& value : measured.values) value *= 0.5;
    }
    return measured;
}
#endif

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;
    try {
        const Options options = parse_options(argc, argv);
        const auto build_started = clock::now();
        auto offsets = read_raw<std::uint32_t>(options.offsets, static_cast<std::size_t>(options.nodes) + 1);
        auto indices = read_raw<std::uint32_t>(options.indices, static_cast<std::size_t>(options.entries));
        const auto host_loaded = clock::now();

        sycl::queue queue{sycl::gpu_selector_v};
        sygraph::graph::Properties properties;
        properties.directed = options.directed;
        properties.weighted =
#if defined(SYGRAPH_RUNNER_SSSP)
            true;
#else
            false;
#endif
        Measurement measured;
        std::chrono::steady_clock::time_point graph_built;
        std::chrono::steady_clock::time_point e2e_started;
        std::chrono::steady_clock::time_point e2e_finished;
        auto values =
#if defined(SYGRAPH_RUNNER_SSSP)
            read_raw<double>(options.weights, static_cast<std::size_t>(options.entries));
#else
            std::vector<double>(static_cast<std::size_t>(options.entries), 1.0);
#endif
            sygraph::formats::CSR<double, std::uint32_t, std::uint32_t> csr{
                std::move(offsets), std::move(indices), std::move(values)};
            auto graph = sygraph::graph::build::fromCSR<sygraph::memory::space::device>(
                queue, std::move(csr), properties);
            queue.wait_and_throw();
            graph_built = clock::now();
            e2e_started = graph_built;
            measured =
#if defined(SYGRAPH_RUNNER_SSSP)
                run_sssp(graph, options);
#else
                run_bc(graph, options);
#endif
            e2e_finished = clock::now();

        write_raw(options.result_output, measured.values);
        const std::string result_sha256 = sha256(measured.values);
        std::uint64_t finite_count = 0;
        std::uint64_t inf_count = 0;
        long double finite_sum = 0.0L;
        double maximum = 0.0;
        for (double value : measured.values) {
            if (std::isfinite(value)) {
                ++finite_count;
                finite_sum += value;
                maximum = std::max(maximum, value);
            } else {
                ++inf_count;
            }
        }
        const std::string device = queue.get_device().get_info<sycl::info::device::name>();
        std::cout << "RESULT_JSON {"
                  << "\"status\":\"ok\","
                  << "\"function\":\"" << options.function << "\","
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
                  << "\"host_csr_load_seconds\":" << seconds<clock>(build_started, host_loaded) << ','
                  << "\"device_graph_build_seconds\":" << seconds<clock>(host_loaded, graph_built) << ','
                  << "\"build_seconds\":" << seconds<clock>(build_started, graph_built) << ','
                  << "\"e2e_seconds\":" << seconds<clock>(e2e_started, e2e_finished) << ','
                  << "\"kernel_seconds\":" << measured.kernel_seconds << ','
                  << "\"result_sha256\":\"" << result_sha256 << "\","
                  << "\"result_shape\":[" << measured.rows << ',' << measured.columns << "],"
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
