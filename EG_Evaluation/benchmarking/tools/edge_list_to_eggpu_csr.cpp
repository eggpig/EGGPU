#include <algorithm>
#include <chrono>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    fs::path input;
    fs::path output_dir;
    std::string name;
    std::string format;
    std::string source_url;
    std::int64_t num_nodes = -1;
    std::int64_t expected_edges = -1;
    int input_base = 0;
    bool directed = false;
    bool mirror_undirected = false;
    bool relabel = false;
};

struct ScanStats {
    std::int64_t input_edges = 0;
    std::int64_t accepted_edges = 0;
    std::int64_t self_loops = 0;
};

static std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char ch : value) {
        if (ch == '\\' || ch == '"') out.push_back('\\');
        if (ch == '\n') {
            out += "\\n";
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

static bool parse_two_integers(const char* line, std::int64_t* a, std::int64_t* b) {
    char* end = nullptr;
    errno = 0;
    long long first = std::strtoll(line, &end, 10);
    if (end == line || errno != 0) return false;
    const char* second_begin = end;
    errno = 0;
    long long second = std::strtoll(second_begin, &end, 10);
    if (end == second_begin || errno != 0) return false;
    *a = static_cast<std::int64_t>(first);
    *b = static_cast<std::int64_t>(second);
    return true;
}

template <typename Callback>
static ScanStats scan_edges(
    const Options& opt,
    Callback callback,
    std::unordered_map<std::int64_t, int>* relabel_map = nullptr,
    std::vector<std::int64_t>* original_labels = nullptr,
    bool insert_labels = false) {
    FILE* input = std::fopen(opt.input.c_str(), "rb");
    if (input == nullptr) {
        throw std::runtime_error("cannot open input: " + opt.input.string());
    }
    char* line = nullptr;
    std::size_t capacity = 0;
    bool matrix_dimensions_seen = opt.format != "matrix-market";
    ScanStats stats;
    while (getline(&line, &capacity, input) >= 0) {
        const char* cursor = line;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r') ++cursor;
        if (*cursor == '\0' || *cursor == '\n' || *cursor == '#' || *cursor == '%') {
            continue;
        }
        std::int64_t raw_u = 0;
        std::int64_t raw_v = 0;
        if (!parse_two_integers(cursor, &raw_u, &raw_v)) continue;
        if (!matrix_dimensions_seen) {
            if (opt.num_nodes >= 0 && (raw_u != opt.num_nodes || raw_v != opt.num_nodes)) {
                std::free(line);
                std::fclose(input);
                throw std::runtime_error("Matrix Market dimensions do not match --num-nodes");
            }
            matrix_dimensions_seen = true;
            continue;
        }
        ++stats.input_edges;
        const std::int64_t source_u = raw_u - opt.input_base;
        const std::int64_t source_v = raw_v - opt.input_base;
        int u = -1;
        int v = -1;
        if (opt.relabel) {
            auto map_label = [&](std::int64_t label) -> int {
                auto found = relabel_map->find(label);
                if (found != relabel_map->end()) return found->second;
                if (!insert_labels) {
                    throw std::runtime_error("node label disappeared between CSR passes");
                }
                if (relabel_map->size() >= static_cast<std::size_t>(opt.num_nodes)) {
                    throw std::runtime_error("unique source labels exceed --num-nodes");
                }
                const int mapped = static_cast<int>(relabel_map->size());
                relabel_map->emplace(label, mapped);
                original_labels->push_back(label);
                return mapped;
            };
            u = map_label(source_u);
            v = map_label(source_v);
        } else {
            if (source_u < 0 || source_v < 0 || source_u >= opt.num_nodes || source_v >= opt.num_nodes) {
                std::free(line);
                std::fclose(input);
                throw std::runtime_error("edge endpoint is outside [0, num_nodes)");
            }
            u = static_cast<int>(source_u);
            v = static_cast<int>(source_v);
        }
        if (u < 0 || v < 0 || u >= opt.num_nodes || v >= opt.num_nodes) {
            std::free(line);
            std::fclose(input);
            throw std::runtime_error("edge endpoint is outside [0, num_nodes)");
        }
        if (u == v) {
            ++stats.self_loops;
            continue;
        }
        callback(u, v);
        ++stats.accepted_edges;
    }
    std::free(line);
    std::fclose(input);
    return stats;
}

template <typename T>
static void write_raw(const fs::path& path, const std::vector<T>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create output: " + path.string());
    constexpr std::size_t chunk_bytes = 1ULL << 30;
    const char* data = reinterpret_cast<const char*>(values.data());
    const std::size_t total = values.size() * sizeof(T);
    std::size_t written = 0;
    while (written < total) {
        const std::size_t count = std::min(chunk_bytes, total - written);
        output.write(data + written, static_cast<std::streamsize>(count));
        if (!output) throw std::runtime_error("failed writing output: " + path.string());
        written += count;
    }
}

static Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value after " + arg);
            return argv[i];
        };
        if (arg == "--input") opt.input = next();
        else if (arg == "--output-dir") opt.output_dir = next();
        else if (arg == "--name") opt.name = next();
        else if (arg == "--format") opt.format = next();
        else if (arg == "--source-url") opt.source_url = next();
        else if (arg == "--num-nodes") opt.num_nodes = std::stoll(next());
        else if (arg == "--expected-edges") opt.expected_edges = std::stoll(next());
        else if (arg == "--input-base") opt.input_base = std::stoi(next());
        else if (arg == "--directed") opt.directed = true;
        else if (arg == "--mirror-undirected") opt.mirror_undirected = true;
        else if (arg == "--relabel") opt.relabel = true;
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (opt.input.empty() || opt.output_dir.empty() || opt.name.empty()) {
        throw std::runtime_error("--input, --output-dir, and --name are required");
    }
    if (opt.format != "edge-list" && opt.format != "matrix-market") {
        throw std::runtime_error("--format must be edge-list or matrix-market");
    }
    if (opt.num_nodes < 0 || opt.num_nodes >= INT_MAX) {
        throw std::runtime_error("--num-nodes must fit the signed 32-bit CSR ABI");
    }
    if (!opt.directed && !opt.mirror_undirected) {
        throw std::runtime_error("undirected input requires --mirror-undirected");
    }
    return opt;
}

int main(int argc, char** argv) {
    try {
        const auto conversion_started = std::chrono::steady_clock::now();
        const Options opt = parse_options(argc, argv);
        fs::create_directories(opt.output_dir);
        std::vector<std::uint64_t> degrees(static_cast<std::size_t>(opt.num_nodes), 0);
        std::unordered_map<std::int64_t, int> relabel_map;
        std::vector<std::int64_t> original_labels;
        if (opt.relabel) {
            relabel_map.reserve(static_cast<std::size_t>(opt.num_nodes * 1.3));
            original_labels.reserve(static_cast<std::size_t>(opt.num_nodes));
        }
        ScanStats first;
        if (opt.relabel) {
            first = scan_edges(
                opt,
                [](int, int) {},
                &relabel_map,
                &original_labels,
                true);
            if (relabel_map.size() != static_cast<std::size_t>(opt.num_nodes)) {
                throw std::runtime_error(
                    "unique source label count does not match --num-nodes: " +
                    std::to_string(relabel_map.size()) + " vs " +
                    std::to_string(opt.num_nodes));
            }
            std::sort(original_labels.begin(), original_labels.end());
            relabel_map.clear();
            relabel_map.reserve(static_cast<std::size_t>(opt.num_nodes * 1.3));
            for (std::size_t idx = 0; idx < original_labels.size(); ++idx) {
                relabel_map.emplace(original_labels[idx], static_cast<int>(idx));
            }
            const ScanStats counted = scan_edges(
                opt,
                [&](int u, int v) {
                    ++degrees[static_cast<std::size_t>(u)];
                    if (opt.mirror_undirected) ++degrees[static_cast<std::size_t>(v)];
                },
                &relabel_map,
                &original_labels,
                false);
            if (counted.accepted_edges != first.accepted_edges ||
                counted.self_loops != first.self_loops) {
                throw std::runtime_error("input changed between label and degree passes");
            }
        } else {
            first = scan_edges(opt, [&](int u, int v) {
                ++degrees[static_cast<std::size_t>(u)];
                if (opt.mirror_undirected) ++degrees[static_cast<std::size_t>(v)];
            });
        }
        if (opt.expected_edges >= 0 && first.accepted_edges != opt.expected_edges) {
            throw std::runtime_error(
                "accepted edge count does not match --expected-edges: " +
                std::to_string(first.accepted_edges) + " vs " +
                std::to_string(opt.expected_edges));
        }

        std::vector<int> offsets(static_cast<std::size_t>(opt.num_nodes) + 1, 0);
        std::int64_t entries = 0;
        std::int64_t nonzero_nodes = 0;
        std::uint64_t max_degree = 0;
        for (std::int64_t node = 0; node < opt.num_nodes; ++node) {
            const std::uint64_t degree = degrees[static_cast<std::size_t>(node)];
            if (degree > 0) ++nonzero_nodes;
            if (degree > max_degree) max_degree = degree;
            entries += static_cast<std::int64_t>(degree);
            if (entries >= INT_MAX) {
                throw std::runtime_error(
                    "stored CSR entries exceed the signed 32-bit EGGPU limit");
            }
            offsets[static_cast<std::size_t>(node) + 1] = static_cast<int>(entries);
        }

        std::vector<int> indices(static_cast<std::size_t>(entries));
        std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
        const ScanStats second = scan_edges(opt, [&](int u, int v) {
            indices[static_cast<std::size_t>(cursor[static_cast<std::size_t>(u)]++)] = v;
            if (opt.mirror_undirected) {
                indices[static_cast<std::size_t>(cursor[static_cast<std::size_t>(v)]++)] = u;
            }
        }, opt.relabel ? &relabel_map : nullptr,
           opt.relabel ? &original_labels : nullptr,
           false);
        if (second.accepted_edges != first.accepted_edges || second.self_loops != first.self_loops) {
            throw std::runtime_error("input changed between the two CSR construction passes");
        }

        const fs::path offsets_path = opt.output_dir / (opt.name + ".offsets.i32");
        const fs::path indices_path = opt.output_dir / (opt.name + ".indices.i32");
        const fs::path labels_path = opt.output_dir / (opt.name + ".original-labels.i64");
        const fs::path metadata_path = opt.output_dir / (opt.name + ".json");
        write_raw(offsets_path, offsets);
        write_raw(indices_path, indices);
        if (opt.relabel) write_raw(labels_path, original_labels);

        const double conversion_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - conversion_started).count();

        std::ofstream metadata(metadata_path, std::ios::trunc);
        if (!metadata) throw std::runtime_error("cannot create metadata file");
        metadata << "{\n"
                 << "  \"format\": \"eggpu-csr-v1\",\n"
                 << "  \"generation\": 1,\n"
                 << "  \"name\": \"" << json_escape(opt.name) << "\",\n"
                 << "  \"directed\": " << (opt.directed ? "true" : "false") << ",\n"
                 << "  \"num_nodes\": " << opt.num_nodes << ",\n"
                 << "  \"num_edges\": " << first.accepted_edges << ",\n"
                 << "  \"num_entries\": " << entries << ",\n"
                 << "  \"max_degree\": " << max_degree << ",\n"
                 << "  \"nonzero_degree_nodes\": " << nonzero_nodes << ",\n"
                 << "  \"self_loops_removed\": " << first.self_loops << ",\n"
                 << "  \"duplicates_removed\": 0,\n"
                 << "  \"duplicate_policy\": \"source-certified-simple\",\n"
                 << "  \"conversion_seconds\": " << conversion_seconds << ",\n"
                 << "  \"relabelled\": " << (opt.relabel ? "true" : "false") << ",\n"
                 << "  \"node_labels\": \"zero_based_contiguous\",\n"
                 << "  \"offset_dtype\": \"int32\",\n"
                 << "  \"index_dtype\": \"int32\",\n"
                 << "  \"offsets_path\": \"" << offsets_path.filename().string() << "\",\n"
                 << "  \"indices_path\": \"" << indices_path.filename().string() << "\",\n"
                 << "  \"original_labels_path\": "
                 << (opt.relabel ? "\"" + labels_path.filename().string() + "\"" : "null")
                 << ",\n"
                 << "  \"source\": \"" << json_escape(opt.source_url) << "\",\n"
                 << "  \"source_file\": \"" << json_escape(opt.input.string()) << "\"\n"
                 << "}\n";

        std::cout << "Wrote " << metadata_path << " nodes=" << opt.num_nodes
                  << " logical_edges=" << first.accepted_edges
                  << " csr_entries=" << entries
                  << " self_loops_removed=" << first.self_loops << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << std::endl;
        return 2;
    }
}
