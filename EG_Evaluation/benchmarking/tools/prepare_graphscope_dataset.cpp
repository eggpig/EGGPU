#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct Options {
  fs::path input;
  fs::path output;
  std::string name;
  bool directed = false;
  bool contiguous_one_based = false;
  bool sparse_one_based = false;
  bool matrix_market = false;
  bool skip_reverse = false;
  bool skip_undirected = false;
  std::uint64_t num_nodes = 0;
};

static void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --input PATH --output DIR --name NAME [--directed]"
         " [--contiguous-one-based --num-nodes N] [--matrix-market]"
         " [--sparse-one-based --num-nodes N]"
         " [--skip-reverse] [--skip-undirected]\n";
}

static Options parse_args(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&]() -> std::string {
      if (++i >= argc) {
        throw std::runtime_error("missing value after " + arg);
      }
      return argv[i];
    };
    if (arg == "--input") {
      options.input = require_value();
    } else if (arg == "--output") {
      options.output = require_value();
    } else if (arg == "--name") {
      options.name = require_value();
    } else if (arg == "--num-nodes") {
      options.num_nodes = std::stoull(require_value());
    } else if (arg == "--directed") {
      options.directed = true;
    } else if (arg == "--contiguous-one-based") {
      options.contiguous_one_based = true;
    } else if (arg == "--sparse-one-based") {
      options.sparse_one_based = true;
    } else if (arg == "--matrix-market") {
      options.matrix_market = true;
    } else if (arg == "--skip-reverse") {
      options.skip_reverse = true;
    } else if (arg == "--skip-undirected") {
      options.skip_undirected = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (options.input.empty() || options.output.empty() || options.name.empty()) {
    throw std::runtime_error("--input, --output, and --name are required");
  }
  if (options.contiguous_one_based && options.sparse_one_based) {
    throw std::runtime_error(
        "--contiguous-one-based and --sparse-one-based are mutually exclusive");
  }
  if ((options.contiguous_one_based || options.sparse_one_based) &&
      options.num_nodes == 0) {
    throw std::runtime_error(
        "--num-nodes is required with one-based streaming modes");
  }
  return options;
}

static bool ignored_line(const std::string& line) {
  const auto first = line.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return true;
  }
  const char c = line[first];
  return c == '#' || c == '%' || c == '/';
}

static std::uint64_t edge_weight(std::uint64_t src, std::uint64_t dst,
                                 std::uint64_t num_nodes) {
  const auto mod = std::max<std::uint64_t>(1, num_nodes);
  const unsigned __int128 product =
      static_cast<unsigned __int128>(src) * static_cast<unsigned __int128>(dst);
  return 1 + static_cast<std::uint64_t>(product % mod);
}

static void write_vertices(const fs::path& path, std::uint64_t num_nodes) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot create " + path.string());
  }
  out << "id\n";
  for (std::uint64_t node = 0; node < num_nodes; ++node) {
    out << node << '\n';
  }
}

static void write_edge(std::ofstream& out, std::uint64_t src, std::uint64_t dst,
                       std::uint64_t num_nodes) {
  out << src << ',' << dst << ',' << edge_weight(src, dst, num_nodes) << '\n';
}

struct PreparedCounts {
  std::uint64_t num_nodes = 0;
  std::uint64_t directed_edges = 0;
  std::uint64_t undirected_edges = 0;
  std::uint64_t self_loops_removed = 0;
};

static PreparedCounts prepare_generic(const Options& options) {
  std::ifstream input(options.input);
  if (!input) {
    throw std::runtime_error("cannot open " + options.input.string());
  }

  std::vector<std::pair<std::int64_t, std::int64_t>> raw_edges;
  std::vector<std::int64_t> labels;
  std::string line;
  while (std::getline(input, line)) {
    if (ignored_line(line)) {
      continue;
    }
    std::istringstream parser(line);
    std::int64_t src = 0;
    std::int64_t dst = 0;
    if (!(parser >> src >> dst)) {
      continue;
    }
    raw_edges.emplace_back(src, dst);
    labels.push_back(src);
    labels.push_back(dst);
  }
  if (raw_edges.empty()) {
    throw std::runtime_error("no parseable edges in " + options.input.string());
  }

  std::sort(labels.begin(), labels.end());
  labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
  std::unordered_map<std::int64_t, std::uint32_t> remap;
  remap.reserve(labels.size() * 2);
  for (std::uint32_t index = 0; index < labels.size(); ++index) {
    remap.emplace(labels[index], index);
  }

  std::vector<std::pair<std::uint32_t, std::uint32_t>> directed;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> undirected;
  directed.reserve(raw_edges.size());
  undirected.reserve(raw_edges.size());
  PreparedCounts counts;
  counts.num_nodes = labels.size();
  for (const auto& edge : raw_edges) {
    const auto src = remap.at(edge.first);
    const auto dst = remap.at(edge.second);
    if (src == dst) {
      ++counts.self_loops_removed;
      continue;
    }
    directed.emplace_back(src, dst);
    undirected.emplace_back(std::min(src, dst), std::max(src, dst));
  }
  std::sort(directed.begin(), directed.end());
  directed.erase(std::unique(directed.begin(), directed.end()), directed.end());
  std::sort(undirected.begin(), undirected.end());
  undirected.erase(std::unique(undirected.begin(), undirected.end()),
                   undirected.end());
  counts.directed_edges = directed.size();
  counts.undirected_edges = undirected.size();

  write_vertices(options.output / "vertices.csv", counts.num_nodes);

  if (options.directed) {
    std::ofstream edges(options.output / "edges.csv");
    std::ofstream reverse(options.output / "reverse.csv");
    if (!edges || !reverse) {
      throw std::runtime_error("cannot create directed edge outputs");
    }
    edges << "src,dst,weight\n";
    reverse << "src,dst,weight\n";
    for (const auto& edge : directed) {
      write_edge(edges, edge.first, edge.second, counts.num_nodes);
      write_edge(reverse, edge.second, edge.first, counts.num_nodes);
    }
  }

  if (!options.skip_undirected) {
    std::ofstream once(options.output / "undirected.csv");
    if (!once) {
      throw std::runtime_error("cannot create undirected.csv");
    }
    once << "src,dst,weight\n";
    for (const auto& edge : undirected) {
      write_edge(once, edge.first, edge.second, counts.num_nodes);
    }
  }

  if (!options.directed) {
    std::ofstream bidirected(options.output / "bidirected.csv");
    if (!bidirected) {
      throw std::runtime_error("cannot create bidirected.csv");
    }
    bidirected << "src,dst,weight\n";
    for (const auto& edge : undirected) {
      write_edge(bidirected, edge.first, edge.second, counts.num_nodes);
      write_edge(bidirected, edge.second, edge.first, counts.num_nodes);
    }
  }
  return counts;
}

static PreparedCounts prepare_contiguous(const Options& options) {
  std::ifstream input(options.input);
  if (!input) {
    throw std::runtime_error("cannot open " + options.input.string());
  }
  std::ofstream edges;
  std::ofstream reverse;
  std::ofstream undirected;
  std::ofstream bidirected;
  if (options.directed) {
    edges.open(options.output / "edges.csv");
    if (!options.skip_reverse) {
      reverse.open(options.output / "reverse.csv");
    }
  } else {
    undirected.open(options.output / "undirected.csv");
    bidirected.open(options.output / "bidirected.csv");
  }
  if ((options.directed && !edges) ||
      (options.directed && !options.skip_reverse && !reverse) ||
      (!options.directed && (!undirected || !bidirected))) {
    throw std::runtime_error("cannot create edge output files");
  }
  if (edges) edges << "src,dst,weight\n";
  if (reverse) reverse << "src,dst,weight\n";
  if (undirected) undirected << "src,dst,weight\n";
  if (bidirected) bidirected << "src,dst,weight\n";

  PreparedCounts counts;
  counts.num_nodes = options.num_nodes;
  bool dimensions_consumed = !options.matrix_market;
  std::string line;
  while (std::getline(input, line)) {
    if (ignored_line(line)) {
      continue;
    }
    std::istringstream parser(line);
    std::uint64_t src = 0;
    std::uint64_t dst = 0;
    if (!(parser >> src >> dst)) {
      continue;
    }
    if (!dimensions_consumed) {
      dimensions_consumed = true;
      continue;
    }
    if (src == 0 || dst == 0 || src > options.num_nodes ||
        dst > options.num_nodes) {
      throw std::runtime_error("one-based endpoint outside declared range");
    }
    --src;
    --dst;
    if (src == dst) {
      ++counts.self_loops_removed;
      continue;
    }
    if (options.directed) {
      write_edge(edges, src, dst, counts.num_nodes);
      if (reverse) {
        write_edge(reverse, dst, src, counts.num_nodes);
      }
      ++counts.directed_edges;
    } else {
      const auto lo = std::min(src, dst);
      const auto hi = std::max(src, dst);
      write_edge(undirected, lo, hi, counts.num_nodes);
      write_edge(bidirected, lo, hi, counts.num_nodes);
      write_edge(bidirected, hi, lo, counts.num_nodes);
      ++counts.undirected_edges;
    }
  }
  write_vertices(options.output / "vertices.csv", counts.num_nodes);
  return counts;
}

static PreparedCounts prepare_sparse_one_based(const Options& options) {
  std::ifstream first_pass(options.input);
  if (!first_pass) {
    throw std::runtime_error("cannot open " + options.input.string());
  }

  std::vector<std::uint8_t> present(1, 0);
  std::uint64_t observed_nodes = 0;
  std::string line;
  while (std::getline(first_pass, line)) {
    if (ignored_line(line)) {
      continue;
    }
    std::istringstream parser(line);
    std::uint64_t src = 0;
    std::uint64_t dst = 0;
    if (!(parser >> src >> dst)) {
      continue;
    }
    if (src == 0 || dst == 0) {
      throw std::runtime_error("sparse one-based input contains endpoint zero");
    }
    const auto required = std::max(src, dst) + 1;
    if (required > present.size()) {
      present.resize(required, 0);
    }
    if (!present[src]) {
      present[src] = 1;
      ++observed_nodes;
    }
    if (!present[dst]) {
      present[dst] = 1;
      ++observed_nodes;
    }
  }
  if (observed_nodes != options.num_nodes) {
    throw std::runtime_error(
        "sparse one-based input contains " + std::to_string(observed_nodes) +
        " observed nodes; expected " + std::to_string(options.num_nodes));
  }

  constexpr std::uint32_t missing = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> remap(present.size(), missing);
  std::uint32_t dense = 0;
  for (std::size_t source = 1; source < present.size(); ++source) {
    if (present[source]) {
      remap[source] = dense++;
    }
  }

  std::ifstream second_pass(options.input);
  if (!second_pass) {
    throw std::runtime_error("cannot reopen " + options.input.string());
  }
  std::ofstream edges;
  std::ofstream reverse;
  std::ofstream undirected;
  std::ofstream bidirected;
  if (options.directed) {
    edges.open(options.output / "edges.csv");
    if (!options.skip_reverse) {
      reverse.open(options.output / "reverse.csv");
    }
  } else {
    undirected.open(options.output / "undirected.csv");
    bidirected.open(options.output / "bidirected.csv");
  }
  if ((options.directed && !edges) ||
      (options.directed && !options.skip_reverse && !reverse) ||
      (!options.directed && (!undirected || !bidirected))) {
    throw std::runtime_error("cannot create sparse one-based edge outputs");
  }
  if (edges) edges << "src,dst,weight\n";
  if (reverse) reverse << "src,dst,weight\n";
  if (undirected) undirected << "src,dst,weight\n";
  if (bidirected) bidirected << "src,dst,weight\n";

  PreparedCounts counts;
  counts.num_nodes = observed_nodes;
  while (std::getline(second_pass, line)) {
    if (ignored_line(line)) {
      continue;
    }
    std::istringstream parser(line);
    std::uint64_t source_label = 0;
    std::uint64_t target_label = 0;
    if (!(parser >> source_label >> target_label)) {
      continue;
    }
    if (source_label >= remap.size() || target_label >= remap.size() ||
        remap[source_label] == missing || remap[target_label] == missing) {
      throw std::runtime_error("inconsistent endpoint between streaming passes");
    }
    const auto src = remap[source_label];
    const auto dst = remap[target_label];
    if (src == dst) {
      ++counts.self_loops_removed;
      continue;
    }
    if (options.directed) {
      write_edge(edges, src, dst, counts.num_nodes);
      if (reverse) {
        write_edge(reverse, dst, src, counts.num_nodes);
      }
      ++counts.directed_edges;
    } else {
      const auto lo = std::min(src, dst);
      const auto hi = std::max(src, dst);
      write_edge(undirected, lo, hi, counts.num_nodes);
      write_edge(bidirected, lo, hi, counts.num_nodes);
      write_edge(bidirected, hi, lo, counts.num_nodes);
      ++counts.undirected_edges;
    }
  }
  write_vertices(options.output / "vertices.csv", counts.num_nodes);
  return counts;
}

static std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (char c : value) {
    if (c == '\\' || c == '"') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

static void write_manifest(const Options& options, const PreparedCounts& counts) {
  std::ofstream out(options.output / "manifest.json");
  if (!out) {
    throw std::runtime_error("cannot create manifest.json");
  }
  const auto path_or_null = [&](const char* name, bool present) {
    if (!present) {
      return std::string("null");
    }
    return std::string("\"") +
           json_escape((options.output / name).string()) + "\"";
  };
  out << "{\n"
      << "  \"format\": \"graphscope-normalized-csv-v1\",\n"
      << "  \"name\": \"" << json_escape(options.name) << "\",\n"
      << "  \"graph_type\": \"" << (options.directed ? "directed" : "undirected")
      << "\",\n"
      << "  \"num_nodes\": " << counts.num_nodes << ",\n"
      << "  \"directed_edges\": " << counts.directed_edges << ",\n"
      << "  \"undirected_edges\": " << counts.undirected_edges << ",\n"
      << "  \"self_loops_removed\": " << counts.self_loops_removed << ",\n"
      << "  \"vertices\": "
      << path_or_null("vertices.csv", true) << ",\n"
      << "  \"edges\": "
      << path_or_null("edges.csv", options.directed) << ",\n"
      << "  \"reverse\": "
      << path_or_null("reverse.csv",
                      options.directed && !options.skip_reverse)
      << ",\n"
      << "  \"undirected\": "
      << path_or_null("undirected.csv",
                      !options.directed || !options.skip_undirected)
      << ",\n"
      << "  \"bidirected\": "
      << path_or_null("bidirected.csv", !options.directed) << ",\n"
      << "  \"source\": \"" << json_escape(options.input.string()) << "\",\n"
      << "  \"source_size_bytes\": " << fs::file_size(options.input) << ",\n"
      << "  \"source_mtime_ns\": "
      << fs::last_write_time(options.input).time_since_epoch().count() << ",\n"
      << "  \"preprocessing_in_timed_build\": false,\n"
      << "  \"normalization\": \"remove self loops, deduplicate generic inputs, "
         "and map source labels to deterministic zero-based contiguous IDs\"\n"
      << "}\n";
}

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);
    fs::create_directories(options.output);
    const PreparedCounts counts =
        options.sparse_one_based
            ? prepare_sparse_one_based(options)
            : (options.contiguous_one_based ? prepare_contiguous(options)
                                            : prepare_generic(options));
    write_manifest(options, counts);
    std::cout << (options.output / "manifest.json") << '\n';
    return 0;
  } catch (const std::exception& error) {
    usage(argv[0]);
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
