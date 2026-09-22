#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t kSentinel = std::numeric_limits<std::uint64_t>::max();
constexpr std::size_t kRadixBits = 16;
constexpr std::size_t kRadixBuckets = 1ULL << kRadixBits;
constexpr std::size_t kRadixPasses = 64 / kRadixBits;

struct Options {
    fs::path offsets_path;
    fs::path indices_path;
    fs::path output_dir;
    fs::path work_dir;
    fs::path allowed_root;
    std::string name;
    std::int64_t num_nodes = -1;
    std::int64_t num_entries = -1;
    int threads = 0;
    bool force = false;
};

struct PhaseTimes {
    double input_validation = 0.0;
    double canonicalization = 0.0;
    double canonical_sort = 0.0;
    double deduplication = 0.0;
    double lower_view = 0.0;
    double upper_view = 0.0;
    double degree_and_forward = 0.0;
    double validation = 0.0;
    double write = 0.0;
    double total = 0.0;
};

struct ProjectionStats {
    std::int64_t self_loops_removed = 0;
    std::int64_t non_self_loop_arcs = 0;
    std::int64_t unique_edge_count = 0;
    std::int64_t duplicate_or_reciprocal_arcs_removed = 0;
    std::int64_t isolated_nodes = 0;
    int max_degree = 0;
    int max_forward_degree = 0;
    int threads = 1;
    std::string sort_backend;
};

class Timer {
public:
    Timer() : started_(std::chrono::steady_clock::now()) {}

    double seconds() const {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - started_)
            .count();
    }

private:
    std::chrono::steady_clock::time_point started_;
};

template <typename T>
class MappedArray {
public:
    MappedArray(const fs::path& path, std::size_t expected_items)
        : path_(path), count_(expected_items) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open input array: " + path.string());
        }
        struct stat info {};
        if (::fstat(fd_, &info) != 0) {
            close_file();
            throw std::runtime_error("cannot stat input array: " + path.string());
        }
        const std::uint64_t expected_bytes =
            static_cast<std::uint64_t>(expected_items) * sizeof(T);
        if (info.st_size < 0 ||
            static_cast<std::uint64_t>(info.st_size) != expected_bytes) {
            const auto found = info.st_size;
            close_file();
            std::ostringstream message;
            message << "input array size mismatch for " << path << ": expected "
                    << expected_bytes << " bytes, found " << found;
            throw std::runtime_error(message.str());
        }
        if (expected_bytes == 0) {
            return;
        }
        void* mapping = ::mmap(
            nullptr,
            static_cast<std::size_t>(expected_bytes),
            PROT_READ,
            MAP_PRIVATE,
            fd_,
            0);
        if (mapping == MAP_FAILED) {
            close_file();
            throw std::runtime_error("cannot mmap input array: " + path.string());
        }
        data_ = static_cast<const T*>(mapping);
        bytes_ = static_cast<std::size_t>(expected_bytes);
#ifdef MADV_SEQUENTIAL
        ::madvise(const_cast<T*>(data_), bytes_, MADV_SEQUENTIAL);
#endif
    }

    MappedArray(const MappedArray&) = delete;
    MappedArray& operator=(const MappedArray&) = delete;

    ~MappedArray() {
        if (data_ != nullptr) {
            ::munmap(const_cast<T*>(data_), bytes_);
        }
        close_file();
    }

    const T* data() const { return data_; }
    const T& operator[](std::size_t index) const { return data_[index]; }
    std::size_t size() const { return count_; }

private:
    void close_file() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    fs::path path_;
    int fd_ = -1;
    const T* data_ = nullptr;
    std::size_t count_ = 0;
    std::size_t bytes_ = 0;
};

class WorkRunGuard {
public:
    explicit WorkRunGuard(fs::path path) : path_(std::move(path)) {}
    WorkRunGuard(const WorkRunGuard&) = delete;
    WorkRunGuard& operator=(const WorkRunGuard&) = delete;
    ~WorkRunGuard() {
        std::error_code error;
        fs::remove_all(path_, error);
    }
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    escaped << "\\u" << std::hex << std::setw(4)
                            << std::setfill('0') << static_cast<int>(ch)
                            << std::dec;
                } else {
                    escaped << static_cast<char>(ch);
                }
        }
    }
    return escaped.str();
}

bool path_is_within(const fs::path& root, const fs::path& candidate) {
    auto root_it = root.begin();
    auto candidate_it = candidate.begin();
    for (; root_it != root.end(); ++root_it, ++candidate_it) {
        if (candidate_it == candidate.end() || *candidate_it != *root_it) {
            return false;
        }
    }
    return true;
}

fs::path canonical_directory_under(
    const fs::path& allowed_root,
    const fs::path& value,
    const char* label) {
    fs::create_directories(value);
    const fs::path resolved = fs::weakly_canonical(value);
    if (!path_is_within(allowed_root, resolved)) {
        throw std::runtime_error(
            std::string(label) + " must be inside --allowed-root: " +
            resolved.string());
    }
    return resolved;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            return argv[i];
        };
        if (argument == "--offsets") {
            options.offsets_path = next();
        } else if (argument == "--indices") {
            options.indices_path = next();
        } else if (argument == "--num-nodes") {
            options.num_nodes = std::stoll(next());
        } else if (argument == "--num-entries") {
            options.num_entries = std::stoll(next());
        } else if (argument == "--output-dir") {
            options.output_dir = next();
        } else if (argument == "--work-dir") {
            options.work_dir = next();
        } else if (argument == "--allowed-root") {
            options.allowed_root = next();
        } else if (argument == "--name") {
            options.name = next();
        } else if (argument == "--threads") {
            options.threads = std::stoi(next());
        } else if (argument == "--force") {
            options.force = true;
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }

    if (options.offsets_path.empty() || options.indices_path.empty() ||
        options.output_dir.empty() || options.work_dir.empty() ||
        options.allowed_root.empty() || options.name.empty()) {
        throw std::runtime_error(
            "--offsets, --indices, --num-nodes, --num-entries, --output-dir, "
            "--work-dir, --allowed-root, and --name are required");
    }
    if (options.num_nodes < 0 ||
        options.num_nodes >= std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "--num-nodes must fit the signed int32 CSR ABI");
    }
    if (options.num_entries < 0 ||
        options.num_entries >= std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "--num-entries must fit the signed int32 CSR ABI");
    }
    if (options.threads < 0) {
        throw std::runtime_error("--threads must be nonnegative");
    }
    if (options.name.find('/') != std::string::npos ||
        options.name.find('\\') != std::string::npos) {
        throw std::runtime_error("--name must not contain path separators");
    }
    return options;
}

int configure_threads(int requested) {
#ifdef _OPENMP
    omp_set_dynamic(0);
    int available = std::max(1, omp_get_max_threads());
    int selected = requested > 0 ? std::min(requested, available) : available;
    omp_set_num_threads(selected);
    return selected;
#else
    (void)requested;
    return 1;
#endif
}

std::uint64_t edge_key(int lower, int upper) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(lower))
            << 32) |
        static_cast<std::uint32_t>(upper);
}

int key_source(std::uint64_t key) {
    return static_cast<int>(static_cast<std::uint32_t>(key >> 32));
}

int key_target(std::uint64_t key) {
    return static_cast<int>(static_cast<std::uint32_t>(key));
}

void parallel_radix_sort(
    std::vector<std::uint64_t>& values,
    std::vector<std::uint64_t>& scratch,
    int thread_count) {
    const std::size_t item_count = values.size();
    scratch.resize(item_count);
    if (item_count < 2) {
        return;
    }

    std::vector<std::uint64_t> counts(
        static_cast<std::size_t>(thread_count) * kRadixBuckets);
    std::vector<std::uint64_t> positions(counts.size());
    std::vector<std::uint64_t> bucket_starts(kRadixBuckets);

    for (std::size_t pass = 0; pass < kRadixPasses; ++pass) {
        std::fill(counts.begin(), counts.end(), 0);
        const unsigned shift = static_cast<unsigned>(pass * kRadixBits);

#ifdef _OPENMP
#pragma omp parallel num_threads(thread_count)
#endif
        {
#ifdef _OPENMP
            const int thread_id = omp_get_thread_num();
#else
            const int thread_id = 0;
#endif
            const std::size_t begin =
                item_count * static_cast<std::size_t>(thread_id) /
                static_cast<std::size_t>(thread_count);
            const std::size_t end =
                item_count * static_cast<std::size_t>(thread_id + 1) /
                static_cast<std::size_t>(thread_count);
            std::uint64_t* local_counts =
                counts.data() +
                static_cast<std::size_t>(thread_id) * kRadixBuckets;
            for (std::size_t index = begin; index < end; ++index) {
                const std::size_t bucket =
                    static_cast<std::size_t>((values[index] >> shift) &
                                             (kRadixBuckets - 1));
                ++local_counts[bucket];
            }
        }

        std::uint64_t global_cursor = 0;
        for (std::size_t bucket = 0; bucket < kRadixBuckets; ++bucket) {
            bucket_starts[bucket] = global_cursor;
            std::uint64_t thread_cursor = global_cursor;
            for (int thread_id = 0; thread_id < thread_count; ++thread_id) {
                const std::size_t index =
                    static_cast<std::size_t>(thread_id) * kRadixBuckets +
                    bucket;
                positions[index] = thread_cursor;
                thread_cursor += counts[index];
            }
            global_cursor = thread_cursor;
        }
        if (global_cursor != item_count) {
            throw std::runtime_error("radix histogram lost input items");
        }

#ifdef _OPENMP
#pragma omp parallel num_threads(thread_count)
#endif
        {
#ifdef _OPENMP
            const int thread_id = omp_get_thread_num();
#else
            const int thread_id = 0;
#endif
            const std::size_t begin =
                item_count * static_cast<std::size_t>(thread_id) /
                static_cast<std::size_t>(thread_count);
            const std::size_t end =
                item_count * static_cast<std::size_t>(thread_id + 1) /
                static_cast<std::size_t>(thread_count);
            std::uint64_t* local_positions =
                positions.data() +
                static_cast<std::size_t>(thread_id) * kRadixBuckets;
            for (std::size_t index = begin; index < end; ++index) {
                const std::uint64_t value = values[index];
                const std::size_t bucket =
                    static_cast<std::size_t>((value >> shift) &
                                             (kRadixBuckets - 1));
                scratch[static_cast<std::size_t>(
                    local_positions[bucket]++)] = value;
            }
        }
        values.swap(scratch);
    }
}

template <typename T>
void write_raw(const fs::path& path, const std::vector<T>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create output: " + path.string());
    }
    constexpr std::size_t kChunkBytes = 1ULL << 30;
    const char* data = reinterpret_cast<const char*>(values.data());
    const std::size_t total_bytes = values.size() * sizeof(T);
    std::size_t written = 0;
    while (written < total_bytes) {
        const std::size_t count =
            std::min(kChunkBytes, total_bytes - written);
        output.write(data + written, static_cast<std::streamsize>(count));
        if (!output) {
            throw std::runtime_error(
                "failed writing output: " + path.string());
        }
        written += count;
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed closing output: " + path.string());
    }
}

void build_offsets(
    const std::vector<std::uint64_t>& sorted_keys,
    std::int64_t num_nodes,
    std::vector<int>& offsets,
    std::vector<int>& indices) {
    offsets.assign(static_cast<std::size_t>(num_nodes) + 1, 0);
    indices.resize(sorted_keys.size());
    for (std::size_t index = 0; index < sorted_keys.size(); ++index) {
        const int source = key_source(sorted_keys[index]);
        const int target = key_target(sorted_keys[index]);
        if (offsets[static_cast<std::size_t>(source) + 1] ==
            std::numeric_limits<int>::max()) {
            throw std::runtime_error("one projection row exceeds int32");
        }
        ++offsets[static_cast<std::size_t>(source) + 1];
        indices[index] = target;
    }
    std::int64_t prefix = 0;
    for (std::int64_t node = 0; node < num_nodes; ++node) {
        prefix += offsets[static_cast<std::size_t>(node) + 1];
        if (prefix >= std::numeric_limits<int>::max()) {
            throw std::runtime_error(
                "one projection half exceeds the signed int32 CSR limit");
        }
        offsets[static_cast<std::size_t>(node) + 1] =
            static_cast<int>(prefix);
    }
}

bool rank_less(int source, int target, const std::vector<int>& degree) {
    const int source_degree = degree[static_cast<std::size_t>(source)];
    const int target_degree = degree[static_cast<std::size_t>(target)];
    return source_degree < target_degree ||
        (source_degree == target_degree && source < target);
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

struct Fingerprint {
    std::uint64_t count = 0;
    std::uint64_t sum = 0;
    std::uint64_t xor_value = 0;

    void add(int first, int second) {
        const int lower = std::min(first, second);
        const int upper = std::max(first, second);
        const std::uint64_t hashed = splitmix64(edge_key(lower, upper));
        ++count;
        sum += hashed;
        xor_value ^= hashed;
    }

    bool operator==(const Fingerprint& other) const {
        return count == other.count && sum == other.sum &&
            xor_value == other.xor_value;
    }
};

Fingerprint validate_half_view(
    const char* name,
    const std::vector<int>& offsets,
    const std::vector<int>& indices,
    std::int64_t num_nodes,
    bool lower_to_upper) {
    Fingerprint fingerprint;
    if (offsets.size() != static_cast<std::size_t>(num_nodes) + 1 ||
        offsets.empty() || offsets.front() != 0 ||
        offsets.back() != static_cast<int>(indices.size())) {
        throw std::runtime_error(std::string("invalid ") + name + " offsets");
    }
    int previous_offset = 0;
    for (std::int64_t source = 0; source < num_nodes; ++source) {
        const int begin = offsets[static_cast<std::size_t>(source)];
        const int end = offsets[static_cast<std::size_t>(source) + 1];
        if (begin < previous_offset || end < begin ||
            end > static_cast<int>(indices.size())) {
            throw std::runtime_error(
                std::string("nonmonotonic ") + name + " offsets");
        }
        int previous_target = -1;
        for (int position = begin; position < end; ++position) {
            const int target = indices[static_cast<std::size_t>(position)];
            if (target < 0 || target >= num_nodes ||
                (lower_to_upper && target <= source) ||
                (!lower_to_upper && target >= source) ||
                target <= previous_target) {
                throw std::runtime_error(
                    std::string("invalid or duplicate edge in ") + name);
            }
            previous_target = target;
            fingerprint.add(static_cast<int>(source), target);
        }
        previous_offset = end;
    }
    return fingerprint;
}

Fingerprint validate_forward_view(
    const std::vector<int>& offsets,
    const std::vector<int>& indices,
    const std::vector<int>& degree,
    std::int64_t num_nodes) {
    Fingerprint fingerprint;
    if (offsets.size() != static_cast<std::size_t>(num_nodes) + 1 ||
        offsets.empty() || offsets.front() != 0 ||
        offsets.back() != static_cast<int>(indices.size())) {
        throw std::runtime_error("invalid degree-oriented forward offsets");
    }
    for (std::int64_t source = 0; source < num_nodes; ++source) {
        const int begin = offsets[static_cast<std::size_t>(source)];
        const int end = offsets[static_cast<std::size_t>(source) + 1];
        if (begin < 0 || end < begin ||
            end > static_cast<int>(indices.size())) {
            throw std::runtime_error(
                "nonmonotonic degree-oriented forward offsets");
        }
        int previous_target = -1;
        for (int position = begin; position < end; ++position) {
            const int target = indices[static_cast<std::size_t>(position)];
            if (target < 0 || target >= num_nodes ||
                target <= previous_target ||
                !rank_less(static_cast<int>(source), target, degree)) {
                throw std::runtime_error(
                    "invalid edge in degree-oriented forward CSR");
            }
            previous_target = target;
            fingerprint.add(static_cast<int>(source), target);
        }
    }
    return fingerprint;
}

std::int64_t max_rss_kib() {
    struct rusage usage {};
    if (::getrusage(RUSAGE_SELF, &usage) != 0) {
        return -1;
    }
    return static_cast<std::int64_t>(usage.ru_maxrss);
}

std::string output_stem(const std::string& name) {
    return name + ".logical-undirected";
}

void write_core_manifest(
    const fs::path& path,
    const Options& options,
    const ProjectionStats& stats,
    const PhaseTimes& times,
    const std::vector<std::pair<std::string, fs::path>>& artifacts) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "cannot create core manifest: " + path.string());
    }
    auto artifact_name = [&](const std::string& role) -> std::string {
        for (const auto& item : artifacts) {
            if (item.first == role) {
                return item.second.filename().string();
            }
        }
        throw std::runtime_error("missing artifact role: " + role);
    };

    output << std::setprecision(17)
           << "{\n"
           << "  \"format\": \"eggpu-logical-undirected-projection-v1\",\n"
           << "  \"generation\": 1,\n"
           << "  \"name\": \"" << json_escape(options.name) << "\",\n"
           << "  \"num_nodes\": " << options.num_nodes << ",\n"
           << "  \"source_num_entries\": " << options.num_entries << ",\n"
           << "  \"non_self_loop_arcs\": " << stats.non_self_loop_arcs
           << ",\n"
           << "  \"self_loops_removed\": " << stats.self_loops_removed
           << ",\n"
           << "  \"duplicate_or_reciprocal_arcs_removed\": "
           << stats.duplicate_or_reciprocal_arcs_removed << ",\n"
           << "  \"unique_edge_count\": " << stats.unique_edge_count
           << ",\n"
           << "  \"isolated_nodes\": " << stats.isolated_nodes << ",\n"
           << "  \"max_degree\": " << stats.max_degree << ",\n"
           << "  \"max_forward_degree\": " << stats.max_forward_degree
           << ",\n"
           << "  \"node_labels\": \"zero_based_contiguous\",\n"
           << "  \"offset_dtype\": \"int32\",\n"
           << "  \"index_dtype\": \"int32\",\n"
           << "  \"degree_dtype\": \"int32\",\n"
           << "  \"canonical_edge_rule\": \"store {u,v} once with u < v\",\n"
           << "  \"self_loop_policy\": \"remove\",\n"
           << "  \"duplicate_policy\": \"collapse directed duplicates and reciprocal arcs\",\n"
           << "  \"forward_orientation\": \"lexicographic (degree, node_id) from lower rank to higher rank\",\n"
           << "  \"lower_V_path\": \""
           << json_escape(artifact_name("lower_V")) << "\",\n"
           << "  \"lower_E_path\": \""
           << json_escape(artifact_name("lower_E")) << "\",\n"
           << "  \"upper_V_path\": \""
           << json_escape(artifact_name("upper_V")) << "\",\n"
           << "  \"upper_E_path\": \""
           << json_escape(artifact_name("upper_E")) << "\",\n"
           << "  \"degree_path\": \""
           << json_escape(artifact_name("degree")) << "\",\n"
           << "  \"forward_V_path\": \""
           << json_escape(artifact_name("forward_V")) << "\",\n"
           << "  \"forward_E_path\": \""
           << json_escape(artifact_name("forward_E")) << "\",\n"
           << "  \"preprocessing_core\": {\n"
           << "    \"sort_backend\": \"" << stats.sort_backend << "\",\n"
           << "    \"threads\": " << stats.threads << ",\n"
           << "    \"radix_bits_per_pass\": " << kRadixBits << ",\n"
           << "    \"radix_passes\": " << kRadixPasses << ",\n"
           << "    \"max_rss_kib\": " << max_rss_kib() << ",\n"
           << "    \"input_validation_seconds\": " << times.input_validation
           << ",\n"
           << "    \"canonicalization_seconds\": " << times.canonicalization
           << ",\n"
           << "    \"canonical_sort_seconds\": " << times.canonical_sort
           << ",\n"
           << "    \"deduplication_seconds\": " << times.deduplication
           << ",\n"
           << "    \"lower_view_seconds\": " << times.lower_view << ",\n"
           << "    \"upper_view_seconds\": " << times.upper_view << ",\n"
           << "    \"degree_and_forward_seconds\": "
           << times.degree_and_forward << ",\n"
           << "    \"validation_seconds\": " << times.validation << ",\n"
           << "    \"write_seconds\": " << times.write << ",\n"
           << "    \"total_seconds\": " << times.total << "\n"
           << "  }\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Timer total_timer;
        Options options = parse_options(argc, argv);
        const fs::path allowed_root =
            fs::weakly_canonical(options.allowed_root);
        if (!fs::is_directory(allowed_root)) {
            throw std::runtime_error("--allowed-root is not a directory");
        }
        options.output_dir = canonical_directory_under(
            allowed_root, fs::absolute(options.output_dir), "--output-dir");
        options.work_dir = canonical_directory_under(
            allowed_root, fs::absolute(options.work_dir), "--work-dir");
        options.offsets_path = fs::weakly_canonical(options.offsets_path);
        options.indices_path = fs::weakly_canonical(options.indices_path);

        const int thread_count = configure_threads(options.threads);
        ProjectionStats stats;
        stats.threads = thread_count;
#ifdef _OPENMP
        stats.sort_backend = "openmp-lsd-radix-sort";
#else
        stats.sort_backend = "serial-lsd-radix-sort";
#endif
        PhaseTimes times;

        const std::string stem = output_stem(options.name);
        const std::vector<std::pair<std::string, fs::path>> final_artifacts = {
            {"lower_V", options.output_dir / (stem + ".lower_V.i32")},
            {"lower_E", options.output_dir / (stem + ".lower_E.i32")},
            {"upper_V", options.output_dir / (stem + ".upper_V.i32")},
            {"upper_E", options.output_dir / (stem + ".upper_E.i32")},
            {"degree", options.output_dir / (stem + ".degree.i32")},
            {"forward_V", options.output_dir / (stem + ".forward_V.i32")},
            {"forward_E", options.output_dir / (stem + ".forward_E.i32")},
        };
        const fs::path final_manifest =
            options.output_dir / (stem + ".json");
        if (!options.force) {
            for (const auto& artifact : final_artifacts) {
                if (fs::exists(artifact.second)) {
                    throw std::runtime_error(
                        "output already exists; pass --force to replace it: " +
                        artifact.second.string());
                }
            }
            if (fs::exists(final_manifest)) {
                throw std::runtime_error(
                    "manifest already exists; pass --force to replace it: " +
                    final_manifest.string());
            }
        }

        const fs::path run_dir = options.work_dir /
            ("projection-" + std::to_string(::getpid()) + "-" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(run_dir);
        if (!path_is_within(allowed_root, fs::weakly_canonical(run_dir))) {
            throw std::runtime_error(
                "derived work run directory escaped --allowed-root");
        }
        WorkRunGuard work_guard(run_dir);

        Timer phase_timer;
        MappedArray<int> offsets(
            options.offsets_path,
            static_cast<std::size_t>(options.num_nodes) + 1);
        MappedArray<int> indices(
            options.indices_path,
            static_cast<std::size_t>(options.num_entries));
        if (offsets[0] != 0 ||
            offsets[static_cast<std::size_t>(options.num_nodes)] !=
                options.num_entries) {
            throw std::runtime_error(
                "source CSR offsets must start at zero and end at num_entries");
        }
        int previous_offset = 0;
        for (std::int64_t node = 0; node <= options.num_nodes; ++node) {
            const int offset = offsets[static_cast<std::size_t>(node)];
            if (offset < previous_offset || offset < 0 ||
                offset > options.num_entries) {
                throw std::runtime_error(
                    "source CSR offsets are nonmonotonic or out of range");
            }
            previous_offset = offset;
        }
        times.input_validation = phase_timer.seconds();

        phase_timer = Timer();
        std::vector<std::uint64_t> canonical_keys(
            static_cast<std::size_t>(options.num_entries), kSentinel);
        std::int64_t self_loops = 0;
        int invalid_endpoint = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024) num_threads(thread_count) \
    reduction(+ : self_loops) reduction(+ : invalid_endpoint)
#endif
        for (std::int64_t source = 0; source < options.num_nodes; ++source) {
            const int begin = offsets[static_cast<std::size_t>(source)];
            const int end = offsets[static_cast<std::size_t>(source) + 1];
            for (int position = begin; position < end; ++position) {
                const int target =
                    indices[static_cast<std::size_t>(position)];
                if (target < 0 || target >= options.num_nodes) {
                    ++invalid_endpoint;
                    continue;
                }
                if (target == source) {
                    ++self_loops;
                    continue;
                }
                const int lower =
                    std::min(static_cast<int>(source), target);
                const int upper =
                    std::max(static_cast<int>(source), target);
                canonical_keys[static_cast<std::size_t>(position)] =
                    edge_key(lower, upper);
            }
        }
        if (invalid_endpoint != 0) {
            throw std::runtime_error(
                "source CSR contains endpoints outside [0, num_nodes)");
        }
        stats.self_loops_removed = self_loops;
        stats.non_self_loop_arcs = options.num_entries - self_loops;
        times.canonicalization = phase_timer.seconds();

        phase_timer = Timer();
        std::vector<std::uint64_t> scratch;
        parallel_radix_sort(canonical_keys, scratch, thread_count);
        times.canonical_sort = phase_timer.seconds();

        phase_timer = Timer();
        const std::size_t valid_count =
            static_cast<std::size_t>(stats.non_self_loop_arcs);
        if (valid_count > canonical_keys.size()) {
            throw std::runtime_error(
                "self-loop accounting exceeds source entry count");
        }
        if (valid_count < canonical_keys.size() &&
            canonical_keys[valid_count] != kSentinel) {
            throw std::runtime_error(
                "radix sort did not place removed self-loops at the end");
        }
        std::size_t unique_count = 0;
        for (std::size_t index = 0; index < valid_count; ++index) {
            const std::uint64_t key = canonical_keys[index];
            if (unique_count == 0 ||
                key != canonical_keys[unique_count - 1]) {
                canonical_keys[unique_count++] = key;
            }
        }
        if (unique_count >=
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "unique undirected edges exceed one int32 projection half");
        }
        canonical_keys.resize(unique_count);
        stats.unique_edge_count = static_cast<std::int64_t>(unique_count);
        stats.duplicate_or_reciprocal_arcs_removed =
            stats.non_self_loop_arcs - stats.unique_edge_count;
        times.deduplication = phase_timer.seconds();

        phase_timer = Timer();
        std::vector<int> lower_V;
        std::vector<int> lower_E;
        build_offsets(
            canonical_keys,
            options.num_nodes,
            lower_V,
            lower_E);
        times.lower_view = phase_timer.seconds();

        phase_timer = Timer();
        scratch.resize(unique_count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count)
#endif
        for (std::int64_t index = 0;
             index < static_cast<std::int64_t>(unique_count);
             ++index) {
            const std::uint64_t key =
                canonical_keys[static_cast<std::size_t>(index)];
            scratch[static_cast<std::size_t>(index)] =
                edge_key(key_target(key), key_source(key));
        }
        canonical_keys.resize(unique_count);
        parallel_radix_sort(scratch, canonical_keys, thread_count);
        std::vector<int> upper_V;
        std::vector<int> upper_E;
        build_offsets(
            scratch,
            options.num_nodes,
            upper_V,
            upper_E);
        times.upper_view = phase_timer.seconds();

        phase_timer = Timer();
        std::vector<int> degree(
            static_cast<std::size_t>(options.num_nodes), 0);
        for (std::int64_t node = 0; node < options.num_nodes; ++node) {
            const std::int64_t lower_degree =
                static_cast<std::int64_t>(
                    lower_V[static_cast<std::size_t>(node) + 1]) -
                lower_V[static_cast<std::size_t>(node)];
            const std::int64_t upper_degree =
                static_cast<std::int64_t>(
                    upper_V[static_cast<std::size_t>(node) + 1]) -
                upper_V[static_cast<std::size_t>(node)];
            const std::int64_t total_degree = lower_degree + upper_degree;
            if (total_degree >= std::numeric_limits<int>::max()) {
                throw std::runtime_error(
                    "one undirected projection degree exceeds int32");
            }
            degree[static_cast<std::size_t>(node)] =
                static_cast<int>(total_degree);
            if (total_degree == 0) {
                ++stats.isolated_nodes;
            }
            stats.max_degree =
                std::max(stats.max_degree, static_cast<int>(total_degree));
        }

        std::vector<int> forward_V(
            static_cast<std::size_t>(options.num_nodes) + 1, 0);
        for (std::int64_t lower = 0; lower < options.num_nodes; ++lower) {
            const int begin = lower_V[static_cast<std::size_t>(lower)];
            const int end = lower_V[static_cast<std::size_t>(lower) + 1];
            for (int position = begin; position < end; ++position) {
                const int upper =
                    lower_E[static_cast<std::size_t>(position)];
                const int source = rank_less(
                                       static_cast<int>(lower),
                                       upper,
                                       degree)
                    ? static_cast<int>(lower)
                    : upper;
                if (forward_V[static_cast<std::size_t>(source) + 1] ==
                    std::numeric_limits<int>::max()) {
                    throw std::runtime_error(
                        "one forward CSR row exceeds int32");
                }
                ++forward_V[static_cast<std::size_t>(source) + 1];
            }
        }
        std::int64_t forward_prefix = 0;
        for (std::int64_t node = 0; node < options.num_nodes; ++node) {
            const int row_count =
                forward_V[static_cast<std::size_t>(node) + 1];
            stats.max_forward_degree =
                std::max(stats.max_forward_degree, row_count);
            forward_prefix += row_count;
            if (forward_prefix >= std::numeric_limits<int>::max()) {
                throw std::runtime_error(
                    "forward CSR entries exceed the signed int32 limit");
            }
            forward_V[static_cast<std::size_t>(node) + 1] =
                static_cast<int>(forward_prefix);
        }
        if (forward_prefix != stats.unique_edge_count) {
            throw std::runtime_error(
                "forward orientation did not preserve every unique edge");
        }

        std::vector<int> forward_E(unique_count);
        std::vector<int> forward_cursor(
            forward_V.begin(), forward_V.end() - 1);
        for (std::int64_t lower = 0; lower < options.num_nodes; ++lower) {
            const int begin = lower_V[static_cast<std::size_t>(lower)];
            const int end = lower_V[static_cast<std::size_t>(lower) + 1];
            for (int position = begin; position < end; ++position) {
                const int upper =
                    lower_E[static_cast<std::size_t>(position)];
                const bool lower_first = rank_less(
                    static_cast<int>(lower), upper, degree);
                const int source =
                    lower_first ? static_cast<int>(lower) : upper;
                const int target =
                    lower_first ? upper : static_cast<int>(lower);
                const int output_position =
                    forward_cursor[static_cast<std::size_t>(source)]++;
                forward_E[static_cast<std::size_t>(output_position)] =
                    target;
            }
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4096) num_threads(thread_count)
#endif
        for (std::int64_t node = 0; node < options.num_nodes; ++node) {
            const int begin = forward_V[static_cast<std::size_t>(node)];
            const int end = forward_V[static_cast<std::size_t>(node) + 1];
            if (end - begin > 1) {
                std::sort(
                    forward_E.begin() + begin,
                    forward_E.begin() + end);
            }
        }
        times.degree_and_forward = phase_timer.seconds();

        phase_timer = Timer();
        const Fingerprint lower_fingerprint = validate_half_view(
            "lower half CSR",
            lower_V,
            lower_E,
            options.num_nodes,
            true);
        const Fingerprint upper_fingerprint = validate_half_view(
            "upper half CSR",
            upper_V,
            upper_E,
            options.num_nodes,
            false);
        const Fingerprint forward_fingerprint = validate_forward_view(
            forward_V,
            forward_E,
            degree,
            options.num_nodes);
        if (!(lower_fingerprint == upper_fingerprint) ||
            !(lower_fingerprint == forward_fingerprint) ||
            lower_fingerprint.count !=
                static_cast<std::uint64_t>(stats.unique_edge_count)) {
            throw std::runtime_error(
                "projection views do not represent the same edge set");
        }
        std::int64_t degree_sum = 0;
        for (const int value : degree) {
            if (value < 0) {
                throw std::runtime_error(
                    "projection contains a negative degree");
            }
            degree_sum += value;
        }
        if (degree_sum != 2 * stats.unique_edge_count) {
            throw std::runtime_error(
                "projection degree sum does not equal 2|E|");
        }
        times.validation = phase_timer.seconds();

        const std::vector<std::pair<std::string, fs::path>> work_artifacts = {
            {"lower_V", run_dir / final_artifacts[0].second.filename()},
            {"lower_E", run_dir / final_artifacts[1].second.filename()},
            {"upper_V", run_dir / final_artifacts[2].second.filename()},
            {"upper_E", run_dir / final_artifacts[3].second.filename()},
            {"degree", run_dir / final_artifacts[4].second.filename()},
            {"forward_V", run_dir / final_artifacts[5].second.filename()},
            {"forward_E", run_dir / final_artifacts[6].second.filename()},
        };

        phase_timer = Timer();
        write_raw(work_artifacts[0].second, lower_V);
        write_raw(work_artifacts[1].second, lower_E);
        write_raw(work_artifacts[2].second, upper_V);
        write_raw(work_artifacts[3].second, upper_E);
        write_raw(work_artifacts[4].second, degree);
        write_raw(work_artifacts[5].second, forward_V);
        write_raw(work_artifacts[6].second, forward_E);

        for (std::size_t index = 0; index < final_artifacts.size(); ++index) {
            if (options.force && fs::exists(final_artifacts[index].second)) {
                fs::remove(final_artifacts[index].second);
            }
            fs::rename(
                work_artifacts[index].second,
                final_artifacts[index].second);
        }
        times.write = phase_timer.seconds();
        times.total = total_timer.seconds();
        const fs::path work_manifest =
            run_dir / final_manifest.filename();
        write_core_manifest(
            work_manifest, options, stats, times, final_artifacts);
        if (options.force && fs::exists(final_manifest)) {
            fs::remove(final_manifest);
        }
        fs::rename(work_manifest, final_manifest);

        std::cout << "Wrote " << final_manifest
                  << " source_arcs=" << options.num_entries
                  << " self_loops_removed=" << stats.self_loops_removed
                  << " unique_undirected_edges=" << stats.unique_edge_count
                  << " collapsed_arcs="
                  << stats.duplicate_or_reciprocal_arcs_removed
                  << " threads=" << stats.threads
                  << " total_seconds=" << times.total << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
