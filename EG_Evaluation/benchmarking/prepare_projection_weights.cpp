#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

struct Mapping {
    int fd = -1;
    void* data = MAP_FAILED;
    std::size_t bytes = 0;

    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping(Mapping&& other) noexcept
        : fd(other.fd), data(other.data), bytes(other.bytes) {
        other.fd = -1;
        other.data = MAP_FAILED;
        other.bytes = 0;
    }
    Mapping& operator=(Mapping&&) = delete;

    ~Mapping() {
        if (data != MAP_FAILED) {
            munmap(data, bytes);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
};

std::string argument(int argc, char** argv, const std::string& name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    throw std::invalid_argument("missing argument " + name);
}

std::int64_t parse_i64(const std::string& value, const char* label) {
    std::size_t consumed = 0;
    const long long parsed = std::stoll(value, &consumed);
    if (consumed != value.size() || parsed < 0) {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    return static_cast<std::int64_t>(parsed);
}

Mapping map_readonly(const std::string& path, std::size_t expected_bytes) {
    Mapping result;
    result.fd = open(path.c_str(), O_RDONLY);
    if (result.fd < 0) {
        throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    }
    struct stat metadata {};
    if (fstat(result.fd, &metadata) != 0 ||
        metadata.st_size < 0 ||
        static_cast<std::size_t>(metadata.st_size) != expected_bytes) {
        throw std::runtime_error("unexpected file size for " + path);
    }
    result.bytes = expected_bytes;
    result.data = mmap(nullptr, result.bytes, PROT_READ, MAP_SHARED, result.fd, 0);
    if (result.data == MAP_FAILED) {
        throw std::runtime_error("cannot mmap " + path + ": " + std::strerror(errno));
    }
    return result;
}

Mapping map_output(const std::string& path, std::size_t bytes) {
    Mapping result;
    result.fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (result.fd < 0) {
        throw std::runtime_error("cannot create " + path + ": " + std::strerror(errno));
    }
    if (ftruncate(result.fd, static_cast<off_t>(bytes)) != 0) {
        throw std::runtime_error("cannot size " + path + ": " + std::strerror(errno));
    }
    result.bytes = bytes;
    result.data = mmap(nullptr, result.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, result.fd, 0);
    if (result.data == MAP_FAILED) {
        throw std::runtime_error("cannot mmap output " + path + ": " + std::strerror(errno));
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string offsets_path = argument(argc, argv, "--offsets");
        const std::string indices_path = argument(argc, argv, "--indices");
        const std::string output_path = argument(argc, argv, "--output");
        const std::int64_t num_nodes =
            parse_i64(argument(argc, argv, "--num-nodes"), "num-nodes");
        const std::int64_t num_edges =
            parse_i64(argument(argc, argv, "--num-edges"), "num-edges");
        const int threads = static_cast<int>(
            parse_i64(argument(argc, argv, "--threads"), "threads"));
        if (num_nodes >= std::numeric_limits<int>::max() ||
            num_edges >= std::numeric_limits<int>::max()) {
            throw std::invalid_argument("projection exceeds the signed int32 ABI");
        }
        if (threads > 0) {
            omp_set_num_threads(threads);
        }

        Mapping offsets = map_readonly(
            offsets_path, static_cast<std::size_t>(num_nodes + 1) * sizeof(int));
        Mapping indices = map_readonly(
            indices_path, static_cast<std::size_t>(num_edges) * sizeof(int));
        const std::filesystem::path final_path(output_path);
        const std::filesystem::path partial_path =
            final_path.string() + ".incomplete-" + std::to_string(getpid());
        Mapping output = map_output(
            partial_path.string(),
            static_cast<std::size_t>(num_edges) * sizeof(double));

        const int* offset_data = static_cast<const int*>(offsets.data);
        const int* index_data = static_cast<const int*>(indices.data);
        double* weight_data = static_cast<double*>(output.data);
        if (offset_data[0] != 0 || offset_data[num_nodes] != num_edges) {
            throw std::runtime_error("projection offsets have invalid endpoints");
        }

        std::int64_t invalid = 0;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : invalid)
        for (std::int64_t source = 0; source < num_nodes; ++source) {
            const int begin = offset_data[source];
            const int end = offset_data[source + 1];
            if (begin < 0 || end < begin || end > num_edges) {
                ++invalid;
                continue;
            }
            for (int position = begin; position < end; ++position) {
                const int target = index_data[position];
                if (target <= source || target >= num_nodes) {
                    ++invalid;
                    continue;
                }
                const std::uint64_t product =
                    static_cast<std::uint64_t>(source) *
                    static_cast<std::uint64_t>(target);
                weight_data[position] = 1.0 +
                    static_cast<double>(
                        product % static_cast<std::uint64_t>(num_nodes));
            }
        }
        if (invalid != 0) {
            throw std::runtime_error(
                "projection contains " + std::to_string(invalid) +
                " invalid edge records");
        }
        if (msync(output.data, output.bytes, MS_SYNC) != 0 ||
            fsync(output.fd) != 0) {
            throw std::runtime_error("failed to persist projection weights");
        }
        if (munmap(output.data, output.bytes) != 0) {
            throw std::runtime_error("failed to unmap projection weights");
        }
        output.data = MAP_FAILED;
        close(output.fd);
        output.fd = -1;
        if (rename(partial_path.c_str(), final_path.c_str()) != 0) {
            throw std::runtime_error("failed to finalize projection weights");
        }
        std::cout << final_path << " " << num_edges << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 2;
    }
}
