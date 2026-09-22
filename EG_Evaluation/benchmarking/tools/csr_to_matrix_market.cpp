#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

template <typename T> class MappedFile {
 public:
  MappedFile(const std::string& path, std::size_t count) : count_(count) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    const std::size_t bytes = count * sizeof(T);
    void* ptr = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (ptr == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);
    data_ = static_cast<const T*>(ptr);
    bytes_ = bytes;
    ::madvise(const_cast<T*>(data_), bytes_, MADV_SEQUENTIAL);
  }
  ~MappedFile() {
    if (data_) ::munmap(const_cast<T*>(data_), bytes_);
    if (fd_ >= 0) ::close(fd_);
  }
  const T* data() const { return data_; }
 private:
  int fd_ = -1;
  const T* data_ = nullptr;
  std::size_t count_ = 0;
  std::size_t bytes_ = 0;
};

std::string value(int argc, char** argv, const std::string& key) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == key) return argv[i + 1];
  }
  throw std::runtime_error("missing argument " + key);
}

void flush(std::ofstream& out, std::string& buffer) {
  out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  buffer.clear();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string offsets_path = value(argc, argv, "--offsets");
    const std::string indices_path = value(argc, argv, "--indices");
    const std::string weights_path = value(argc, argv, "--weights");
    const std::string output_path = value(argc, argv, "--output");
    const std::uint64_t nodes = std::stoull(value(argc, argv, "--nodes"));
    const std::uint64_t entries = std::stoull(value(argc, argv, "--entries"));
    const bool directed = std::stoi(value(argc, argv, "--directed")) != 0;

    MappedFile<std::int32_t> offsets(offsets_path, nodes + 1);
    MappedFile<std::int32_t> indices(indices_path, entries);
    MappedFile<double> weights(weights_path, entries);
    if (static_cast<std::uint64_t>(offsets.data()[nodes]) != entries) {
      throw std::runtime_error("CSR terminal offset does not equal num_entries");
    }
    const std::uint64_t output_entries = directed ? entries : entries / 2;
    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create " + output_path);
    // Keep the representation accepted by the maintained and legacy Gunrock
    // MatrixMarket readers.  The benchmark weights are integer-valued doubles,
    // so writing them in a `real` matrix preserves the values while matching
    // the format used by the validated small/medium-graph runner.
    out << "%%MatrixMarket matrix coordinate real "
        << (directed ? "general\n" : "symmetric\n");
    out << nodes << ' ' << nodes << ' ' << output_entries << '\n';

    std::string buffer;
    buffer.reserve(16 * 1024 * 1024);
    std::uint64_t written = 0;
    for (std::uint64_t source = 0; source < nodes; ++source) {
      const std::uint64_t begin = static_cast<std::uint32_t>(offsets.data()[source]);
      const std::uint64_t end = static_cast<std::uint32_t>(offsets.data()[source + 1]);
      for (std::uint64_t edge = begin; edge < end; ++edge) {
        const std::uint64_t target = static_cast<std::uint32_t>(indices.data()[edge]);
        if (!directed && source >= target) continue;
        const auto weight = static_cast<std::uint64_t>(std::llround(weights.data()[edge]));
        buffer.append(std::to_string(source + 1));
        buffer.push_back(' ');
        buffer.append(std::to_string(target + 1));
        buffer.push_back(' ');
        buffer.append(std::to_string(weight));
        buffer.push_back('\n');
        ++written;
        if (buffer.size() >= 16 * 1024 * 1024) flush(out, buffer);
      }
    }
    flush(out, buffer);
    out.close();
    if (written != output_entries) {
      throw std::runtime_error(
          "written entry count mismatch: " + std::to_string(written) + " != " +
          std::to_string(output_entries));
    }
    std::cout << "nodes=" << nodes << " entries=" << written
              << " output=" << output_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "csr_to_matrix_market: " << error.what() << '\n';
    return 2;
  }
}
