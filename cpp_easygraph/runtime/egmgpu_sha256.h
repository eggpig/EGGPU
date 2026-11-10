#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace easygraph {
namespace egmgpu {

class Sha256 {
public:
    Sha256();
    void update(const unsigned char* data, std::size_t size);
    std::string final_hex();

private:
    void transform(const unsigned char block[64]);

    std::uint32_t state_[8];
    std::uint64_t byte_count_;
    unsigned char buffer_[64];
    std::size_t buffer_size_;
    bool finalized_;
};

std::string sha256_file(const std::string& path);

}  // namespace egmgpu
}  // namespace easygraph
