#include "egmgpu_sha256.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace easygraph {
namespace egmgpu {
namespace {

const std::uint32_t kRoundConstants[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

std::uint32_t rotate_right(std::uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32U - shift));
}

}  // namespace

Sha256::Sha256()
    : byte_count_(0), buffer_size_(0), finalized_(false) {
    state_[0] = 0x6a09e667U;
    state_[1] = 0xbb67ae85U;
    state_[2] = 0x3c6ef372U;
    state_[3] = 0xa54ff53aU;
    state_[4] = 0x510e527fU;
    state_[5] = 0x9b05688cU;
    state_[6] = 0x1f83d9abU;
    state_[7] = 0x5be0cd19U;
    std::memset(buffer_, 0, sizeof(buffer_));
}

void Sha256::update(const unsigned char* data, std::size_t size) {
    if (finalized_) {
        throw std::logic_error("SHA-256 cannot be updated after finalization");
    }
    if (size != 0 && data == NULL) {
        throw std::invalid_argument("SHA-256 input pointer is null");
    }
    if (static_cast<std::uint64_t>(size) >
        (std::numeric_limits<std::uint64_t>::max)() - byte_count_) {
        throw std::overflow_error("SHA-256 byte count overflow");
    }
    byte_count_ += static_cast<std::uint64_t>(size);
    while (size != 0) {
        const std::size_t copied =
            (std::min)(size, sizeof(buffer_) - buffer_size_);
        std::memcpy(buffer_ + buffer_size_, data, copied);
        buffer_size_ += copied;
        data += copied;
        size -= copied;
        if (buffer_size_ == sizeof(buffer_)) {
            transform(buffer_);
            buffer_size_ = 0;
        }
    }
}

void Sha256::transform(const unsigned char block[64]) {
    std::uint32_t words[64];
    for (std::size_t index = 0; index < 16; ++index) {
        words[index] =
            (static_cast<std::uint32_t>(block[index * 4]) << 24) |
            (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16) |
            (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8) |
            static_cast<std::uint32_t>(block[index * 4 + 3]);
    }
    for (std::size_t index = 16; index < 64; ++index) {
        const std::uint32_t s0 = rotate_right(words[index - 15], 7) ^
                                 rotate_right(words[index - 15], 18) ^
                                 (words[index - 15] >> 3);
        const std::uint32_t s1 = rotate_right(words[index - 2], 17) ^
                                 rotate_right(words[index - 2], 19) ^
                                 (words[index - 2] >> 10);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < 64; ++index) {
        const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                                   rotate_right(e, 25);
        const std::uint32_t choice = (e & f) ^ ((~e) & g);
        const std::uint32_t temporary1 =
            h + sum1 + choice + kRoundConstants[index] + words[index];
        const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                                   rotate_right(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

std::string Sha256::final_hex() {
    if (finalized_) {
        throw std::logic_error("SHA-256 was already finalized");
    }
    const std::uint64_t bit_count = byte_count_ * 8U;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56) {
        std::memset(buffer_ + buffer_size_, 0, sizeof(buffer_) - buffer_size_);
        transform(buffer_);
        buffer_size_ = 0;
    }
    std::memset(buffer_ + buffer_size_, 0, 56 - buffer_size_);
    for (std::size_t index = 0; index < 8; ++index) {
        buffer_[63 - index] =
            static_cast<unsigned char>((bit_count >> (index * 8)) & 0xffU);
    }
    transform(buffer_);
    finalized_ = true;

    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < 8; ++index) {
        result << std::setw(8) << state_[index];
    }
    return result.str();
}

std::string sha256_file(const std::string& path) {
    std::ifstream stream(path.c_str(), std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open file for SHA-256: " + path);
    }
    Sha256 digest;
    std::vector<unsigned char> buffer(1U << 20);
    while (stream) {
        stream.read(reinterpret_cast<char*>(&buffer[0]),
                    static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count > 0) {
            digest.update(&buffer[0], static_cast<std::size_t>(count));
        }
    }
    if (!stream.eof()) {
        throw std::runtime_error("failed while hashing file: " + path);
    }
    return digest.final_hex();
}

}  // namespace egmgpu
}  // namespace easygraph
