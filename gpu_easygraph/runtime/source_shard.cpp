#include "source_shard.h"

#include <algorithm>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {

std::vector<ContiguousShard> balanced_contiguous_shards(
    std::size_t item_count,
    std::size_t max_shards) {
    if (max_shards == 0) {
        throw std::invalid_argument("source sharding requires at least one device");
    }
    if (item_count == 0) {
        return std::vector<ContiguousShard>();
    }

    const std::size_t shard_count = std::min(item_count, max_shards);
    const std::size_t quotient = item_count / shard_count;
    const std::size_t remainder = item_count % shard_count;
    std::vector<ContiguousShard> shards;
    shards.reserve(shard_count);
    std::size_t begin = 0;
    for (std::size_t ordinal = 0; ordinal < shard_count; ++ordinal) {
        const std::size_t size = quotient + (ordinal < remainder ? 1U : 0U);
        ContiguousShard shard;
        shard.begin = begin;
        shard.end = begin + size;
        shards.push_back(shard);
        begin = shard.end;
    }
    return shards;
}

}  // namespace runtime
}  // namespace gpu_easygraph
