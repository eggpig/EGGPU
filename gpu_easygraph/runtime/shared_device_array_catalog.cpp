#include "shared_device_array_catalog.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {
namespace {

const std::size_t kDefaultBudgetBytes =
    static_cast<std::size_t>(512) * 1024U * 1024U;

bool canonical_sha256(const std::string& value) {
    return value.size() == 64U &&
        value.find_first_not_of("0123456789abcdef") == std::string::npos;
}

std::size_t configured_budget() {
    const char* value = std::getenv("EGMGPU_SHARED_DEVICE_CACHE_BYTES");
    if (!value || !*value) return kDefaultBudgetBytes;
    const std::string text(value);
    if (text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(
            "EGMGPU_SHARED_DEVICE_CACHE_BYTES must be an unsigned byte count");
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno == ERANGE || !end || *end != '\0' || parsed >
            static_cast<unsigned long long>(
                std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(
            "EGMGPU_SHARED_DEVICE_CACHE_BYTES must be an unsigned byte count");
    }
    return static_cast<std::size_t>(parsed);
}

void hash_combine(std::size_t* seed, std::size_t value) {
    *seed ^= value + static_cast<std::size_t>(0x9e3779b9U) +
        (*seed << 6U) + (*seed >> 2U);
}

}  // namespace

SharedArrayKey::SharedArrayKey()
    : device_id(-1), kind(SharedArrayKind::Offsets), bytes(0U) {}

bool SharedArrayKey::operator==(const SharedArrayKey& other) const {
    return device_id == other.device_id && kind == other.kind &&
        sha256 == other.sha256 && compatibility == other.compatibility &&
        bytes == other.bytes;
}

std::size_t SharedArrayKeyHash::operator()(const SharedArrayKey& key) const {
    std::size_t seed = std::hash<int>()(key.device_id);
    hash_combine(&seed, std::hash<int>()(static_cast<int>(key.kind)));
    hash_combine(&seed, std::hash<std::string>()(key.sha256));
    hash_combine(&seed, std::hash<std::string>()(key.compatibility));
    hash_combine(&seed, std::hash<std::size_t>()(key.bytes));
    return seed;
}

SharedDeviceArray::SharedDeviceArray(
    const SharedArrayKey& key,
    void* data,
    const std::shared_ptr<void>& owner)
    : key_(key), data_(data), owner_(owner) {
    if (key_.bytes == 0U || !data_ || !owner_) {
        throw std::invalid_argument(
            "a shared device array requires nonempty owned storage");
    }
}

const SharedArrayKey& SharedDeviceArray::key() const { return key_; }
void* SharedDeviceArray::data() const { return data_; }
std::size_t SharedDeviceArray::bytes() const { return key_.bytes; }

SharedArrayAcquireResult::SharedArrayAcquireResult()
    : cache_hit(false), retained(false), new_bytes(0U), retained_bytes(0U) {}

SharedArrayCallStats::SharedArrayCallStats()
    : hits(0U), new_bytes(0U), retained_bytes(0U), all_retained(true) {}

void SharedArrayCallStats::add(const SharedArrayAcquireResult& result) {
    if (result.cache_hit) ++hits;
    new_bytes += result.new_bytes;
    retained_bytes = result.retained_bytes;
    all_retained = all_retained && result.retained;
}

SharedDeviceArrayCatalog::SharedDeviceArrayCatalog(std::size_t budget_bytes)
    : budget_bytes_(budget_bytes),
      retained_bytes_(0U),
      clock_(0U),
      hits_(0U),
      misses_(0U),
      evictions_(0U),
      allocated_bytes_(0U) {}

void SharedDeviceArrayCatalog::validate_key(const SharedArrayKey& key) const {
    if (key.device_id < 0 || key.bytes == 0U ||
        !canonical_sha256(key.sha256) || key.compatibility.empty()) {
        throw std::invalid_argument(
            "shared device-array key requires device, canonical SHA-256, "
            "nonzero bytes, and a compatibility scope");
    }
}

bool SharedDeviceArrayCatalog::evict_one_idle() {
    Map::iterator victim = entries_.end();
    for (Map::iterator item = entries_.begin(); item != entries_.end(); ++item) {
        if (item->second.array.use_count() != 1U) continue;
        if (victim == entries_.end() ||
            item->second.last_use < victim->second.last_use) {
            victim = item;
        }
    }
    if (victim == entries_.end()) return false;
    retained_bytes_ -= victim->second.array->bytes();
    entries_.erase(victim);
    ++evictions_;
    return true;
}

void SharedDeviceArrayCatalog::trim_idle_to_budget() {
    while (retained_bytes_ > budget_bytes_ && evict_one_idle()) {}
}

SharedArrayAcquireResult SharedDeviceArrayCatalog::acquire(
    const SharedArrayKey& key, const Factory& factory) {
    validate_key(key);
    if (!factory) {
        throw std::invalid_argument("shared device-array factory is empty");
    }
    Map::iterator found = entries_.find(key);
    if (found != entries_.end()) {
        found->second.last_use = ++clock_;
        ++hits_;
        SharedArrayAcquireResult result;
        result.array = found->second.array;
        result.cache_hit = true;
        result.retained = true;
        result.retained_bytes = retained_bytes_;
        return result;
    }

    std::shared_ptr<SharedDeviceArray> built = factory(key);
    if (!built || !(built->key() == key) || built->bytes() != key.bytes ||
        !built->data()) {
        throw std::invalid_argument(
            "shared device-array factory returned mismatched storage");
    }
    ++misses_;
    allocated_bytes_ += key.bytes;
    ++clock_;

    SharedArrayAcquireResult result;
    result.array = built;
    result.new_bytes = key.bytes;
    if (key.bytes <= budget_bytes_) {
        while (retained_bytes_ > budget_bytes_ - key.bytes &&
               evict_one_idle()) {}
        if (retained_bytes_ <= budget_bytes_ - key.bytes) {
            Entry entry;
            entry.array = built;
            entry.last_use = clock_;
            entries_.insert(std::make_pair(key, entry));
            retained_bytes_ += key.bytes;
            result.retained = true;
        }
    }
    result.retained_bytes = retained_bytes_;
    return result;
}

SharedArrayAcquireResult SharedDeviceArrayCatalog::find_existing(const SharedArrayKey& key) {
    validate_key(key);
    SharedArrayAcquireResult result;
    auto found=entries_.find(key);
    if(found==entries_.end())return result;
    found->second.last_use=++clock_;++hits_;
    result.array=found->second.array;result.cache_hit=true;result.retained=true;
    result.retained_bytes=retained_bytes_;return result;
}

SharedArrayAcquireResult SharedDeviceArrayCatalog::reuse_idle(
    const SharedArrayKey& previous, const SharedArrayKey& next,
    const std::function<void(void*)>& patch) {
    validate_key(next);
    SharedArrayAcquireResult result;
    if (previous.device_id != next.device_id || previous.bytes != next.bytes ||
        previous.kind != next.kind || previous.compatibility != next.compatibility)
        throw std::invalid_argument("dynamic patch changes allocation geometry");
    if (entries_.find(next) != entries_.end()) return result;
    auto found = entries_.find(previous);
    if (found == entries_.end() || found->second.array.use_count() != 1U)
        return result;
    auto old = found->second.array;
    retained_bytes_ -= old->bytes();
    entries_.erase(found);
    // No view, workspace or untracked shared_ptr holder exists at this point.
    // Patch fences device completion before publishing the replacement key.
    patch(old->data());
    std::shared_ptr<void> owner = old->owner_;
    auto replacement = std::make_shared<SharedDeviceArray>(next,old->data(),owner);
    Entry entry; entry.array = replacement; entry.last_use = ++clock_;
    entries_.insert(std::make_pair(next,entry));retained_bytes_ += next.bytes;
    result.array = replacement;result.retained=true;
    result.retained_bytes=retained_bytes_;return result;
}

void SharedDeviceArrayCatalog::set_budget_bytes(std::size_t budget_bytes) {
    budget_bytes_ = budget_bytes;
    trim_idle_to_budget();
}

SharedArrayCatalogStats SharedDeviceArrayCatalog::stats() const {
    SharedArrayCatalogStats result;
    result.budget_bytes = budget_bytes_;
    result.retained_bytes = retained_bytes_;
    result.retained_entries = entries_.size();
    result.active_entries = 0U;
    for (Map::const_iterator item = entries_.begin(); item != entries_.end();
         ++item) {
        if (item->second.array.use_count() > 1U) ++result.active_entries;
    }
    result.hits = hits_;
    result.misses = misses_;
    result.evictions = evictions_;
    result.allocated_bytes = allocated_bytes_;
    return result;
}

std::size_t SharedDeviceArrayCatalog::clear() {
    const std::size_t detached = retained_bytes_;
    entries_.clear();
    retained_bytes_ = 0U;
    clock_ = 0U;
    hits_ = 0U;
    misses_ = 0U;
    evictions_ = 0U;
    allocated_bytes_ = 0U;
    return detached;
}

const char* shared_array_kind_name(SharedArrayKind kind) {
    switch (kind) {
        case SharedArrayKind::Offsets: return "offsets";
        case SharedArrayKind::Indices: return "indices";
        case SharedArrayKind::LocalToGlobal: return "local_to_global";
        case SharedArrayKind::Owners: return "owners";
        case SharedArrayKind::Weights: return "weights";
    }
    return "unknown";
}

SharedDeviceArrayCatalog& shared_device_array_catalog() {
    static thread_local SharedDeviceArrayCatalog catalog(configured_budget());
    return catalog;
}

std::size_t partition_state_cache_capacity() {
    const char* value = std::getenv("EGMGPU_PARTITION_STATE_CACHE_ENTRIES");
    if (!value || !*value) return 8U;
    const std::string text(value);
    if (text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(
            "EGMGPU_PARTITION_STATE_CACHE_ENTRIES must be in [1, 1024]");
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno == ERANGE || !end || *end != '\0' || parsed == 0U ||
        parsed > 1024U) {
        throw std::invalid_argument(
            "EGMGPU_PARTITION_STATE_CACHE_ENTRIES must be in [1, 1024]");
    }
    return static_cast<std::size_t>(parsed);
}


void SharedDeviceArrayCatalog::append_managed_inventory(ManagedStateInventory& inventory) const {
    ManagedOwnerRecord owner;
    owner.owner_id = "catalog:" + std::to_string(inventory.device_id);
    owner.kind = "catalog"; owner.algorithm = "shared";
    owner.device_id = inventory.device_id;
    inventory.owner(owner);
    for (const auto& entry : entries_) {
        const auto& array = entry.second.array;
        if (array->key().device_id != inventory.device_id)
            throw std::runtime_error("Cross-device catalog allocation");
        inventory.allocation(array->data(), array->bytes(), owner.owner_id,
            shared_array_kind_name(array->key().kind), false, true, array.use_count() - 1U);
    }
}

void SharedDeviceArrayCatalog::release_managed_idle(ManagedEvictRecord& receipt) {
    const auto bytes_before = retained_bytes_;
    const auto entries_before = entries_.size();
    while (evict_one_idle()) {}
    receipt.idle_catalog_released_bytes += bytes_before - retained_bytes_;
    receipt.idle_catalog_released_entries += entries_before - entries_.size();
}

}  // namespace runtime
}  // namespace gpu_easygraph
