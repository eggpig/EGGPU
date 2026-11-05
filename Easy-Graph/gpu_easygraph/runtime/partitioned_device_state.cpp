#include "partitioned_device_state.h"
#include "device_allocation_observer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <map>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpu_easygraph {
namespace runtime {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorName(status) << " ("
            << cudaGetErrorString(status) << ")";
    throw std::runtime_error(message.str());
}

using ProfileClock = std::chrono::steady_clock;

struct UploadProfileEvents {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    UploadProfileEvents() {
        check_cuda(cudaEventCreate(&begin), "create upload profile begin event");
        const cudaError_t status = cudaEventCreate(&end);
        if (status != cudaSuccess) {
            cudaEventDestroy(begin);
            begin = nullptr;
            check_cuda(status, "create upload profile end event");
        }
    }
    ~UploadProfileEvents() {
        if (end) cudaEventDestroy(end);
        if (begin) cudaEventDestroy(begin);
    }
};

thread_local DeviceStateProfile state_profile;
thread_local std::unique_ptr<UploadProfileEvents> upload_profile_events;

double profile_elapsed(const ProfileClock::time_point& begin) {
    return std::chrono::duration<double>(ProfileClock::now() - begin).count();
}

void copy_host_to_device(
    void* destination, const void* source, std::size_t bytes,
    const char* operation) {
    if (!state_profile.enabled) {
        check_cuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice), operation);
        return;
    }
    if (!upload_profile_events) upload_profile_events.reset(new UploadProfileEvents());
    UploadProfileEvents& events = *upload_profile_events;
    check_cuda(cudaEventRecord(events.begin, nullptr), "record upload profile begin");
    const ProfileClock::time_point begin = ProfileClock::now();
    const cudaError_t status = cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);
    const double wall = profile_elapsed(begin);
    check_cuda(status, operation);
    check_cuda(cudaEventRecord(events.end, nullptr), "record upload profile end");
    check_cuda(cudaEventSynchronize(events.end), "complete upload profile event");
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, events.begin, events.end),
               "read upload profile elapsed time");
    state_profile.h2d_wall_seconds += wall;
    state_profile.h2d_device_seconds += static_cast<double>(milliseconds) / 1000.0;
    state_profile.h2d_bytes += static_cast<std::uint64_t>(bytes);
    ++state_profile.h2d_count;
}

void check_current_device(int expected, const char* operation) {
    int current = -1;
    check_cuda(cudaGetDevice(&current), operation);
    if (current != expected) {
        throw std::runtime_error(
            "partition device state must be accessed by its fixed worker");
    }
}

template <typename T>
std::shared_ptr<DeviceAllocation> copy_to_device(
    int device_id, const std::vector<T>& source, const char* operation) {
    if (source.empty()) return std::shared_ptr<DeviceAllocation>();
    const std::size_t bytes = source.size() * sizeof(T);
    std::shared_ptr<DeviceAllocation> allocation(
        new DeviceAllocation(device_id, bytes));
    copy_host_to_device(allocation->data(), source.data(), bytes, operation);
    return allocation;
}

std::size_t allocation_bytes(const std::shared_ptr<DeviceAllocation>& allocation) {
    return allocation ? allocation->bytes() : 0U;
}

std::size_t allocation_bytes(const std::shared_ptr<SharedDeviceArray>& allocation) {
    return allocation ? allocation->bytes() : 0U;
}

}  // namespace

void begin_device_state_profile(bool enabled) {
    upload_profile_events.reset();
    state_profile = DeviceStateProfile();
    state_profile.enabled = enabled;
}

DeviceStateProfile end_device_state_profile() {
    const DeviceStateProfile result = state_profile;
    state_profile = DeviceStateProfile();
    upload_profile_events.reset();
    return result;
}

PartitionDeviceUpload::PartitionDeviceUpload()
    : partition_id(-1),
      device_id(-1),
      graph_identity(0),
      generation(0),
      owned_vertex_count(0),
      device_metadata_policy("full"),
      local_vertex_count(0) {}

DeviceAllocation::DeviceAllocation(int device_id, std::size_t bytes)
    : device_id_(device_id), pointer_(nullptr), bytes_(bytes) {
    if (device_id_ < 0 || bytes_ == 0) {
        throw std::invalid_argument(
            "device allocation requires a device and nonzero size");
    }
    check_current_device(device_id_, "cudaGetDevice before partition allocation");
    if (state_profile.enabled) {
        const ProfileClock::time_point begin = ProfileClock::now();
        const cudaError_t status = cudaMalloc(&pointer_, bytes_);
        const double wall = profile_elapsed(begin);
        check_cuda(status, "cudaMalloc partition state");
        state_profile.malloc_wall_seconds += wall;
        state_profile.malloc_bytes += static_cast<std::uint64_t>(bytes_);
        ++state_profile.malloc_count;
    } else {
        check_cuda(cudaMalloc(&pointer_, bytes_), "cudaMalloc partition state");
    }
    try {
        allocation_generation_ = observe_device_allocation(device_id_, pointer_, bytes_);
    } catch (...) {
        cudaFree(pointer_);
        pointer_ = nullptr;
        throw;
    }
}

DeviceAllocation::~DeviceAllocation() {
    if (!pointer_) return;
    int previous = -1;
    const cudaError_t get_status = cudaGetDevice(&previous);
    if (get_status == cudaSuccess && previous != device_id_) {
        if (cudaSetDevice(device_id_) != cudaSuccess) { observe_device_free(device_id_, pointer_, false); return; }
    }
    const cudaError_t free_status = cudaFree(pointer_);
    observe_device_free(device_id_, pointer_, free_status == cudaSuccess);
    if (get_status == cudaSuccess && previous != device_id_) {
        cudaSetDevice(previous);
    }
}

void* DeviceAllocation::data() const { return pointer_; }
std::size_t DeviceAllocation::bytes() const { return bytes_; }
int DeviceAllocation::device_id() const { return device_id_; }

extern "C" cudaError_t eggpu_dynamic_scatter_f64(double*,const int*,const double*,std::size_t);
namespace {
struct DynamicMirror {
    SharedArrayKey key;
    std::vector<double> values;
    std::weak_ptr<SharedDeviceArray> device;
};
thread_local std::map<std::string,DynamicMirror> dynamic_mirrors;
thread_local std::map<std::string,std::uint64_t> dynamic_counters;
std::string dynamic_route() {
    const char* value=std::getenv("EGGPU_DYNAMIC_WEIGHT_ROUTE");
    const std::string route=value ? value : "persistent_full";
    if(route!="persistent_full" && route!="lazy_sparse_cow" && route!="lease_sparse")
        throw std::invalid_argument("invalid dynamic weight route");
    return route;
}
void sparse_patch(void* target,int device,const std::vector<int>& positions,
                  const std::vector<double>& values) {
    if(positions.empty())return;
    DeviceAllocation dp(device,positions.size()*sizeof(int));
    DeviceAllocation dv(device,values.size()*sizeof(double));
    copy_host_to_device(dp.data(),positions.data(),dp.bytes(),"upload dynamic positions");
    copy_host_to_device(dv.data(),values.data(),dv.bytes(),"upload dynamic values");
    check_cuda(eggpu_dynamic_scatter_f64(static_cast<double*>(target),
        static_cast<const int*>(dp.data()),static_cast<const double*>(dv.data()),values.size()),
        "launch dynamic sparse patch");
    check_cuda(cudaStreamSynchronize(nullptr),"complete dynamic sparse patch");
    dynamic_counters["patch_h2d_bytes"]+=dp.bytes()+dv.bytes();
    dynamic_counters["patch_logical_write_bytes"]+=values.size()*sizeof(double);
    dynamic_counters["patch_packets"]++;
    dynamic_counters["patch_positions"]+=positions.size();
    dynamic_counters["peak_packet_device_bytes"]=std::max(dynamic_counters["peak_packet_device_bytes"],
        std::uint64_t(dp.bytes()+dv.bytes()));
}
}
std::map<std::string,std::uint64_t> dynamic_weight_counters() {
    auto result=dynamic_counters;
    std::uint64_t host=0;for(const auto& item:dynamic_mirrors)host+=item.second.values.size()*sizeof(double);
    result["host_mirror_live_bytes"]=host;
    result["abi_version"]=1;
    return result;
}
std::shared_ptr<SharedDeviceArray> upload_verified_shared_device_array(
    int device_id, SharedArrayKind kind, const std::string& verified_sha256,
    const std::string& compatibility, const void* source, std::size_t bytes,
    const char* operation, SharedArrayAcquireResult* acquisition,
    const ExactWeightPatch* native_patch) {
    if(bytes==0U) {
        if(source)throw std::invalid_argument("zero-byte array has host storage");
        if(acquisition)*acquisition=SharedArrayAcquireResult();
        return std::shared_ptr<SharedDeviceArray>();
    }
    if(!source)throw std::invalid_argument("null shared array host storage");
    SharedArrayKey key;key.device_id=device_id;key.kind=kind;
    key.sha256=verified_sha256;key.compatibility=compatibility;key.bytes=bytes;
    const std::string route=dynamic_route();
    const bool weighted=kind==SharedArrayKind::Weights;
    // Only actual f64 layouts are eligible. Compact U8 remains a full upload.
    const bool eligible=weighted && bytes%sizeof(double)==0 &&
        compatibility.find("u8")==std::string::npos;
    const std::string slot=compatibility+"|bytes="+std::to_string(bytes);
    auto mirror=dynamic_mirrors.find(slot);
    bool used_native_packet=false;
    auto update_mirror=[&](const SharedArrayAcquireResult& acquired) {
        if(!eligible || route=="persistent_full")return;
        if(dynamic_mirrors.size()>=64 && mirror==dynamic_mirrors.end())dynamic_mirrors.erase(dynamic_mirrors.begin());
        DynamicMirror& updated=dynamic_mirrors[slot];
        if(!(updated.key==key) || updated.values.size()!=bytes/sizeof(double)) {
            if(used_native_packet) {
                for(std::size_t i=0;i<native_patch->positions.size();++i)
                    updated.values[native_patch->positions[i]]=native_patch->values[i];
                dynamic_counters["host_mirror_packet_write_bytes"]+=native_patch->values.size()*sizeof(double);
            } else {
                const double* begin=static_cast<const double*>(source);
                updated.values.assign(begin,begin+bytes/sizeof(double));
                dynamic_counters["host_mirror_copy_bytes"]+=bytes;
            }
        } else dynamic_counters["host_mirror_copy_skips"]++;
        updated.key=key;updated.device=acquired.array;
    };
    // A content-addressed hit already owns the exact requested value bytes.
    // Do not construct and then discard an unrelated historical dirty packet.
    SharedArrayAcquireResult existing=shared_device_array_catalog().find_existing(key);
    if(existing.array) {
        if(weighted) {
            dynamic_counters["weight_cache_hits"]++;
            dynamic_counters["weight_cache_hit_before_diff"]++;
        }
        update_mirror(existing);
        if(acquisition)*acquisition=existing;
        return existing.array;
    }
    const char* dense_setting=std::getenv("EGGPU_DENSE_WEIGHT_LEASE");
    const std::string dense_policy=dense_setting ? dense_setting : "0";
    if (dense_policy!="0" && dense_policy!="1")
        throw std::invalid_argument("EGGPU_DENSE_WEIGHT_LEASE must be 0 or 1");
    const bool allow_dense_lease=dense_policy=="1";
    bool has_previous=false;
    bool sparse=false;
    std::vector<int> positions;std::vector<double> values;
    SharedArrayKey previous;
    if(eligible && route!="persistent_full" && mirror!=dynamic_mirrors.end() &&
        !(mirror->second.key==key) && !mirror->second.device.expired()) {
        const auto start=ProfileClock::now();
        const double* incoming=static_cast<const double*>(source);
        const std::size_t n=bytes/sizeof(double);
        if(n>std::size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("dynamic f64 array exceeds int32 positions");
        if(native_patch) {
            ++dynamic_counters["native_packet_candidates"];
            if(native_patch->target_sha256!=key.sha256 || native_patch->positions.size()!=native_patch->values.size())
                throw std::invalid_argument("native packet target identity or shape mismatch");
            if(native_patch->base_sha256==mirror->second.key.sha256 && mirror->second.key.device_id==device_id &&
               mirror->second.values.size()==n) {
                // The producer is native immutable-base construction. This
                // consumer still checks its target values and exact base key.
                for(std::size_t i=0;i<native_patch->positions.size();++i) {
                    const int position=native_patch->positions[i];
                    if(position<0 || static_cast<std::size_t>(position)>=n ||
                       std::memcmp(&incoming[position],&native_patch->values[i],sizeof(double))!=0)
                        throw std::invalid_argument("native packet position/value mismatch");
                }
                positions=native_patch->positions;values=native_patch->values;
                used_native_packet=true;++dynamic_counters["native_packet_exact_base_accepted"];
                dynamic_counters["native_packet_validated_positions"]+=positions.size();
            } else ++dynamic_counters["native_packet_base_mismatch_rejected"];
        }
        if(!used_native_packet) {
            for(std::size_t i=0;i<n;++i)
                if(std::memcmp(&incoming[i],&mirror->second.values[i],sizeof(double))!=0) {
                    positions.push_back(static_cast<int>(i));values.push_back(incoming[i]);
                }
            dynamic_counters["host_diff_bytes"]+=bytes;
            dynamic_counters["host_diff_nanoseconds"]+=std::uint64_t(profile_elapsed(start)*1e9);
        } else dynamic_counters["native_packet_validation_nanoseconds"]+=std::uint64_t(profile_elapsed(start)*1e9);
        sparse=positions.size()<=n/4;
        previous=mirror->second.key;
        has_previous=true;
    }
    SharedArrayAcquireResult result;
    // Exclusivity permits overwriting storage independently of density.
    // Density selects sparse scatter versus full H2D; a live reader denies both.
    if(has_previous && route=="lease_sparse" && (sparse || allow_dense_lease)) {
        if(!sparse)++dynamic_counters["dense_lease_candidates"];
        result=shared_device_array_catalog().reuse_idle(previous,key,
          [&](void* target) {
            check_cuda(cudaDeviceSynchronize(),"complete readers before dynamic lease");
            if(sparse) sparse_patch(target,device_id,positions,values);
            else {
                copy_host_to_device(target,source,bytes,"overwrite exclusive dense weights");
                dynamic_counters["weight_full_h2d_bytes"]+=bytes;
                ++dynamic_counters["weight_full_uploads"];
                dynamic_counters["dense_lease_h2d_bytes"]+=bytes;
            }
          });
        if(result.array) {
            dynamic_counters["lease_reuses"]++;
            dynamic_counters["lease_reused_bytes"]+=bytes;
            if(!sparse) {
                ++dynamic_counters["dense_lease_reuses"];
                dynamic_counters["dense_lease_reused_bytes"]+=bytes;
            }
        } else {
            dynamic_counters["lease_denied_nonexclusive_or_missing"]++;
            if(!sparse)++dynamic_counters["dense_lease_denied_nonexclusive_or_missing"];
        }
    }
    if(!result.array)result=shared_device_array_catalog().acquire(key,
      [&](const SharedArrayKey& requested) {
        std::shared_ptr<DeviceAllocation> allocation(new DeviceAllocation(device_id,bytes));
        auto old=sparse ? mirror->second.device.lock() : std::shared_ptr<SharedDeviceArray>();
        if(old && old->key()==previous) {
            check_cuda(cudaMemcpy(allocation->data(),old->data(),bytes,cudaMemcpyDeviceToDevice),"clone immutable dynamic weights");
            dynamic_counters["cow_d2d_bytes"]+=bytes;dynamic_counters["cow_copies"]++;
            sparse_patch(allocation->data(),device_id,positions,values);
        } else {
            copy_host_to_device(allocation->data(),source,bytes,operation);
            if(weighted) {dynamic_counters["weight_full_h2d_bytes"]+=bytes;dynamic_counters["weight_full_uploads"]++;}
        }
        std::shared_ptr<void> owner=allocation;
        return std::make_shared<SharedDeviceArray>(requested,allocation->data(),owner);
      });
    if(weighted && result.cache_hit)dynamic_counters["weight_cache_hits"]++;
    update_mirror(result);
    if(acquisition)*acquisition=result;
    return result.array;
}

namespace {
// Full keeps its previous state identity. The projected view receives its
// own view/workspace key even when the caller supplies the same graph view.
std::string page_rank_projection_signature(const PartitionDeviceUpload& upload) {
    if (upload.device_metadata_policy == "full") return upload.view_signature;
    if (upload.device_metadata_policy == "pull_only")
        return upload.view_signature + ":pr-device-metadata=pull_only-v13";
    throw std::invalid_argument("PageRank device metadata policy must be full or pull_only");
}
}  // namespace

ResidentPartitionView::ResidentPartitionView(const PartitionDeviceUpload& upload)
    : managed_owner_id_(new_managed_owner_id("pr", "view", upload.device_id)),
      state_key_(
          upload.graph_identity,
      upload.generation,
          stable_signature64(page_rank_projection_signature(upload)),
          DeviceGroup(upload.device_group),
          GraphLayout::Partitioned),
      orientation_(upload.orientation),
      graph_identity_text_(upload.graph_identity_text),
      view_signature_text_(page_rank_projection_signature(upload)),
      device_metadata_policy_(upload.device_metadata_policy),
      resident_bytes_(0),
      shared_array_count_(0U) {
    std::memset(&view_, 0, sizeof(view_));
}

std::shared_ptr<ResidentPartitionView> ResidentPartitionView::upload(
    const PartitionDeviceUpload& upload) {
    std::shared_ptr<ResidentPartitionView> state(
        new ResidentPartitionView(upload));
    state->initialize(upload);
    // Pageable H2D cudaMemcpy can return after host staging but before DMA.
    // Publish only a complete immutable view to nonblocking worker streams.
    check_cuda(cudaStreamSynchronize(nullptr), "complete PageRank view upload");
    return state;
}

ResidentPartitionView::~ResidentPartitionView() {}

void ResidentPartitionView::initialize(const PartitionDeviceUpload& upload) {
    const bool full_metadata = upload.device_metadata_policy == "full";
    // Existing full upload callers derive this extent from the complete map.
    const std::size_t local_count = full_metadata
        ? upload.local_to_global.size() : upload.local_vertex_count;
    if (upload.partition_id < 0 || upload.device_id < 0 ||
        upload.graph_identity_text.empty() || upload.view_signature.empty() ||
        upload.device_group.empty() ||
        std::find(upload.device_group.begin(), upload.device_group.end(),
                  upload.device_id) == upload.device_group.end() ||
        (upload.orientation != "incoming" &&
         upload.orientation != "undirected") ||
        local_count < upload.owned_vertex_count ||
        (full_metadata && upload.owners.size() != local_count) ||
        (!full_metadata && (!upload.local_to_global.empty() || !upload.owners.empty())) ||
        upload.offsets.size() != upload.owned_vertex_count + 1U ||
        (!upload.weights.empty() && upload.indices.size() != upload.weights.size()) ||
        upload.owned_out_weight.size() != upload.owned_vertex_count ||
        (upload.owned_vertex_count == 0 &&
         (local_count != 0U || !upload.local_to_global.empty() || !upload.owners.empty() ||
          !upload.indices.empty() || !upload.weights.empty()))) {
        throw std::invalid_argument("invalid partition device upload");
    }
    if (upload.offsets.front() != 0 || upload.offsets.back() < 0 ||
        static_cast<std::size_t>(upload.offsets.back()) != upload.indices.size()) {
        throw std::invalid_argument("invalid partition device CSR offsets");
    }
    for (std::size_t edge = 0; edge < upload.indices.size(); ++edge) {
        if (upload.indices[edge] < 0 ||
            static_cast<std::size_t>(upload.indices[edge]) >=
                local_count) {
            throw std::invalid_argument("partition device CSR index is out of range");
        }
    }
    check_current_device(upload.device_id, "cudaGetDevice before partition upload");
    if (!upload.topology_compatibility.empty()) {
        SharedArrayAcquireResult topology_array;
        shared_offsets_ = upload_verified_shared_device_array(
            upload.device_id, SharedArrayKind::Offsets, upload.offsets_sha256,
            upload.topology_compatibility, upload.offsets.data(),
            upload.offsets.size() * sizeof(int), "share PageRank CSR offsets", &topology_array);
        if (shared_offsets_) { shared_array_stats_.add(topology_array); ++shared_array_count_; }
        shared_indices_ = upload_verified_shared_device_array(
            upload.device_id, SharedArrayKind::Indices, upload.indices_sha256,
            upload.topology_compatibility, upload.indices.empty() ? nullptr : upload.indices.data(),
            upload.indices.size() * sizeof(int), "share PageRank CSR indices", &topology_array);
        if (shared_indices_) { shared_array_stats_.add(topology_array); ++shared_array_count_; }
    } else {
        offsets_ = copy_to_device(
            upload.device_id, upload.offsets, "copy partition offsets");
        indices_ = copy_to_device(
            upload.device_id, upload.indices, "copy partition indices");
    }
    SharedArrayAcquireResult acquired_weights;
    weights_ = upload_verified_shared_device_array(
        upload.device_id, SharedArrayKind::Weights, upload.weights_sha256,
        upload.weight_compatibility,
        upload.weights.empty() ? nullptr : upload.weights.data(),
        upload.weights.size() * sizeof(double), "share PageRank weights",
        &acquired_weights);
    if (weights_) { shared_array_stats_.add(acquired_weights); ++shared_array_count_; }
    if (full_metadata) {
        SharedArrayAcquireResult acquired;
        local_to_global_ = upload_verified_shared_device_array(
            upload.device_id, SharedArrayKind::LocalToGlobal,
            upload.local_to_global_sha256, upload.ownership_compatibility,
            upload.local_to_global.empty() ? nullptr : upload.local_to_global.data(),
            upload.local_to_global.size() * sizeof(std::uint64_t),
            "copy shared partition local-to-global map", &acquired);
        if (local_to_global_) {
            shared_array_stats_.add(acquired);
            ++shared_array_count_;
        }
        owners_ = upload_verified_shared_device_array(
            upload.device_id, SharedArrayKind::Owners, upload.owners_sha256,
            upload.ownership_compatibility,
            upload.owners.empty() ? nullptr : upload.owners.data(),
            upload.owners.size() * sizeof(int),
            "copy shared partition owners", &acquired);
        if (owners_) {
            shared_array_stats_.add(acquired);
            ++shared_array_count_;
        }
    }
    owned_out_weight_ = copy_to_device(
        upload.device_id, upload.owned_out_weight,
        "copy partition outgoing weights");
    pagerank_heavy_rows_ = copy_to_device(
        upload.device_id, upload.pagerank_heavy_rows, "copy PageRank heavy row list");
    halo_source_local_ = copy_to_device(
        upload.device_id, upload.halo_source_local, "copy PageRank halo sources");
    halo_destination_local_ = copy_to_device(
        upload.device_id, upload.halo_destination_local, "copy PageRank halo destinations");

    view_.partition_id = upload.partition_id;
    view_.device_id = upload.device_id;
    view_.owned_vertex_count = upload.owned_vertex_count;
    view_.local_vertex_count = local_count;
    view_.adjacency_count = upload.indices.size();
    view_.offsets = shared_offsets_ ? static_cast<int*>(shared_offsets_->data())
        : (offsets_ ? static_cast<int*>(offsets_->data()) : nullptr);
    view_.indices = shared_indices_ ? static_cast<int*>(shared_indices_->data())
        : (indices_ ? static_cast<int*>(indices_->data()) : nullptr);
    view_.weights = weights_ ? static_cast<double*>(weights_->data()) : nullptr;
    view_.local_to_global = local_to_global_
        ? static_cast<std::uint64_t*>(local_to_global_->data()) : nullptr;
    view_.owners = owners_ ? static_cast<int*>(owners_->data()) : nullptr;
    view_.owned_out_weight = owned_out_weight_
        ? static_cast<double*>(owned_out_weight_->data()) : nullptr;
    view_.pagerank_heavy_rows = pagerank_heavy_rows_
        ? static_cast<int*>(pagerank_heavy_rows_->data()) : nullptr;
    view_.pagerank_heavy_row_count = upload.pagerank_heavy_rows.size();
    view_.halo_source_local = halo_source_local_
        ? static_cast<int*>(halo_source_local_->data()) : nullptr;
    view_.halo_destination_local = halo_destination_local_
        ? static_cast<int*>(halo_destination_local_->data()) : nullptr;
    resident_bytes_ = allocation_bytes(offsets_) + allocation_bytes(indices_) +
        allocation_bytes(weights_) + allocation_bytes(local_to_global_) +
        allocation_bytes(owners_) + allocation_bytes(owned_out_weight_) +
        allocation_bytes(pagerank_heavy_rows_) +
        allocation_bytes(halo_source_local_) + allocation_bytes(halo_destination_local_) +
        allocation_bytes(shared_offsets_) + allocation_bytes(shared_indices_);
}

const StateKey& ResidentPartitionView::state_key() const { return state_key_; }
const std::string& ResidentPartitionView::orientation() const {
    return orientation_;
}
const std::string& ResidentPartitionView::graph_identity_text() const {
    return graph_identity_text_;
}
const std::string& ResidentPartitionView::view_signature_text() const {
    return view_signature_text_;
}
const std::string& ResidentPartitionView::device_metadata_policy() const {
    return device_metadata_policy_;
}
const PartitionDeviceView& ResidentPartitionView::view() const { return view_; }
std::size_t ResidentPartitionView::resident_bytes() const {
    return resident_bytes_;
}
const SharedArrayCallStats& ResidentPartitionView::shared_array_stats() const {
    return shared_array_stats_;
}
std::size_t ResidentPartitionView::shared_array_count() const {
    return shared_array_count_;
}
bool ResidentPartitionView::cacheable() const {
    return shared_array_stats_.all_retained;
}

PartitionWorkspace::PartitionWorkspace(int device_id)
    : managed_owner_id_(new_managed_owner_id("pr", "workspace", device_id)),
      device_id_(device_id) {
    if (device_id_ < 0) {
        throw std::invalid_argument("partition workspace requires a device");
    }
}

RetainedSparseHaloMaps ResidentPartitionView::ensure_sparse_halo_maps(
    const std::string& plan_key,const std::vector<int>& source,
    const std::vector<int>& packed,cudaStream_t stream) {
    check_current_device(view_.device_id,"retained sparse halo mapping worker");
    if (plan_key.empty() || source.size()!=packed.size() || source.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid retained sparse mapping identity/shape");
    if (!sparse_halo_plan_key_.empty() && sparse_halo_plan_key_!=plan_key)
        throw std::invalid_argument("retained sparse map belongs to another immutable plan");
    RetainedSparseHaloMaps result;
    if (sparse_halo_plan_key_==plan_key) {
        if (source.size()!=sparse_halo_count_) throw std::invalid_argument("retained sparse map count changed");
        result.cache_hit=true;
    } else {
        // Only an internal, fully validated immutable host plan reaches this
        // path. Validate first construction, then publish after both transfers.
        for (std::size_t i=0;i<source.size();++i)
            if (source[i]<0 || static_cast<std::size_t>(source[i])>=view_.owned_vertex_count || packed[i]!=static_cast<int>(i))
                throw std::invalid_argument("retained sparse map index mismatch");
        const std::size_t bytes=source.size()*sizeof(int);
        if (bytes>(std::numeric_limits<std::size_t>::max()-resident_bytes_)/2U)
            throw std::overflow_error("retained sparse map capacity overflow");
        std::string key_copy=plan_key;
        std::shared_ptr<DeviceAllocation> a,b;
        if (bytes) {
            a.reset(new DeviceAllocation(view_.device_id,bytes));b.reset(new DeviceAllocation(view_.device_id,bytes));
            check_cuda(cudaMemcpyAsync(a->data(),source.data(),bytes,cudaMemcpyHostToDevice,stream),"upload retained sparse source map");
            check_cuda(cudaMemcpyAsync(b->data(),packed.data(),bytes,cudaMemcpyHostToDevice,stream),"upload retained sparse packed map");
            check_cuda(cudaStreamSynchronize(stream),"publish retained sparse maps");
        }
        sparse_halo_source_.swap(a);sparse_halo_packed_.swap(b);sparse_halo_plan_key_.swap(key_copy);
        sparse_halo_count_=source.size();resident_bytes_+=2U*bytes;
        result.new_bytes=result.h2d_bytes=2U*bytes;
    }
    result.source=sparse_halo_source_ ? static_cast<const int*>(sparse_halo_source_->data()) : nullptr;
    result.packed=sparse_halo_packed_ ? static_cast<const int*>(sparse_halo_packed_->data()) : nullptr;
    result.bytes=allocation_bytes(sparse_halo_source_)+allocation_bytes(sparse_halo_packed_);
    return result;
}

std::size_t PartitionWorkspace::ensure_sparse_send_capacity(std::size_t count) {
    check_current_device(device_id_,"retained sparse send workspace worker");
    if (count>std::numeric_limits<std::size_t>::max()/sizeof(double)) throw std::overflow_error("sparse send capacity overflow");
    const std::size_t bytes=count*sizeof(double);
    if (bytes && (!sparse_send_ || sparse_send_->bytes()<bytes)) {
        std::shared_ptr<DeviceAllocation> next(new DeviceAllocation(device_id_,bytes));sparse_send_.swap(next);return bytes;
    }
    return 0U;
}
double* PartitionWorkspace::sparse_send() const {return sparse_send_ ? static_cast<double*>(sparse_send_->data()) : nullptr;}
std::size_t PartitionWorkspace::sparse_send_capacity_bytes() const {return allocation_bytes(sparse_send_);}

void PartitionWorkspace::ensure(
    std::size_t local_vertex_count, std::size_t owned_vertex_count) {
    check_current_device(device_id_, "cudaGetDevice before workspace allocation");
    const std::size_t local_bytes = local_vertex_count * sizeof(double);
    const std::size_t owned_bytes = owned_vertex_count * sizeof(double);
    if (owned_bytes != 0 &&
        (!current_rank_ || current_rank_->bytes() < owned_bytes)) {
        current_rank_.reset(new DeviceAllocation(device_id_, owned_bytes));
    }
    if (owned_bytes != 0 &&
        (!next_rank_ || next_rank_->bytes() < owned_bytes)) {
        next_rank_.reset(new DeviceAllocation(device_id_, owned_bytes));
    }
    if (local_bytes != 0 &&
        (!local_contribution_ || local_contribution_->bytes() < local_bytes)) {
        local_contribution_.reset(new DeviceAllocation(device_id_, local_bytes));
    }
    if (!scalars_) {
        scalars_.reset(new DeviceAllocation(device_id_, 2U * sizeof(double)));
    }
}

int PartitionWorkspace::device_id() const { return device_id_; }
double* PartitionWorkspace::current_rank() const {
    return current_rank_ ? static_cast<double*>(current_rank_->data()) : nullptr;
}
double* PartitionWorkspace::next_rank() const {
    return next_rank_ ? static_cast<double*>(next_rank_->data()) : nullptr;
}
double* PartitionWorkspace::local_contribution() const {
    return local_contribution_
        ? static_cast<double*>(local_contribution_->data())
        : nullptr;
}
double* PartitionWorkspace::scalars() const {
    return static_cast<double*>(scalars_->data());
}
void PartitionWorkspace::advance_rank() {
    current_rank_.swap(next_rank_);
}
std::size_t PartitionWorkspace::capacity_bytes() const {
    return allocation_bytes(current_rank_) + allocation_bytes(next_rank_) +
        allocation_bytes(local_contribution_) + allocation_bytes(scalars_) + allocation_bytes(sparse_send_);
}

bool PartitionedDeviceStateRegistry::RegistryKey::operator==(
    const RegistryKey& other) const {
    return partition_id == other.partition_id && state == other.state;
}

std::size_t PartitionedDeviceStateRegistry::RegistryKeyHash::operator()(
    const RegistryKey& key) const {
    std::size_t seed = StateKeyHash()(key.state);
    seed ^= std::hash<int>()(key.partition_id) +
        static_cast<std::size_t>(0x9e3779b9U) + (seed << 6U) + (seed >> 2U);
    return seed;
}

void PartitionedDeviceStateRegistry::touch_and_trim(const RegistryKey& key) {
    const std::size_t capacity = partition_state_cache_capacity();
    for (std::deque<RegistryKey>::iterator item = lru_.begin();
         item != lru_.end();) {
        if (*item == key) item = lru_.erase(item);
        else ++item;
    }
    lru_.push_back(key);
    std::size_t remaining = lru_.size();
    while (lru_.size() > capacity && remaining--) {
        const RegistryKey victim = lru_.front();
        lru_.pop_front();
        const auto view = views_.find(victim);
        const auto workspace = workspaces_.find(victim);
        if ((view != views_.end() && view->second.use_count() > 1U) ||
            (workspace != workspaces_.end() && workspace->second.use_count() > 1U)) {
            // A live readonly view remains inventoried until its lease ends.
            // Cache entry count is a soft bound while every victim is leased.
            lru_.push_back(victim);
            continue;
        }
        views_.erase(victim);
        workspaces_.erase(victim);
    }
}

std::vector<std::shared_ptr<ResidentPartitionView> >
PartitionedDeviceStateRegistry::hold_graph_views(
    const std::string& graph_id, std::uint64_t generation) const {
    std::vector<std::shared_ptr<ResidentPartitionView> > held;
    for (const auto& item : views_) {
        if (item.second->graph_identity_text() == graph_id &&
            item.second->state_key().generation == generation)
            held.push_back(item.second);
    }
    return held;
}

std::shared_ptr<ResidentPartitionView> PartitionedDeviceStateRegistry::acquire(
    const PartitionDeviceUpload& upload,
    bool* cache_hit,
    SharedArrayCallStats* shared_stats) {
    const std::string projection_signature = page_rank_projection_signature(upload);
    const RegistryKey key{
        StateKey(
            upload.graph_identity,
            upload.generation,
            stable_signature64(projection_signature),
            DeviceGroup(upload.device_group),
            GraphLayout::Partitioned),
        upload.partition_id};
    std::unordered_map<
        RegistryKey,
        std::shared_ptr<ResidentPartitionView>,
        RegistryKeyHash>::iterator found = views_.find(key);
    if (found != views_.end()) {
        if (found->second->graph_identity_text() == upload.graph_identity_text &&
            found->second->orientation() == upload.orientation &&
            found->second->view_signature_text() == projection_signature) {
            if (cache_hit) *cache_hit = true;
            if (shared_stats) {
                shared_stats->hits = found->second->shared_array_count();
                shared_stats->new_bytes = 0U;
                shared_stats->retained_bytes =
                    shared_device_array_catalog().stats().retained_bytes;
                shared_stats->all_retained = true;
            }
            std::shared_ptr<ResidentPartitionView> result = found->second;
            touch_and_trim(key);
            return result;
        }
        // A 64-bit signature collision must never alias two physical views.
        views_.erase(found);
        workspaces_.erase(key);
    }
    std::shared_ptr<ResidentPartitionView> built =
        ResidentPartitionView::upload(upload);
    // A newer generation supersedes older cached generations for this graph,
    // partition, view, and device group.  In-flight calls retain shared leases.
    for (std::unordered_map<
             RegistryKey,
             std::shared_ptr<ResidentPartitionView>,
             RegistryKeyHash>::iterator item = views_.begin();
         item != views_.end();) {
        const StateKey& state = item->first.state;
        if (item->first.partition_id == upload.partition_id &&
            state.graph_identity == upload.graph_identity &&
            state.device_group == key.state.device_group &&
            state.view_signature == key.state.view_signature &&
            state.generation != upload.generation) {
            workspaces_.erase(item->first);
            item = views_.erase(item);
        } else {
            ++item;
        }
    }
    if (built->cacheable()) {
        touch_and_trim(key);
        views_[key] = built;
    }
    if (cache_hit) *cache_hit = false;
    if (shared_stats) {
        *shared_stats = built->shared_array_stats();
        shared_stats->retained_bytes =
            shared_device_array_catalog().stats().retained_bytes;
    }
    return built;
}

std::shared_ptr<PartitionWorkspace> PartitionedDeviceStateRegistry::workspace(
    const std::shared_ptr<ResidentPartitionView>& view) {
    if (!view) throw std::invalid_argument("workspace requires a resident view");
    if (managed_workspace_per_worker()) {
        if (!pooled_workspace_) {
            pooled_workspace_.reset(new PartitionWorkspace(view->view().device_id));
        } else if (pooled_workspace_.use_count() != 1) {
            // A caller may hold another partition on this worker (serial2), or
            // a prior lease may still be live. Never alias its mutable state.
            ++managed_pool_fallbacks_;
            std::shared_ptr<PartitionWorkspace> transient(new PartitionWorkspace(view->view().device_id));
            transient->ensure(view->view().local_vertex_count, view->view().owned_vertex_count);
            return transient;
        } else {
            ++managed_pool_reuses_;
        }
        pooled_workspace_->ensure(view->view().local_vertex_count, view->view().owned_vertex_count);
        return pooled_workspace_;
    }
    if (!view->cacheable()) {
        std::shared_ptr<PartitionWorkspace> transient(
            new PartitionWorkspace(view->view().device_id));
        transient->ensure(
            view->view().local_vertex_count, view->view().owned_vertex_count);
        return transient;
    }
    const RegistryKey key{view->state_key(), view->view().partition_id};
    std::shared_ptr<PartitionWorkspace>& entry = workspaces_[key];
    if (!entry) {
        entry.reset(new PartitionWorkspace(view->view().device_id));
    }
    entry->ensure(
        view->view().local_vertex_count, view->view().owned_vertex_count);
    return entry;
}

void PartitionedDeviceStateRegistry::evict(
    const StateKey& key, int partition_id) {
    const RegistryKey registry_key{key, partition_id};
    views_.erase(registry_key);
    workspaces_.erase(registry_key);
    for (std::deque<RegistryKey>::iterator item = lru_.begin();
         item != lru_.end();) {
        if (*item == registry_key) item = lru_.erase(item);
        else ++item;
    }
}

void PartitionedDeviceStateRegistry::clear() {
    pooled_workspace_.reset();
    views_.clear();
    workspaces_.clear();
    lru_.clear();
}

std::size_t PartitionedDeviceStateRegistry::resident_view_count() const {
    return views_.size();
}
std::size_t PartitionedDeviceStateRegistry::workspace_count() const {
    return workspaces_.size() + (pooled_workspace_ ? 1U : 0U);
}

PartitionedDeviceStateRegistry& partitioned_device_state_registry() {
    static thread_local PartitionedDeviceStateRegistry registry;
    return registry;
}

std::uint64_t stable_signature64(const std::string& value) {
    const std::uint64_t offset = UINT64_C(1469598103934665603);
    const std::uint64_t prime = UINT64_C(1099511628211);
    std::uint64_t hash = offset;
    for (std::size_t index = 0; index < value.size(); ++index) {
        hash ^= static_cast<unsigned char>(value[index]);
        hash *= prime;
    }
    return hash;
}


void ResidentPartitionView::append_managed_allocations(
    ManagedStateInventory& inventory, const ManagedOwnerRecord& owner) const {
    inventory.owner(owner);
    inventory.add(offsets_, owner.owner_id, "pr.offsets", false);
    inventory.add(indices_, owner.owner_id, "pr.indices", false);
    inventory.add(shared_offsets_, owner.owner_id, "pr.shared_offsets", true);
    inventory.add(shared_indices_, owner.owner_id, "pr.shared_indices", true);
    inventory.add(weights_, owner.owner_id, "pr.weights", true);
    inventory.add(local_to_global_, owner.owner_id, "pr.local_to_global", true);
    inventory.add(owners_, owner.owner_id, "pr.owners", true);
    inventory.add(owned_out_weight_, owner.owner_id, "pr.owned_out_weight", false);
    inventory.add(pagerank_heavy_rows_, owner.owner_id, "pr.pagerank_heavy_rows", false);
    inventory.add(halo_source_local_, owner.owner_id, "pr.halo_source_local", false);
    inventory.add(halo_destination_local_, owner.owner_id, "pr.halo_destination_local", false);
    inventory.add(sparse_halo_source_,owner.owner_id,"pr.sparse_send_source_map",false);
    inventory.add(sparse_halo_packed_,owner.owner_id,"pr.sparse_send_packed_map",false);
}

void PartitionWorkspace::append_managed_allocations(
    ManagedStateInventory& inventory, const ManagedOwnerRecord& owner) const {
    inventory.owner(owner);
    inventory.add(current_rank_, owner.owner_id, "pr.workspace.current_rank");
    inventory.add(next_rank_, owner.owner_id, "pr.workspace.next_rank");
    inventory.add(local_contribution_, owner.owner_id, "pr.workspace.local_contribution");
    inventory.add(scalars_, owner.owner_id, "pr.workspace.scalars");
    inventory.add(sparse_send_,owner.owner_id,"pr.workspace.sparse_send");
}

void PartitionedDeviceStateRegistry::append_managed_inventory(ManagedStateInventory& inventory) const {
    for (const auto& entry : views_) {
        const auto& view = entry.second;
        ManagedOwnerRecord owner;
        owner.owner_id = view->managed_owner_id();
        owner.kind = "view"; owner.algorithm = "pr";
        owner.device_id = inventory.device_id;
        owner.graph_id = view->graph_identity_text();
        owner.partition_id = entry.first.partition_id;
        owner.generation = entry.first.state.generation;
        owner.active_leases = view.use_count() - 1U;
        view->append_managed_allocations(inventory, owner);
    }
    for (const auto& entry : workspaces_) {
        const auto& workspace = entry.second;
        ManagedOwnerRecord owner;
        owner.owner_id = workspace->managed_owner_id();
        owner.kind = "workspace"; owner.algorithm = "pr";
        owner.policy = "per_view"; owner.device_id = inventory.device_id;
        owner.partition_id = entry.first.partition_id;
        owner.generation = entry.first.state.generation;
        const auto view = views_.find(entry.first);
        if (view != views_.end()) {
            owner.graph_id = view->second->graph_identity_text();
            owner.parent_owner_id = view->second->managed_owner_id();
        }
        owner.active_leases = workspace.use_count() - 1U;
        workspace->append_managed_allocations(inventory, owner);
    }
    if (pooled_workspace_) {
        ManagedOwnerRecord owner;
        owner.owner_id = pooled_workspace_->managed_owner_id();
        owner.kind = "workspace"; owner.algorithm = "pr";
        owner.policy = "per_worker"; owner.device_id = inventory.device_id;
        owner.active_leases = pooled_workspace_.use_count() - 1U;
        pooled_workspace_->append_managed_allocations(inventory, owner);
    }
    inventory.counters["pr_pool_reuses"] = managed_pool_reuses_;
    inventory.counters["pr_pool_active_fallbacks"] = managed_pool_fallbacks_;
}

void PartitionedDeviceStateRegistry::managed_evict(const std::set<std::string>& requested,
    bool release_workspaces, ManagedEvictRecord& receipt) {
    for (auto item = views_.begin(); item != views_.end();) {
        const std::string id = item->second->managed_owner_id();
        if (!requested.count(id)) { ++item; continue; }
        const auto workspace = workspaces_.find(item->first);
        if (item->second.use_count() != 1 ||
            (workspace != workspaces_.end() && workspace->second.use_count() != 1)) {
            receipt.skipped_active.push_back(id); ++item; continue;
        }
        if (workspace != workspaces_.end()) {
            receipt.released_workspaces.push_back(workspace->second->managed_owner_id());
            workspaces_.erase(workspace);
        }
        const RegistryKey key = item->first;
        for (auto lru = lru_.begin(); lru != lru_.end();) {
            if (*lru == key) lru = lru_.erase(lru); else ++lru;
        }
        receipt.removed.push_back(id);
        item = views_.erase(item);
    }
    for (auto item = workspaces_.begin(); item != workspaces_.end();) {
        const std::string id = item->second->managed_owner_id();
        if (!release_workspaces && !requested.count(id)) { ++item; continue; }
        if (item->second.use_count() != 1) {
            receipt.skipped_active.push_back(id); ++item; continue;
        }
        receipt.released_workspaces.push_back(id);
        item = workspaces_.erase(item);
    }
    if (pooled_workspace_ &&
        (release_workspaces || requested.count(pooled_workspace_->managed_owner_id()))) {
        if (pooled_workspace_.use_count() == 1) {
            receipt.released_workspaces.push_back(pooled_workspace_->managed_owner_id());
            pooled_workspace_.reset();
        } else {
            receipt.skipped_active.push_back(pooled_workspace_->managed_owner_id());
        }
    }
}

}  // namespace runtime
}  // namespace gpu_easygraph
