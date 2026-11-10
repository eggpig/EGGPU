#pragma once
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace gpu_easygraph { namespace runtime {
// Plain owned values: parsing happens once while holding the Python GIL.
// Workers receive snapshots, never references to Python dictionaries or getenv.
struct SsspExecutionPolicy {
    std::string single_owner_execution = "off";
    std::string weight_encoding = "f64";
    std::string wire_encoding = "f64";
    std::string output_team = "worker1";
    std::string atomic_min = "cas";
    std::string exchange_sync = "legacy";
    std::string pinned_placement = "caller";
    std::string host_assembly = "caller";
    std::string relax_policy = "thread";
    std::string control_policy = "atomic";
    std::string weight_identity = "manifest_f64";
    bool profile = true;
    bool pinned_reuse = true;
    bool prepared_uploads = true;
    bool allow_undirected = true;

    static void choice(const std::string& field, const std::string& value,
                       std::initializer_list<const char*> allowed) {
        for (const char* candidate : allowed) if (value == candidate) return;
        throw std::invalid_argument("invalid SSSP execution_policy." + field + ": " + value);
    }
    void validate() const {
        choice("single_owner_execution", single_owner_execution, {"off", "arena_v3_if_eligible"});
        choice("weight_encoding", weight_encoding, {"f64", "u8_exact"});
        choice("wire_encoding", wire_encoding, {"f64", "u32_exact"});
        choice("output_team", output_team, {"worker1", "team16"});
        choice("atomic_min", atomic_min, {"cas", "native_u64"});
        choice("exchange_sync", exchange_sync, {"legacy", "trim"});
        choice("pinned_placement", pinned_placement, {"caller", "numa_local"});
        choice("host_assembly", host_assembly, {"caller", "worker"});
        choice("relax_policy", relax_policy, {"thread", "warp"});
        choice("control_policy", control_policy, {"atomic", "warp"});
        choice("weight_identity", weight_identity, {"compute", "manifest_f64"});
        if (output_team == "team16" && host_assembly != "worker")
            throw std::invalid_argument("SSSP team16 requires worker host assembly");
    }
};
}}  // namespace gpu_easygraph::runtime
