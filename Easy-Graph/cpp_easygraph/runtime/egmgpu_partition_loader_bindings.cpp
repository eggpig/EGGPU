#include "egmgpu_partition_loader_bindings.h"

#include "egmgpu_partition_loader.h"
#include "egmgpu_sha256.h"

#include <memory>
#include <chrono>
#include <exception>
#include <limits>
#include <string>

namespace {

using easygraph::egmgpu::MultiPartitionGraph;
using easygraph::egmgpu::MultiPartitionLoadSpec;
using easygraph::egmgpu::PartitionCSR;
using easygraph::egmgpu::PartitionLoadSpec;
using easygraph::egmgpu::PartitionWeightType;

std::string optional_string(const py::object& value) {
    return value.is_none() ? std::string() : value.cast<std::string>();
}

PartitionWeightType parse_weight_type(const py::object& weights) {
    if (weights.is_none()) {
        return PartitionWeightType::kNone;
    }
    const std::string dtype = weights.attr("dtype").cast<std::string>();
    if (dtype == "float32") {
        return PartitionWeightType::kFloat32;
    }
    if (dtype == "float64") {
        return PartitionWeightType::kFloat64;
    }
    throw std::invalid_argument("validated manifest has an unsupported weight dtype");
}

std::string resolved_array_path(const py::object& array,
                                const py::object& manifest_root) {
    return py::str(array.attr("resolve")(manifest_root)).cast<std::string>();
}

std::string array_sha256(const py::object& array) {
    return array.attr("sha256").cast<std::string>();
}

typedef std::chrono::steady_clock BindingClock;

double seconds_since(const BindingClock::time_point& start) {
    return std::chrono::duration<double>(BindingClock::now() - start).count();
}

std::string canonical_signature(const py::object& value) {
    const std::string bytes = py::module_::import("json").attr("dumps")(
        value, py::arg("sort_keys")=true, py::arg("separators")=py::make_tuple(",", ":"),
        py::arg("ensure_ascii")=true).cast<std::string>();
    easygraph::egmgpu::Sha256 digest;
    digest.update(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
    return digest.final_hex();
}

std::pair<std::string, std::string> native_content_signatures(const MultiPartitionGraph& graph) {
    py::gil_scoped_acquire gil;
    py::list weights;
    for (const auto& part : graph.partitions()) {
        py::dict array, item;
        array["dtype"]="float64"; array["count"]=part.adjacency_count;
        array["sha256"]=part.weights_sha256;
        item["partition_id"]=part.partition_id; item["array"]=array;
        weights.append(item);
    }
    py::dict weight; weight["weights"]=weights;
    const std::string weight_signature=canonical_signature(weight);
    py::dict view;
    view["orientation"]=graph.view_orientation();
    view["weight_key"]=graph.weight_key().empty() ? py::object(py::none()) : py::cast(graph.weight_key());
    view["topology_sha256"]=graph.topology_signature(); view["weight_sha256"]=weight_signature;
    return {weight_signature, canonical_signature(view)};
}

std::string hashlib_weight_sha256(
    const std::shared_ptr<MultiPartitionGraph>& target,
    const unsigned char* data, std::size_t bytes) {
    // Preserve the native size domain even on a Python build with a smaller
    // buffer index range. Ordinary hashing failures still propagate below.
    if (bytes > static_cast<std::size_t>(PY_SSIZE_T_MAX)) {
        easygraph::egmgpu::Sha256 digest;
        digest.update(data, bytes);
        return digest.final_hex();
    }
    // Derived Python buffers can outlive both this callback and a failed commit.
    // The exporter owns only this target; the target has no reverse reference.
    // No Python objects are captured across the caller's released-GIL scope.
    py::gil_scoped_acquire gil;
    using PayloadOwner = std::shared_ptr<MultiPartitionGraph>;
    std::unique_ptr<PayloadOwner> holder(new PayloadOwner(target));
    py::capsule owner(holder.get(), [](void* pointer) {
        delete static_cast<PayloadOwner*>(pointer);
    });
    holder.release();  // Transfer only after capsule creation succeeds.
    py::array exporter(py::dtype::of<std::uint8_t>(),
                       {static_cast<py::ssize_t>(bytes)}, {py::ssize_t(1)},
                       data, owner);
    exporter.attr("setflags")(py::arg("write") = false);
    py::object view = py::reinterpret_steal<py::object>(
        PyMemoryView_FromObject(exporter.ptr()));
    if (!view) throw py::error_already_set();
    try {
        // hashlib may release the GIL; the immutable target stays owned by
        // every derived buffer. Hash all original FP64 bytes without a copy.
        const std::string digest = py::module_::import("hashlib").attr("sha256")(
            view).attr("hexdigest")().cast<std::string>();
        view.attr("release")();
        return digest;
    } catch (...) {
        const std::exception_ptr error = std::current_exception();
        // Release this view, preserving the original exception. Derived views
        // retain the exporter/target. Python reference destruction holds the GIL.
        PyObject* released = PyObject_CallMethod(view.ptr(), "release", nullptr);
        if (released) Py_DECREF(released);
        else PyErr_Clear();
        std::rethrow_exception(error);
    }
}

py::dict native_commit_profile(const MultiPartitionGraph& graph) {
    const auto& p=graph.native_commit_profile();
    py::dict result;
    result["snapshot_origin"]=graph.snapshot_origin();
    result["base_graph_id"]=p.base_graph_id;result["base_view_signature"]=p.base_view_signature;
    result["requested_positions"]=p.requested_positions;result["changed_positions"]=p.changed_positions;
    result["copied_array_payload_bytes"]=p.copied_array_payload_bytes;
    result["shared_topology_payload_bytes"]=p.shared_topology_payload_bytes;
    result["shared_unchanged_weight_payload_bytes"]=p.shared_unchanged_weight_payload_bytes;
    result["host_storage_policy"]=p.host_storage_policy;
    result["weight_bytes_hashed"]=p.weight_bytes_hashed;result["packet_payload_bytes"]=p.packet_payload_bytes;
    result["validation_seconds"]=p.validation_seconds;result["clone_seconds"]=p.clone_seconds;
    result["apply_seconds"]=p.apply_seconds;result["weight_hash_seconds"]=p.weight_hash_seconds;
    result["signature_seconds"]=p.signature_seconds;result["total_seconds"]=p.total_seconds;
    result["file_payload_bytes_read"]=0;result["file_payload_bytes_written"]=0;
    result["scope"]="Native immutable f64 snapshot. deep copies every payload; shared detaches actual dirty weight partitions and retains immutable topology/unchanged arrays. Dirty partitions still receive full flat SHA-256. Payload bytes exclude vector metadata, capacity and packet metadata. No GPU execution. Envelope timings are not additive.";
    py::list parts;
    for (const auto& part : graph.partitions()) {
        py::dict row;row["partition_id"]=part.partition_id;row["device_id"]=part.device_id;
        row["weight_sha256"]=part.weights_sha256;row["weight_count"]=part.adjacency_count;
        row["native_packet_positions"]=part.native_weight_patch ? part.native_weight_patch->positions.size() : 0;
        parts.append(row);
    }
    result["partitions"]=parts;
    return result;
}

py::dict commit_weights_memory(const std::shared_ptr<MultiPartitionGraph>& base,
                              const std::string& new_id, const py::iterable& items) {
    const auto begin=BindingClock::now();
    std::vector<easygraph::egmgpu::AbsoluteWeightSet> updates;
    const py::object numpy_bool=py::module_::import("numpy").attr("bool_");
    for (const auto& item : items) {
        if (!PySequence_Check(item.ptr())) throw std::invalid_argument("update must contain partition, position, absolute value");
        const py::sequence row=py::reinterpret_borrow<py::sequence>(item);
        if (py::len(row)!=3) throw std::invalid_argument("update must contain exactly three fields");
        std::int64_t indices[2];
        for (int i=0;i<2;++i) {
            const py::object value=row[i];
            if (PyBool_Check(value.ptr()) || py::isinstance(value,numpy_bool) || !PyIndex_Check(value.ptr()))
                throw std::invalid_argument("partition and position require non-boolean integers");
            py::object integer=py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
            if (!integer) throw py::error_already_set();
            indices[i]=integer.cast<std::int64_t>();
        }
        if (indices[0]<0 || indices[0]>std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument("partition outside int32 domain");
        easygraph::egmgpu::AbsoluteWeightSet update;
        update.partition_id=static_cast<std::int32_t>(indices[0]);update.position=indices[1];
        update.value=py::float_(row[2]).cast<double>();
        updates.push_back(update);
    }
    std::shared_ptr<MultiPartitionGraph> target;
    {
        py::gil_scoped_release release;
        target=MultiPartitionGraph::commit_absolute_weights(
            base,new_id,updates,native_content_signatures,hashlib_weight_sha256);
    }
    py::dict result=native_commit_profile(*target);
    result["graph"]=py::cast(target);result["graph_id"]=target->graph_id();result["generation"]=target->generation();
    result["topology_sha256"]=target->topology_signature();result["weight_sha256"]=target->weight_signature();
    result["view_signature"]=target->view_signature();result["binding_seconds"]=seconds_since(begin);
    return result;
}

MultiPartitionLoadSpec parse_validated_manifest(const std::string& manifest_path,
                                               const std::string& import_policy) {
    const BindingClock::time_point metadata_start = BindingClock::now();
    // This function is called only while the binding holds the GIL. The Python
    // parser owns JSON/schema/path metadata. Native loading verifies the
    // actual buffers once; do not scan every value in Python beforehand.
    // A standalone control package avoids importing unrelated EasyGraph ML
    // modules. Its absence (including a namespace without an entry point)
    // selects the compatibility package path;
    // dependency failures from a present control package propagate unchanged.
    const py::object standalone = py::module_::import("importlib.util")
        .attr("find_spec")("egmgpu_control");
    const bool standalone_available = !standalone.is_none() &&
        !standalone.attr("loader").is_none();
    py::object control = py::module_::import(standalone_available
        ? "egmgpu_control" : "easygraph.egmgpu_control");
    py::object manifest = control.attr("load_partition_manifest")(
        manifest_path, py::arg("verify_checksums") = false,
        py::arg("validate_arrays") = false);
    const double metadata_seconds = seconds_since(metadata_start);
    const BindingClock::time_point projection_start = BindingClock::now();
    py::object root = manifest.attr("root");

    MultiPartitionLoadSpec spec;
    spec.import_policy = import_policy;
    spec.manifest_metadata_seconds = metadata_seconds;
    spec.manifest_validated = true;
    spec.graph_id = manifest.attr("graph_id").cast<std::string>();
    spec.generation = manifest.attr("generation").cast<std::uint64_t>();
    spec.directed = manifest.attr("directed").cast<bool>();
    spec.weighted = manifest.attr("weighted").cast<bool>();
    spec.view_orientation =
        manifest.attr("view_orientation").cast<std::string>();
    spec.weight_key = optional_string(manifest.attr("weight_key"));
    spec.view_signature = manifest.attr("view_signature").cast<std::string>();
    spec.topology_signature =
        manifest.attr("topology_sha256").cast<std::string>();
    spec.weight_signature = optional_string(manifest.attr("weight_sha256"));
    spec.vertex_count = manifest.attr("vertex_count").cast<std::uint64_t>();
    spec.edge_count = manifest.attr("edge_count").cast<std::uint64_t>();
    spec.adjacency_count =
        manifest.attr("adjacency_count").cast<std::uint64_t>();
    spec.ownership_encoding =
        manifest.attr("ownership_encoding").cast<std::string>();

    const py::tuple partitions = manifest.attr("partitions").cast<py::tuple>();
    spec.partitions.reserve(partitions.size());
    for (std::size_t index = 0; index < partitions.size(); ++index) {
        const py::object partition = partitions[index];
        PartitionLoadSpec partition_spec;
        partition_spec.partition_id =
            partition.attr("partition_id").cast<std::int32_t>();
        partition_spec.device_id =
            partition.attr("device_id").cast<std::int32_t>();
        partition_spec.ownership_encoding =
            partition.attr("ownership_encoding").cast<std::string>();
        partition_spec.owned_vertex_count =
            partition.attr("owned_vertex_count").cast<std::uint64_t>();
        partition_spec.halo_vertex_count =
            partition.attr("halo_vertex_count").cast<std::uint64_t>();
        partition_spec.adjacency_count =
            partition.attr("adjacency_count").cast<std::uint64_t>();
        partition_spec.cut_edge_count =
            partition.attr("cut_edge_count").cast<std::uint64_t>();
        const py::object vertex_begin = partition.attr("vertex_begin");
        const py::object vertex_end = partition.attr("vertex_end");
        partition_spec.has_vertex_range =
            !vertex_begin.is_none() && !vertex_end.is_none();
        if (partition_spec.has_vertex_range) {
            partition_spec.vertex_begin = vertex_begin.cast<std::uint64_t>();
            partition_spec.vertex_end = vertex_end.cast<std::uint64_t>();
        }
        const py::object offsets = partition.attr("offsets");
        const py::object indices = partition.attr("indices");
        const py::object local_to_global = partition.attr("local_to_global");
        const py::object owners = partition.attr("owners");
        partition_spec.offsets_path = resolved_array_path(offsets, root);
        partition_spec.offsets_sha256 = array_sha256(offsets);
        partition_spec.indices_path = resolved_array_path(indices, root);
        partition_spec.indices_sha256 = array_sha256(indices);
        partition_spec.local_to_global_path =
            resolved_array_path(local_to_global, root);
        partition_spec.local_to_global_sha256 = array_sha256(local_to_global);
        partition_spec.owners_path = resolved_array_path(owners, root);
        partition_spec.owners_sha256 = array_sha256(owners);
        const py::object owned_global_ids = partition.attr("owned_global_ids");
        if (!owned_global_ids.is_none()) {
            partition_spec.owned_global_ids_path =
                resolved_array_path(owned_global_ids, root);
            partition_spec.owned_global_ids_sha256 =
                array_sha256(owned_global_ids);
        }
        const py::object weights = partition.attr("weights");
        partition_spec.weight_type = parse_weight_type(weights);
        if (!weights.is_none()) {
            partition_spec.weights_path = resolved_array_path(weights, root);
            partition_spec.weights_sha256 = array_sha256(weights);
        }
        spec.partitions.push_back(partition_spec);
    }
    spec.manifest_projection_seconds = seconds_since(projection_start);
    return spec;
}

std::shared_ptr<MultiPartitionGraph> load_partition_manifest(
    const std::string& manifest_path, const std::string& import_policy) {
    if (import_policy != "full" && import_policy != "structural") {
        throw std::invalid_argument("import_policy must be full or structural");
    }
    const MultiPartitionLoadSpec spec =
        parse_validated_manifest(manifest_path, import_policy);
    py::gil_scoped_release release;
    return easygraph::egmgpu::load_multi_partition_graph(spec);
}

// Called after import, outside the timed loader; no profile JSON construction
// or file write occurs in the import/query paths.
py::dict import_profile_dict(const MultiPartitionGraph& graph) {
    const easygraph::egmgpu::GraphImportProfile& profile = graph.import_profile();
    py::dict result;
    result["schema"] = "egmgpu-import-hostwall-v1";
    result["policy"] = profile.policy;
    result["array_content_digests_verified"] = profile.policy == "full";
    result["catalog_identity_metadata_validated"] = true;
    result["immutable_input_contract"] = graph.snapshot_origin()=="native_memory"
        ? "Native arrays derive from a verified immutable base; changed f64 domains are newly hashed. No file import occurred."
        : "Prepared files stay immutable; graph changes require a new generation.";
    result["clock"] = "std::chrono::steady_clock host wall";
    result["graph_id"] = graph.graph_id();
    result["generation"] = graph.generation();
    result["topology_signature"] = graph.topology_signature();
    result["view_signature"] = graph.view_signature();
    result["manifest_metadata_seconds"] = profile.manifest_metadata_seconds;
    result["manifest_projection_seconds"] = profile.manifest_projection_seconds;
    result["native_total_seconds"] = profile.native_total_seconds;
    result["partition_totals_are_envelopes"] = true;
    py::list partitions;
    double open_size = 0.0, allocation = 0.0, read = 0.0, sha256 = 0.0;
    double endian = 0.0, close = 0.0, metadata = 0.0, structural = 0.0;
    std::uint64_t bytes_read = 0, bytes_hashed = 0, bytes_swapped = 0;
    for (const auto& partition : profile.partitions) {
        py::dict part;
        part["partition_id"] = partition.partition_id;
        part["metadata_seconds"] = partition.metadata_seconds;
        part["structural_seconds"] = partition.structural_seconds;
        part["total_seconds"] = partition.total_seconds;
        metadata += partition.metadata_seconds;
        structural += partition.structural_seconds;
        double part_measured = partition.metadata_seconds + partition.structural_seconds;
        py::list arrays;
        for (const auto& array : partition.arrays) {
            py::dict item;
            item["name"] = array.name;
            item["path"] = array.path;
            item["count"] = array.count;
            item["item_bytes"] = array.item_bytes;
            item["bytes_read"] = array.bytes_read;
            item["bytes_hashed"] = array.bytes_hashed;
            item["bytes_swapped"] = array.bytes_swapped;
            item["open_size_seconds"] = array.open_size_seconds;
            item["allocation_seconds"] = array.allocation_seconds;
            item["read_seconds"] = array.read_seconds;
            item["sha256_seconds"] = array.sha256_seconds;
            item["endian_seconds"] = array.endian_seconds;
            item["close_seconds"] = array.close_seconds;
            item["total_seconds"] = array.total_seconds;
            const double array_measured = array.open_size_seconds + array.allocation_seconds
                + array.read_seconds + array.sha256_seconds + array.endian_seconds + array.close_seconds;
            item["unattributed_seconds"] = array.total_seconds - array_measured;
            part_measured += array.total_seconds;
            open_size += array.open_size_seconds;
            allocation += array.allocation_seconds;
            read += array.read_seconds;
            sha256 += array.sha256_seconds;
            endian += array.endian_seconds;
            close += array.close_seconds;
            bytes_read += array.bytes_read;
            bytes_hashed += array.bytes_hashed;
            bytes_swapped += array.bytes_swapped;
            arrays.append(item);
        }
        part["arrays"] = arrays;
        part["unattributed_seconds"] = partition.total_seconds - part_measured;
        partitions.append(part);
    }
    result["partitions"] = partitions;
    result["bytes_read"] = bytes_read;
    result["bytes_hashed"] = bytes_hashed;
    result["bytes_swapped"] = bytes_swapped;
    py::dict phases;
    phases["array_open_size"] = open_size;
    phases["array_allocation_value_init"] = allocation;
    phases["array_read_complete"] = read;
    phases["array_sha256"] = sha256;
    phases["array_endian"] = endian;
    phases["array_close"] = close;
    phases["partition_metadata"] = metadata;
    phases["partition_structural"] = structural;
    phases["global_metadata"] = profile.global_metadata_seconds;
    phases["global_counts"] = profile.global_counts_seconds;
    phases["global_owned_cover"] = profile.global_cover_seconds;
    phases["global_halo_membership"] = profile.global_halo_seconds;
    const double measured = open_size + allocation + read + sha256 + endian + close
        + metadata + structural + profile.global_metadata_seconds + profile.global_counts_seconds
        + profile.global_cover_seconds + profile.global_halo_seconds;
    // Residual is bookkeeping/instrumentation outside the explicit intervals.
    // It must not be relabelled file I/O, validation, CPU kernel, or GPU time.
    phases["unattributed_bookkeeping"] = profile.native_total_seconds - measured;
    result["native_disjoint_seconds"] = phases;
    return result;
}

}  // namespace

void bind_egmgpu_partition_loader(py::module_& module) {
    py::class_<PartitionCSR>(module, "_EGMGPUPartitionCSR")
        .def_readonly("partition_id", &PartitionCSR::partition_id)
        .def_readonly("device_id", &PartitionCSR::device_id)
        .def_readonly("owned_vertex_count", &PartitionCSR::owned_vertex_count)
        .def_readonly("halo_vertex_count", &PartitionCSR::halo_vertex_count)
        .def_readonly("adjacency_count", &PartitionCSR::adjacency_count)
        .def_readonly("cut_edge_count", &PartitionCSR::cut_edge_count)
        .def_readonly("has_vertex_range", &PartitionCSR::has_vertex_range)
        .def_readonly("vertex_begin", &PartitionCSR::vertex_begin)
        .def_readonly("vertex_end", &PartitionCSR::vertex_end)
        .def_property_readonly("offsets", [](const PartitionCSR& p) -> const std::vector<std::int32_t>& { return p.offsets.view(); })
        .def_readonly("offsets_sha256", &PartitionCSR::offsets_sha256)
        .def_property_readonly("indices", [](const PartitionCSR& p) -> const std::vector<std::int32_t>& { return p.indices.view(); })
        .def_readonly("indices_sha256", &PartitionCSR::indices_sha256)
        .def_property_readonly("local_to_global", [](const PartitionCSR& p) -> const std::vector<std::uint64_t>& { return p.local_to_global.view(); })
        .def_readonly(
            "local_to_global_sha256", &PartitionCSR::local_to_global_sha256)
        .def_property_readonly("owners", [](const PartitionCSR& p) -> const std::vector<std::int32_t>& { return p.owners.view(); })
        .def_readonly("owners_sha256", &PartitionCSR::owners_sha256)
        .def_readonly("ownership_encoding", &PartitionCSR::ownership_encoding)
        .def_property_readonly("owned_global_ids", [](const PartitionCSR& p) -> const std::vector<std::uint64_t>& { return p.owned_global_ids.view(); })
        .def_readonly(
            "owned_global_ids_sha256", &PartitionCSR::owned_global_ids_sha256)
        .def_readonly("weights_sha256", &PartitionCSR::weights_sha256)
        .def_property_readonly("weights_float32", [](const PartitionCSR& p) -> const std::vector<float>& { return p.weights_float32.view(); })
        .def_property_readonly("weights_float64", [](const PartitionCSR& p) -> const std::vector<double>& { return p.weights_float64.view(); })
        .def_property_readonly("host_storage", [](const PartitionCSR& p) {
            py::dict d;
            { py::dict a;a["allocation_id"]=p.offsets.storage_id();a["bytes"]=p.offsets.payload_bytes();a["owners"]=p.offsets.owners();d["offsets"]=a; }
            { py::dict a;a["allocation_id"]=p.indices.storage_id();a["bytes"]=p.indices.payload_bytes();a["owners"]=p.indices.owners();d["indices"]=a; }
            { py::dict a;a["allocation_id"]=p.local_to_global.storage_id();a["bytes"]=p.local_to_global.payload_bytes();a["owners"]=p.local_to_global.owners();d["local_to_global"]=a; }
            { py::dict a;a["allocation_id"]=p.owners.storage_id();a["bytes"]=p.owners.payload_bytes();a["owners"]=p.owners.owners();d["owners"]=a; }
            { py::dict a;a["allocation_id"]=p.owned_global_ids.storage_id();a["bytes"]=p.owned_global_ids.payload_bytes();a["owners"]=p.owned_global_ids.owners();d["owned_global_ids"]=a; }
            { py::dict a;a["allocation_id"]=p.weights_float32.storage_id();a["bytes"]=p.weights_float32.payload_bytes();a["owners"]=p.weights_float32.owners();d["weights_float32"]=a; }
            { py::dict a;a["allocation_id"]=p.weights_float64.storage_id();a["bytes"]=p.weights_float64.payload_bytes();a["owners"]=p.weights_float64.owners();d["weights_float64"]=a; }
            return d;
        })
        .def_property_readonly("local_vertex_count", &PartitionCSR::local_vertex_count)
        .def_property_readonly("weighted", &PartitionCSR::weighted)
        .def_property_readonly("weight_dtype", [](const PartitionCSR& partition) {
            return std::string(easygraph::egmgpu::partition_weight_type_name(
                partition.weight_type));
        });

    py::class_<MultiPartitionGraph, std::shared_ptr<MultiPartitionGraph> >(
        module, "_EGMGPUMultiPartitionGraph")
        .def_property_readonly("snapshot_origin", &MultiPartitionGraph::snapshot_origin)
        .def_property_readonly("native_commit_profile", &native_commit_profile)
        .def("_lifetime_probe", [](const std::shared_ptr<MultiPartitionGraph>& graph) {
            const std::weak_ptr<MultiPartitionGraph> weak=graph;
            return py::cpp_function([weak]() { return !weak.expired(); });
        })
        .def_property_readonly("import_profile", &import_profile_dict)
        .def_property_readonly("graph_id", &MultiPartitionGraph::graph_id)
        .def_property_readonly("generation", &MultiPartitionGraph::generation)
        .def_property_readonly("directed", &MultiPartitionGraph::directed)
        .def_property_readonly("weighted", &MultiPartitionGraph::weighted)
        .def_property_readonly(
            "view_orientation", &MultiPartitionGraph::view_orientation)
        .def_property_readonly("weight_key", &MultiPartitionGraph::weight_key)
        .def_property_readonly(
            "view_signature", &MultiPartitionGraph::view_signature)
        .def_property_readonly(
            "topology_signature", &MultiPartitionGraph::topology_signature)
        .def_property_readonly(
            "weight_signature", &MultiPartitionGraph::weight_signature)
        .def_property_readonly("vertex_count", &MultiPartitionGraph::vertex_count)
        .def_property_readonly("edge_count", &MultiPartitionGraph::edge_count)
        .def_property_readonly(
            "adjacency_count", &MultiPartitionGraph::adjacency_count)
        .def_property_readonly(
            "ownership_encoding", &MultiPartitionGraph::ownership_encoding)
        .def_property_readonly("partition_count", [](const MultiPartitionGraph& graph) {
            return graph.partitions().size();
        })
        .def(
            "partition",
            [](const MultiPartitionGraph& graph, std::size_t index)
                -> const PartitionCSR& {
                if (index >= graph.partitions().size()) {
                    throw py::index_error("partition index out of range");
                }
                return graph.partitions()[index];
            },
            py::return_value_policy::reference_internal);

    module.def(
        "cpp_egmgpu_load_partition_manifest",
        &load_partition_manifest,
        py::arg("manifest_path"), py::arg("import_policy") = "full",
        "Load complete immutable host partitions. full verifies array SHA-256; "
        "structural skips only array SHA-256, retaining catalog identity and "
        "all structural checks. Prepared files must stay immutable; changes "
        "require a new generation. The default is full.");
    module.def("cpp_egmgpu_commit_weights_memory", &commit_weights_memory,
        py::arg("base"), py::arg("new_id"), py::arg("updates"),
        "Construct an immutable f64 snapshot from a verified native base and complete absolute sets. No file payload IO; full host clone and changed-domain hashing remain charged.");
}
