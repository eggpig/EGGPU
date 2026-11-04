#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace easygraph { namespace egmgpu {

inline std::uint64_t next_host_array_identity() {
    static std::atomic<std::uint64_t> next(1);
    const auto value=next.fetch_add(1,std::memory_order_relaxed);
    if (!value) throw std::overflow_error("host array identity overflow");
    return value;
}

// Published PartitionCSR objects are accessed through const graph references.
// No mutable element reference, iterator or pointer escapes this container.
// A copied handle can be edited only after detaching its complete owned array.
template<class T> class ImmutableHostArray {
    struct Storage {
        std::vector<T> values;
        const std::uint64_t id;
        explicit Storage(std::vector<T> input):values(std::move(input)),id(next_host_array_identity()) {}
    };
    std::shared_ptr<Storage> storage_;
public:
    typedef T value_type;
    typedef typename std::vector<T>::const_iterator const_iterator;
    ImmutableHostArray():storage_(std::make_shared<Storage>(std::vector<T>())) {}
    explicit ImmutableHostArray(std::vector<T> input):storage_(std::make_shared<Storage>(std::move(input))) {}
    ImmutableHostArray& operator=(std::vector<T> input) {
        storage_=std::make_shared<Storage>(std::move(input));return *this;
    }
    const std::vector<T>& view() const {return storage_->values;}
    operator const std::vector<T>&() const {return view();}
    const T* data() const {return view().data();}
    const T& operator[](std::size_t i) const {return view()[i];}
    const T& at(std::size_t i) const {return view().at(i);}
    const T& front() const {return view().front();}
    const T& back() const {return view().back();}
    std::size_t size() const {return view().size();}
    std::size_t capacity() const {return view().capacity();}
    bool empty() const {return view().empty();}
    const_iterator begin() const {return view().begin();}
    const_iterator end() const {return view().end();}
    const_iterator cbegin() const {return view().cbegin();}
    const_iterator cend() const {return view().cend();}
    std::uint64_t payload_bytes() const {return size()*sizeof(T);}
    std::uint64_t storage_id() const {return empty()?0:storage_->id;}
    long owners() const {return storage_.use_count();}
    bool shares_with(const ImmutableHostArray& other) const {return storage_==other.storage_;}
    ImmutableHostArray deep_clone() const {return ImmutableHostArray(std::vector<T>(view()));}
    // Explicit mutable construction/commit operation; never exposes an alias.
    std::uint64_t detach() {
        if(storage_.unique())return 0;
        const auto bytes=payload_bytes();storage_=std::make_shared<Storage>(std::vector<T>(view()));return bytes;
    }
    void set(std::size_t position,const T& value) {
        if(position>=size())throw std::out_of_range("host array update");
        detach();storage_->values[position]=value;
    }
};

}} // namespace easygraph::egmgpu
