#include "gang_completion.h"

#include <exception>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {

void await_all_or_rethrow(std::vector<std::future<void> >* completions) {
    if (completions == nullptr) {
        throw std::invalid_argument("gang completions cannot be null");
    }
    std::exception_ptr first_failure;
    for (std::size_t i = 0; i < completions->size(); ++i) {
        try {
            (*completions)[i].get();
        } catch (...) {
            if (!first_failure) {
                first_failure = std::current_exception();
            }
        }
    }
    if (first_failure) {
        std::rethrow_exception(first_failure);
    }
}

}  // namespace runtime
}  // namespace gpu_easygraph
