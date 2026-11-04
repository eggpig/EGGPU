#pragma once

#include <future>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

// Drain every device completion before propagating the first failure in
// device order.  CUDA work cannot be preempted safely; draining the gang keeps
// the reserved devices unavailable until every shard has stopped and prevents
// partial cooperative results from escaping.
void await_all_or_rethrow(std::vector<std::future<void> >* completions);

}  // namespace runtime
}  // namespace gpu_easygraph
