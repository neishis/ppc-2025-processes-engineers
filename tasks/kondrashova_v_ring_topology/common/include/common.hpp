#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace kondrashova_v_ring_topology {

struct Data {
  int source;
  int recipient;
  std::vector<int> data;
};

using InType = Data;
using OutType = std::vector<int>;

using TestType = std::tuple<int, int, std::vector<int>, std::string>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kondrashova_v_ring_topology
