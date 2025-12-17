#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kondrashova_v_gauss_filter_vertical_split
