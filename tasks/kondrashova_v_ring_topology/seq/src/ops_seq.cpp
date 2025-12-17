#include "kondrashova_v_ring_topology/seq/include/ops_seq.hpp"

#include <vector>

#include "kondrashova_v_ring_topology/common/include/common.hpp"

namespace kondrashova_v_ring_topology {

KondrashovaVRingTopologySEQ::KondrashovaVRingTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KondrashovaVRingTopologySEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.source < 0 || input.recipient < 0) {
    return false;
  }

  return true;
}

bool KondrashovaVRingTopologySEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool KondrashovaVRingTopologySEQ::RunImpl() {
  const auto &input = GetInput();

  GetOutput() = input.data;

  return true;
}

bool KondrashovaVRingTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kondrashova_v_ring_topology
