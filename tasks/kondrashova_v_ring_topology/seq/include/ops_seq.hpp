#pragma once

#include <mpi.h>

#include "kondrashova_v_ring_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondrashova_v_ring_topology {

class KondrashovaVRingTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit KondrashovaVRingTopologySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kondrashova_v_ring_topology
