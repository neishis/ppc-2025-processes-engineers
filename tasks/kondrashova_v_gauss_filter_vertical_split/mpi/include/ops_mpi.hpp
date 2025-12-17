#pragma once

#include "kondrashova_v_gauss_filter_vertical_split/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

class KondrashovaVGaussFilterVerticalSplitMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KondrashovaVGaussFilterVerticalSplitMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kondrashova_v_gauss_filter_vertical_split
