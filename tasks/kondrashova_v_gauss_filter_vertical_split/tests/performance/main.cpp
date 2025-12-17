#include <gtest/gtest.h>

#include "kondrashova_v_gauss_filter_vertical_split/common/include/common.hpp"
#include "kondrashova_v_gauss_filter_vertical_split/mpi/include/ops_mpi.hpp"
#include "kondrashova_v_gauss_filter_vertical_split/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

class ExampleRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ExampleRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NesterovATestTaskMPI, NesterovATestTaskSEQ>(PPC_SETTINGS_kondrashova_v_gauss_filter_vertical_split);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kondrashova_v_gauss_filter_vertical_split
