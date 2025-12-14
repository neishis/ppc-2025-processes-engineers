#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "kondrashova_v_ring_topology/common/include/common.hpp"
#include "kondrashova_v_ring_topology/mpi/include/ops_mpi.hpp"
#include "kondrashova_v_ring_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kondrashova_v_ring_topology {

class KondrashovaVRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 100000000;

  InType input_data{};
  OutType expected_output{};

  void SetUp() override {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    input_data.source = 0;
    input_data.recipient = world_size - 1;

    input_data.data.resize(kDataSize);
    std::iota(input_data.data.begin(), input_data.data.end(), 0);

    expected_output = input_data.data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == input_data.recipient) {
      if (output_data.size() != expected_output.size()) {
        return false;
      }
      return output_data == expected_output;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(KondrashovaVRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KondrashovaVRingTopologyMPI, KondrashovaVRingTopologySEQ>(
        PPC_SETTINGS_kondrashova_v_ring_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KondrashovaVRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KondrashovaVRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kondrashova_v_ring_topology
