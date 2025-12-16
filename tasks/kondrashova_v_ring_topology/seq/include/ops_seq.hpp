#pragma once

#include <mpi.h>

#include <vector>

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

  void BroadcastParameters(int rank, int &source, int &recipient, int &data_size);
  void PrepareData(int rank, int data_size, std::vector<int> &data);
  bool HandleTrivialCase(int rank, int world_size, int source, int recipient, const std::vector<int> &data);
  static void CreateRingTopology(int world_size, MPI_Comm &ring_comm);
  static void SendData(int rank, int sender, int next_rank, int step, int data_size, const std::vector<int> &data,
                       const std::vector<int> &buffer, MPI_Comm ring_comm);
  void ReceiveData(int rank, int receiver, int prev_rank, int recipient, std::vector<int> &buffer, MPI_Comm ring_comm);
  void PerformRingTransfer(int rank, int world_size, int source, int recipient, int data_size,
                           const std::vector<int> &data);
};

}  // namespace kondrashova_v_ring_topology
