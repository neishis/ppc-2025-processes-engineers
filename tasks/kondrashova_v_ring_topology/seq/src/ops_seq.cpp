#include "kondrashova_v_ring_topology/seq/include/ops_seq.hpp"
#include <mpi.h>

#include <algorithm>
#include <vector>

namespace kondrashova_v_ring_topology {

KondrashovaVRingTopologySEQ::KondrashovaVRingTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KondrashovaVRingTopologySEQ::ValidationImpl() {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int is_valid = 1;

  if (rank == 0) {
    const auto &input = GetInput();

    if (input.source < 0 || input.source >= world_size) {
      is_valid = 0;
    }
    if (input.recipient < 0 || input.recipient >= world_size) {
      is_valid = 0;
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return is_valid == 1;
}

bool KondrashovaVRingTopologySEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool KondrashovaVRingTopologySEQ::RunImpl() {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int source = 0;
  int recipient = 0;
  int data_size = 0;

  if (rank == 0) {
    source = GetInput().source;
    recipient = GetInput().recipient;
    data_size = static_cast<int>(GetInput().data.size());
  }

  MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&recipient, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> data(data_size);
  if (rank == 0) {
    data = GetInput().data;
  }
  if (data_size > 0) {
    MPI_Bcast(data.data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);
  }

  if (world_size == 1 || source == recipient) {
    if (rank == recipient) {
      GetOutput() = data;
    }
    return true;
  }

  int dims[1] = {world_size};
  int periods[1] = {1}; 
  int reorder = 0;

  MPI_Comm ring_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &ring_comm);

  int prev_rank, next_rank;
  MPI_Cart_shift(ring_comm, 0, 1, &prev_rank, &next_rank);

  int steps = (recipient - source + world_size) % world_size;

  std::vector<int> buffer;

  for (int step = 0; step < steps; step++) {
    int sender = (source + step) % world_size;
    int receiver = (sender + 1) % world_size;

    if (rank == sender) {
      if (step == 0) {
        MPI_Send(&data_size, 1, MPI_INT, next_rank, 0, ring_comm);
        if (data_size > 0) {
          MPI_Send(data.data(), data_size, MPI_INT, next_rank, 1, ring_comm);
        }
      } else {
        int buf_size = static_cast<int>(buffer.size());
        MPI_Send(&buf_size, 1, MPI_INT, next_rank, 0, ring_comm);
        if (buf_size > 0) {
          MPI_Send(buffer.data(), buf_size, MPI_INT, next_rank, 1, ring_comm);
        }
      }
    }

    if (rank == receiver) {
      int recv_size = 0;
      MPI_Recv(&recv_size, 1, MPI_INT, prev_rank, 0, ring_comm, MPI_STATUS_IGNORE);
      buffer.resize(recv_size);
      if (recv_size > 0) {
        MPI_Recv(buffer.data(), recv_size, MPI_INT, prev_rank, 1, ring_comm, MPI_STATUS_IGNORE);
      }

      if (rank == recipient) {
        GetOutput() = buffer;
      }
    }

    MPI_Barrier(ring_comm);
  }

  MPI_Comm_free(&ring_comm);

  return true;
}

bool KondrashovaVRingTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kondrashova_v_ring_topology