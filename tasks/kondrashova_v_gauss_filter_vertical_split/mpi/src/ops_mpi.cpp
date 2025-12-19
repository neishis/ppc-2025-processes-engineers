#include "kondrashova_v_gauss_filter_vertical_split/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "kondrashova_v_gauss_filter_vertical_split/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

const int KondrashovaVGaussFilterVerticalSplitMPI::kGaussKernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

KondrashovaVGaussFilterVerticalSplitMPI::KondrashovaVGaussFilterVerticalSplitMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

uint8_t KondrashovaVGaussFilterVerticalSplitMPI::ApplyGaussToLocalPixel(const std::vector<uint8_t> &local_data,
                                                                        int local_width, int height, int channels,
                                                                        int x, int y, int channel) const {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int px = std::clamp(x + kx, 0, local_width - 1);
      int py = std::clamp(y + ky, 0, height - 1);

      int idx = (py * local_width + px) * channels + channel;
      sum += local_data[idx] * kGaussKernel[ky + 1][kx + 1];
    }
  }

  return static_cast<uint8_t>(std::clamp(sum / kGaussKernelSum, 0, 255));
}

bool KondrashovaVGaussFilterVerticalSplitMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();

    if (input.width < 3 || input.height < 3) {
      return false;
    }
    if (input.channels < 1 || input.channels > 4) {
      return false;
    }

    size_t expected_size = static_cast<size_t>(input.width * input.height * input.channels);
    return input.pixels.size() == expected_size;
  }
  return true;
}

bool KondrashovaVGaussFilterVerticalSplitMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();
    auto &output = GetOutput();

    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.pixels.resize(input.pixels.size());
  }
  return true;
}

void KondrashovaVGaussFilterVerticalSplitMPI::BroadcastImageDimensions(int &width, int &height, int &channels) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    width = GetInput().width;
    height = GetInput().height;
    channels = GetInput().channels;
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void KondrashovaVGaussFilterVerticalSplitMPI::CalculateColumnDistribution(int width, int size,
                                                                          std::vector<int> &col_counts,
                                                                          std::vector<int> &col_offsets) {
  int base_cols = width / size;
  int extra_cols = width % size;

  col_counts.resize(size);
  col_offsets.resize(size);

  for (int i = 0; i < size; ++i) {
    col_counts[i] = base_cols + (i < extra_cols ? 1 : 0);
    col_offsets[i] = (i == 0) ? 0 : col_offsets[i - 1] + col_counts[i - 1];
  }
}

void KondrashovaVGaussFilterVerticalSplitMPI::DistributeImageData(int rank, int size, int width, int height,
                                                                  int channels, const std::vector<int> &col_counts,
                                                                  const std::vector<int> &col_offsets,
                                                                  std::vector<uint8_t> &local_data, int extended_cols) {
  local_data.resize(extended_cols * height * channels);

  if (rank == 0) {
    const auto &input_pixels = GetInput().pixels;

    for (int p = 0; p < size; ++p) {
      int p_start = std::max(0, col_offsets[p] - 1);
      int p_end = std::min(width, col_offsets[p] + col_counts[p] + 1);
      int p_cols = p_end - p_start;

      std::vector<uint8_t> send_data(p_cols * height * channels);

      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < p_cols; ++x) {
          for (int c = 0; c < channels; ++c) {
            int src_idx = (y * width + (p_start + x)) * channels + c;
            int dst_idx = (y * p_cols + x) * channels + c;
            send_data[dst_idx] = input_pixels[src_idx];
          }
        }
      }

      if (p == 0) {
        local_data = send_data;
      } else {
        MPI_Send(send_data.data(), static_cast<int>(send_data.size()), MPI_UINT8_T, p, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    MPI_Status status;
    MPI_Recv(local_data.data(), static_cast<int>(local_data.size()), MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, &status);
  }
}

void KondrashovaVGaussFilterVerticalSplitMPI::ApplyGaussFilterToLocalData(const std::vector<uint8_t> &local_data,
                                                                          std::vector<uint8_t> &local_result,
                                                                          int extended_cols, int local_cols, int height,
                                                                          int channels, int offset_in_extended) {
  local_result.resize(local_cols * height * channels);

  for (int y = 0; y < height; ++y) {
    for (int lx = 0; lx < local_cols; ++lx) {
      int x = offset_in_extended + lx;
      for (int c = 0; c < channels; ++c) {
        int result_idx = (y * local_cols + lx) * channels + c;
        local_result[result_idx] = ApplyGaussToLocalPixel(local_data, extended_cols, height, channels, x, y, c);
      }
    }
  }
}

void KondrashovaVGaussFilterVerticalSplitMPI::GatherResults(int rank, int size, int width, int height, int channels,
                                                            const std::vector<int> &col_counts,
                                                            const std::vector<int> &col_offsets, int local_start_col,
                                                            int local_cols, const std::vector<uint8_t> &local_result) {
  if (rank == 0) {
    auto &output_pixels = GetOutput().pixels;

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < local_cols; ++x) {
        for (int c = 0; c < channels; ++c) {
          int src_idx = (y * local_cols + x) * channels + c;
          int dst_idx = (y * width + (local_start_col + x)) * channels + c;
          output_pixels[dst_idx] = local_result[src_idx];
        }
      }
    }

    for (int p = 1; p < size; ++p) {
      int p_cols = col_counts[p];
      std::vector<uint8_t> recv_data(p_cols * height * channels);
      MPI_Status status;
      MPI_Recv(recv_data.data(), static_cast<int>(recv_data.size()), MPI_UINT8_T, p, 1, MPI_COMM_WORLD, &status);

      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < p_cols; ++x) {
          for (int c = 0; c < channels; ++c) {
            int src_idx = (y * p_cols + x) * channels + c;
            int dst_idx = (y * width + (col_offsets[p] + x)) * channels + c;
            output_pixels[dst_idx] = recv_data[src_idx];
          }
        }
      }
    }
  } else {
    MPI_Send(local_result.data(), static_cast<int>(local_result.size()), MPI_UINT8_T, 0, 1, MPI_COMM_WORLD);
  }
}

void KondrashovaVGaussFilterVerticalSplitMPI::BroadcastResultToAllProcesses(int width, int height, int channels) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    auto &output = GetOutput();
    output.width = width;
    output.height = height;
    output.channels = channels;
    output.pixels.resize(static_cast<size_t>(width * height * channels));
  }

  MPI_Bcast(GetOutput().pixels.data(), static_cast<int>(GetOutput().pixels.size()), MPI_UINT8_T, 0, MPI_COMM_WORLD);
}

bool KondrashovaVGaussFilterVerticalSplitMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int width = 0;
  int height = 0;
  int channels = 0;
  BroadcastImageDimensions(width, height, channels);

  std::vector<int> col_counts;
  std::vector<int> col_offsets;
  CalculateColumnDistribution(width, size, col_counts, col_offsets);

  int local_start_col = col_offsets[rank];
  int local_cols = col_counts[rank];

  int extended_start = std::max(0, local_start_col - 1);
  int extended_end = std::min(width, local_start_col + local_cols + 1);
  int extended_cols = extended_end - extended_start;
  int offset_in_extended = local_start_col - extended_start;

  std::vector<uint8_t> local_data;
  DistributeImageData(rank, size, width, height, channels, col_counts, col_offsets, local_data, extended_cols);

  std::vector<uint8_t> local_result;
  ApplyGaussFilterToLocalData(local_data, local_result, extended_cols, local_cols, height, channels,
                              offset_in_extended);

  GatherResults(rank, size, width, height, channels, col_counts, col_offsets, local_start_col, local_cols,
                local_result);

  BroadcastResultToAllProcesses(width, height, channels);

  return true;
}

bool KondrashovaVGaussFilterVerticalSplitMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kondrashova_v_gauss_filter_vertical_split
