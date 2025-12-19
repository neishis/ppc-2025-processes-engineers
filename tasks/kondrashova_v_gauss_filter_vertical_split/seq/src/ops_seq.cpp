#include "kondrashova_v_gauss_filter_vertical_split/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>
#include "util/include/util.hpp"
#include "kondrashova_v_gauss_filter_vertical_split/common/include/common.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

const int KondrashovaVGaussFilterVerticalSplitSEQ::kGaussKernel[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};


uint8_t KondrashovaVGaussFilterVerticalSplitSEQ::ApplyGaussToPixel(
    const std::vector<uint8_t>& pixels, int width, int height,
    int channels, int x, int y, int channel) const {
  int sum = 0;
  
  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int px = std::clamp(x + kx, 0, width - 1);
      int py = std::clamp(y + ky, 0, height - 1);
      
      int idx = (py * width + px) * channels + channel;
      sum += pixels[idx] * kGaussKernel[ky + 1][kx + 1];
    }
  }
  
  return static_cast<uint8_t>(std::clamp(sum / kGaussKernelSum, 0, 255));
}

KondrashovaVGaussFilterVerticalSplitSEQ::KondrashovaVGaussFilterVerticalSplitSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KondrashovaVGaussFilterVerticalSplitSEQ::ValidationImpl() {
  const auto& input = GetInput();
  
  if (input.width < 3 || input.height < 3) {
    return false;
  }
  
  if (input.channels < 1 || input.channels > 4) {
    return false;
  }
  
  size_t expected_size = static_cast<size_t>(input.width * input.height * input.channels);
  return input.pixels.size() == expected_size;
}

bool KondrashovaVGaussFilterVerticalSplitSEQ::PreProcessingImpl() {
  const auto& input = GetInput();
  auto& output = GetOutput();
  
  output.width = input.width;
  output.height = input.height;
  output.channels = input.channels;
  output.pixels.resize(input.pixels.size());
  
  return true;
}

bool KondrashovaVGaussFilterVerticalSplitSEQ::RunImpl() {
  const auto& input = GetInput();
  auto& output = GetOutput();
  
  int width = input.width;
  int height = input.height;
  int channels = input.channels;
  
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        int idx = (y * width + x) * channels + c;
        output.pixels[idx] = ApplyGaussToPixel(input.pixels, width, height, channels, x, y, c);
      }
    }
  }
  
  return true;
}

bool KondrashovaVGaussFilterVerticalSplitSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kondrashova_v_gauss_filter_vertical_split