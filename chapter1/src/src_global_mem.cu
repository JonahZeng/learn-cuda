#include "src_gpu_img.hpp"

__device__ uint8_t filter_gpu(const uint8_t *in, const int row, const int col, const int ext, const float *kernel,
                              const int src_row, const int src_col)
{
    float res = 0.0f;
    int k_pos = 0;
    for (int k = -ext; k <= ext; ++k)
    {
        for (int j = -ext; j <= ext; ++j)
        {
            res += kernel[k_pos] * in[(src_row + k) * col + src_col + j];
            ++k_pos;
        }
    }
    int res_i = std::roundf(res);
    res_i = (res_i < 0) ? 0 : ((res_i > 255) ? 255 : res_i);

    return static_cast<uint8_t>(res_i);
}

__global__ void gaussianBlur_gpu(const uint8_t *in, const int row, const int col, const int ext, const float *kernel, uint8_t *out)
{
    int x_pos = blockDim.x * blockIdx.x + threadIdx.x;
    int y_pos = blockDim.y * blockIdx.y + threadIdx.y;
    const int dst_width = col - 2 * ext;
    if (x_pos < col - 2 * ext && y_pos < row - 2 * ext)
    {
        int src_row = y_pos + ext;
        int src_col = x_pos + ext;
        out[y_pos * dst_width + x_pos] = filter_gpu(in, row, col, ext, kernel, src_row, src_col);
    }
}