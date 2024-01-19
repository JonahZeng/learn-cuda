#include "src_gpu_img.hpp"

__global__ void gaussianBlur_shared_mem_gpu(const uint8_t *in, const int row, const int col, const int ext, const float *kernel, uint8_t *out)
{
#define BLK_MAX_SIZE ((16 + 6) * (8 + 6))
#define KERNEL_MAX_SIZE 49
    __shared__ uint8_t blk_img_data[BLK_MAX_SIZE];
    __shared__ float blk_kernel[KERNEL_MAX_SIZE];

    int x_pos = blockDim.x * blockIdx.x + threadIdx.x;
    int y_pos = blockDim.y * blockIdx.y + threadIdx.y;
    const int dst_width = col - 2 * ext;
    if (x_pos < col - 2 * ext && y_pos < row - 2 * ext)
    {
        const int blockW_ext = blockDim.x + 2 * ext;
        const int blockH_ext = blockDim.y + 2 * ext;
        if (threadIdx.x == 0u && threadIdx.y == 0u)
        {
            for (int j = 0; j < blockH_ext; ++j)
            {
                for (int k = 0; k < blockW_ext; ++k)
                {
                    blk_img_data[j * blockW_ext + k] = in[(y_pos + j) * col + x_pos + k];
                }
            }

            for (int i = 0; i < (2 * ext + 1) * (2 * ext + 1); ++i)
            {
                blk_kernel[i] = kernel[i];
            }
        }
        // {
        //     const int blk_total_thread = blockDim.x * blockDim.y;
        //     int thread_t_cnt = threadIdx.y * blockDim.x + threadIdx.x;
        //     for (; thread_t_cnt < BLK_MAX_SIZE; thread_t_cnt += blk_total_thread)
        //     {
        //         int blk_row = thread_t_cnt / (16 + 6);
        //         int blk_col = thread_t_cnt - blk_row * (16 + 6);
        //         int src_row = blockIdx.y * blockDim.y + blk_row;
        //         int src_col = blockIdx.x * blockDim.x + blk_col;

        //         blk_img_data[thread_t_cnt] = in[src_row * col + src_col];
        //     }

        //     thread_t_cnt = threadIdx.y * blockDim.x + threadIdx.x;
        //     if (thread_t_cnt < KERNEL_MAX_SIZE)
        //     {
        //         blk_kernel[thread_t_cnt] = kernel[thread_t_cnt];
        //     }
        // }
        __syncthreads();

        float res = 0.0f;
        int k_pos = 0;
        for (int k = -ext; k <= ext; ++k)
        {
            for (int j = -ext; j <= ext; ++j)
            {
                res += blk_kernel[k_pos] * blk_img_data[(int(threadIdx.y + ext) + k) * blockW_ext + int(threadIdx.x + ext) + j];
                ++k_pos;
            }
        }
        int res_i = std::roundf(res);
        out[y_pos * dst_width + x_pos] = (res_i < 0) ? 0 : ((res_i > 255) ? 255 : res_i);
        __syncthreads();
    }
}