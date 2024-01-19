#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

__device__ uint8_t filter_gpu(const uint8_t *in, const int row, const int col, const int ext, const float *kernel,
                              const int src_row, const int src_col);

__global__ void gaussianBlur_gpu(const uint8_t *in, const int row, const int col, const int ext, const float *kernel, uint8_t *out);

__global__ void gaussianBlur_shared_mem_gpu(const uint8_t *in, const int row, const int col, const int ext, const float *kernel, uint8_t *out);