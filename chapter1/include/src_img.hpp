#include "opencv2/imgcodecs.hpp"

uint8_t filter_cpu(const uint8_t *in, const int row, const int col, const int ext, const float *kernel,
                   const int src_row, const int src_col);
void gaussianBlur_cpu(const uint8_t *in, const int row, const int col, const int ext, const float *kernel, uint8_t *out);

cv::Mat get_img_gray_mat(const std::string fileName);