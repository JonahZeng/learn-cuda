#include "src_img.hpp"
#include <string>
#include <iostream>

/******************************************
 * @param in: input image
 * @param row: input image height
 * @param col: input image width
 * @param ext: extend width on origin image
 * @param kernel: filter kernel
 * @param src_row: current pix y pos on input image
 * @param src_col: current pix x pos on input image
 * @return output value for current pix
 ******************************************/
uint8_t filter_cpu(const uint8_t *in, const int row, const int col, const int ext, const float *kernel,
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

/******************************************
 * @param in: input image
 * @param row: input image height
 * @param col: input image width
 * @param ext: extend width on origin image
 * @param kernel: filter kernel
 * @param out: output image
 ******************************************/
void gaussianBlur_cpu(const uint8_t *in, const int row, const int col, const int ext, const float *kernel, uint8_t *out)
{
	if (!in || !out || row == 0 || col == 0)
	{
		std::cerr << "input illeagal" << std::endl;
		return;
	}
	const int dst_width = col - 2 * ext;

	for (int src_row = ext, dst_row = 0; src_row < (row - ext); ++src_row, ++dst_row)
	{
		for (int src_col = ext, dst_col = 0; src_col < (col - ext); ++src_col, ++dst_col)
		{
			out[dst_row * dst_width + dst_col] = filter_cpu(in, row, col, ext, kernel, src_row, src_col);
		}
	}
}

cv::Mat get_img_gray_mat(const std::string fileName)
{
	cv::Mat img = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
	int rows = img.rows;
	int cols = img.cols;
	int channels = img.channels();
	std::cout << "input image rows: " << rows << " cols: " << cols << " channels: " << channels << std::endl;
	return img;
}