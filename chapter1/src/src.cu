#include <iostream>
#include <cstdio>
#include "src_img.hpp"
#include "src_gpu_img.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
// #include "opencv2/opencv.hpp"
#include <chrono>

void error_proc(uint8_t *gpu_data_in, uint8_t *gpu_data_out, float *gpu_kernelData)
{
	std::cerr << "alloc device memory fail" << std::endl;
	if (gpu_data_in)
	{
		cudaFree(gpu_data_in);
	}
	if (gpu_data_out)
	{
		cudaFree(gpu_data_out);
	}
	if (gpu_kernelData)
	{
		cudaFree(gpu_kernelData);
	}
	cudaDeviceReset();
}

void run_cpu(std::string input_fn, std::string output_fn)
{
	std::cout << "-------------------- running on cpu ----------------------" << std::endl;
	//-------------------------cpu------------------------------------
	// read img convert to gray
	cv::Mat img_in = get_img_gray_mat(input_fn);
	auto originSize = img_in.size();
	// reflect extend border
	int ext = 3;
	cv::copyMakeBorder(img_in, img_in, ext, ext, ext, ext, cv::BORDER_REFLECT_101);
	const uint8_t *data_in = img_in.data;
	auto extSize = img_in.size();
	// create output mat
	cv::Mat img_out = cv::Mat::zeros(originSize, CV_8UC1);
	uint8_t *data_out = img_out.data;
	// get gaussian 2d kernel
	int kernelSize = 7;
	double sigma = 1.0;
	cv::Mat kernel = cv::getGaussianKernel(kernelSize, sigma, CV_32F);
	cv::Mat kernel2D = kernel * kernel.t();
	std::cout << "gaussian kernel:" << kernel2D << std::endl
			  << "kernel sum:" << cv::sum(kernel2D) << std::endl;
	float *kernelData = reinterpret_cast<float *>(kernel2D.data);
	// run gaussian filter on cpu and calc time cost(millisecond)
	auto time_start = std::chrono::high_resolution_clock::now();
	gaussianBlur_cpu(data_in, img_in.rows, img_in.cols, ext, kernelData, data_out);
	auto time_end = std::chrono::high_resolution_clock::now();
	auto time_cost = time_end - time_start;
	std::cout << "time cost(ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(time_cost).count() << std::endl;
	cv::imwrite(output_fn, img_out);
}

void run_gpu_global(std::string input_fn, std::string output_fn)
{
	std::cout << "-------------------- running on gpu with global memory ----------------------" << std::endl;
	cv::Mat img_in = get_img_gray_mat(input_fn);
	auto originSize = img_in.size();
	// reflect extend border
	// cv::cuda::GpuMat gpu_img_in;
	// gpu_img_in.upload(img_in);
	int ext = 3;

	cv::copyMakeBorder(img_in, img_in, ext, ext, ext, ext, cv::BORDER_REFLECT_101);
	const uint8_t *data_in = img_in.data;
	auto extSize = img_in.size();
	// create output mat
	cv::Mat img_out = cv::Mat::zeros(originSize, CV_8UC1);
	uint8_t *data_out = img_out.data;
	// get gaussian 2d kernel
	int kernelSize = 7;
	double sigma = 1.0;
	cv::Mat kernel = cv::getGaussianKernel(kernelSize, sigma, CV_32F);
	cv::Mat kernel2D = kernel * kernel.t();
	std::cout << "gaussian kernel:" << kernel2D << std::endl
			  << "kernel sum:" << cv::sum(kernel2D) << std::endl;
	float *kernelData = reinterpret_cast<float *>(kernel2D.data);
	//-------------------------gpu-------------------------------------
	// 7 TPC 14 SM 1792 CORE
	// 128 cuda core per SM
	// 192KB L1 CACHE per SM
	// 4MB L2 CACHE
	// 48KB shared memory per BLOCK
	int device = 0;
	cudaGetDevice(&device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	std::cout << "multiProcessorCount:" << prop.multiProcessorCount << std::endl;
	std::cout << "maxBlocksPerMultiProcessor:" << prop.maxBlocksPerMultiProcessor << std::endl;
	std::cout << "maxThreadsPerBlock:" << prop.maxThreadsPerBlock << std::endl;
	int sharedMemSize = 0;
	cudaDeviceGetAttribute(&sharedMemSize, cudaDevAttrMaxSharedMemoryPerBlock, device);
	std::cout << "max shared memeory size(Byte):" << sharedMemSize << std::endl;

	auto gpu_allocmem_time_start = std::chrono::high_resolution_clock::now();
	uint8_t *gpu_data_in = nullptr;
	uint8_t *gpu_data_out = nullptr;
	float *gpu_kernelData = nullptr;
	cudaError_t err = cudaMalloc(&gpu_data_in, sizeof(uint8_t) * extSize.width * extSize.height);
	if (err != cudaSuccess)
	{
		error_proc(gpu_data_in, gpu_data_out, gpu_kernelData);
		std::exit(EXIT_FAILURE);
	}
	cudaMemcpy(gpu_data_in, data_in, sizeof(uint8_t) * extSize.width * extSize.height, cudaMemcpyHostToDevice);

	err = cudaMalloc(&gpu_data_out, sizeof(uint8_t) * originSize.width * originSize.height);
	if (err != cudaSuccess)
	{
		error_proc(gpu_data_in, gpu_data_out, gpu_kernelData);
		std::exit(EXIT_FAILURE);
	}

	err = cudaMalloc(&gpu_kernelData, sizeof(float) * kernelSize * kernelSize);
	if (err != cudaSuccess)
	{
		error_proc(gpu_data_in, gpu_data_out, gpu_kernelData);
		std::exit(EXIT_FAILURE);
	}
	cudaMemcpy(gpu_kernelData, kernelData, sizeof(float) * kernelSize * kernelSize, cudaMemcpyHostToDevice);

	dim3 myBlockDim(8, 8);
	dim3 myGridDim((originSize.width + 7) / 8, (originSize.height + 7) / 8);
	auto gpu_time_start = std::chrono::high_resolution_clock::now();
	gaussianBlur_gpu<<<myGridDim, myBlockDim>>>(gpu_data_in, img_in.rows, img_in.cols, ext, gpu_kernelData, gpu_data_out);
	cudaDeviceSynchronize();
	auto gpu_time_end = std::chrono::high_resolution_clock::now();
	auto gpu_time_cost = gpu_time_end - gpu_time_start;
	std::cout << "[global mem]gpu kernel run time cost(ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_time_cost).count() << std::endl;
	cudaMemcpy(data_out, gpu_data_out, sizeof(uint8_t) * originSize.width * originSize.height, cudaMemcpyDeviceToHost);
	gpu_time_end = std::chrono::high_resolution_clock::now();
	gpu_time_cost = gpu_time_end - gpu_allocmem_time_start;
	std::cout << "[global mem]gpu total run time cost(ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_time_cost).count() << std::endl;
	cv::imwrite(output_fn, img_out);
	cudaFree(gpu_data_in);
	cudaFree(gpu_data_out);
	cudaFree(gpu_kernelData);
	cudaDeviceReset();
}

void run_gpu_shared(std::string input_fn, std::string output_fn)
{
	std::cout << "-------------------- running on gpu with shared memory ----------------------" << std::endl;
	cv::Mat img_in = get_img_gray_mat(input_fn);
	auto originSize = img_in.size();
	// reflect extend border
	int ext = 3;
	cv::copyMakeBorder(img_in, img_in, ext, ext, ext, ext, cv::BORDER_REFLECT_101);
	const uint8_t *data_in = img_in.data;
	auto extSize = img_in.size();
	// create output mat
	cv::Mat img_out = cv::Mat::zeros(originSize, CV_8UC1);
	uint8_t *data_out = img_out.data;
	// get gaussian 2d kernel
	int kernelSize = 7;
	double sigma = 1.0;
	cv::Mat kernel = cv::getGaussianKernel(kernelSize, sigma, CV_32F);
	cv::Mat kernel2D = kernel * kernel.t();
	std::cout << "gaussian kernel:" << kernel2D << std::endl
			  << "kernel sum:" << cv::sum(kernel2D) << std::endl;
	float *kernelData = reinterpret_cast<float *>(kernel2D.data);
	//-------------------------gpu-------------------------------------
	// 7 TPC 14 SM 1792 CORE
	// 128 cuda core per SM
	// 192KB L1 CACHE per SM
	// 4MB L2 CACHE
	// 48KB shared memory per BLOCK
	int device = 0;
	cudaGetDevice(&device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	std::cout << "multiProcessorCount:" << prop.multiProcessorCount << std::endl;
	std::cout << "maxBlocksPerMultiProcessor:" << prop.maxBlocksPerMultiProcessor << std::endl;
	std::cout << "maxThreadsPerBlock:" << prop.maxThreadsPerBlock << std::endl;
	int sharedMemSize = 0;
	cudaDeviceGetAttribute(&sharedMemSize, cudaDevAttrMaxSharedMemoryPerBlock, device);
	std::cout << "max shared memeory size(Byte):" << sharedMemSize << std::endl;

	auto gpu_allocmem_time_start = std::chrono::high_resolution_clock::now();
	uint8_t *gpu_data_in = nullptr;
	uint8_t *gpu_data_out = nullptr;
	float *gpu_kernelData = nullptr;
	cudaError_t err = cudaMalloc(&gpu_data_in, sizeof(uint8_t) * extSize.width * extSize.height);
	if (err != cudaSuccess)
	{
		error_proc(gpu_data_in, gpu_data_out, gpu_kernelData);
		std::exit(EXIT_FAILURE);
	}
	cudaMemcpy(gpu_data_in, data_in, sizeof(uint8_t) * extSize.width * extSize.height, cudaMemcpyHostToDevice);

	err = cudaMalloc(&gpu_data_out, sizeof(uint8_t) * originSize.width * originSize.height);
	if (err != cudaSuccess)
	{
		error_proc(gpu_data_in, gpu_data_out, gpu_kernelData);
		std::exit(EXIT_FAILURE);
	}

	err = cudaMalloc(&gpu_kernelData, sizeof(float) * kernelSize * kernelSize);
	if (err != cudaSuccess)
	{
		error_proc(gpu_data_in, gpu_data_out, gpu_kernelData);
		std::exit(EXIT_FAILURE);
	}
	cudaMemcpy(gpu_kernelData, kernelData, sizeof(float) * kernelSize * kernelSize, cudaMemcpyHostToDevice);

	dim3 myBlockDim(8, 8);
	dim3 myGridDim((originSize.width + 7) / 8, (originSize.height + 7) / 8);
	auto gpu_time_start = std::chrono::high_resolution_clock::now();
	gaussianBlur_shared_mem_gpu<<<myGridDim, myBlockDim>>>(gpu_data_in, img_in.rows, img_in.cols, ext, gpu_kernelData, gpu_data_out);
	cudaDeviceSynchronize();
	auto gpu_time_end = std::chrono::high_resolution_clock::now();
	auto gpu_time_cost = gpu_time_end - gpu_time_start;
	std::cout << "[shared mem]gpu kernel run time cost(ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_time_cost).count() << std::endl;
	cudaMemcpy(data_out, gpu_data_out, sizeof(uint8_t) * originSize.width * originSize.height, cudaMemcpyDeviceToHost);
	gpu_time_end = std::chrono::high_resolution_clock::now();
	gpu_time_cost = gpu_time_end - gpu_allocmem_time_start;
	std::cout << "[shared mem]gpu total run time cost(ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_time_cost).count() << std::endl;
	cv::imwrite(output_fn, img_out);
	cudaFree(gpu_data_in);
	cudaFree(gpu_data_out);
	cudaFree(gpu_kernelData);
	cudaDeviceReset();
}

void run_no_cudaMalloc_global(std::string input_fn, std::string output_fn)
{
	std::cout << "-------------------- running on gpu with cpu memory ----------------------" << std::endl;
	cv::Mat img_in = get_img_gray_mat(input_fn);
	auto originSize = img_in.size();
	// reflect extend border
	int ext = 3;
	cv::copyMakeBorder(img_in, img_in, ext, ext, ext, ext, cv::BORDER_REFLECT_101);
	const uint8_t *data_in = img_in.data;
	auto extSize = img_in.size();

	int kernelSize = 7;
	double sigma = 1.0;
	cv::Mat kernel = cv::getGaussianKernel(kernelSize, sigma, CV_32F);
	cv::Mat kernel2D = kernel * kernel.t();
	std::cout << "gaussian kernel:" << kernel2D << std::endl
			  << "kernel sum:" << cv::sum(kernel2D) << std::endl;
	float *kernelData = reinterpret_cast<float *>(kernel2D.data);
	//-------------------------gpu-------------------------------------
	// 7 TPC 14 SM 1792 CORE
	// 128 cuda core per SM
	// 192KB L1 CACHE per SM
	// 4MB L2 CACHE
	// 48KB shared memory per BLOCK
	int device = 0;
	cudaGetDevice(&device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	std::cout << "multiProcessorCount:" << prop.multiProcessorCount << std::endl;
	std::cout << "maxBlocksPerMultiProcessor:" << prop.maxBlocksPerMultiProcessor << std::endl;
	std::cout << "maxThreadsPerBlock:" << prop.maxThreadsPerBlock << std::endl;
	int sharedMemSize = 0;
	cudaDeviceGetAttribute(&sharedMemSize, cudaDevAttrMaxSharedMemoryPerBlock, device);
	std::cout << "max shared memeory size(Byte):" << sharedMemSize << std::endl;

	auto gpu_allocmem_time_start = std::chrono::high_resolution_clock::now();

	uint8_t *uni_data_in = nullptr;
	float *uni_kernelData = nullptr;
	uint8_t *uni_data_out = nullptr;
	cudaMallocManaged(&uni_data_in, sizeof(uint8_t) * extSize.width * extSize.height);
	cudaMallocManaged(&uni_kernelData, sizeof(float) * kernelSize * kernelSize);
	cudaMallocManaged(&uni_data_out, sizeof(uint8_t) * originSize.width * originSize.height);
	memcpy(uni_data_in, data_in, sizeof(uint8_t) * extSize.width * extSize.height);
	memcpy(uni_kernelData, kernelData, sizeof(float) * kernelSize * kernelSize);

	dim3 myBlockDim(8, 8);
	dim3 myGridDim((originSize.width + 7) / 8, (originSize.height + 7) / 8);
	auto gpu_time_start = std::chrono::high_resolution_clock::now();
	gaussianBlur_gpu<<<myGridDim, myBlockDim>>>(uni_data_in, img_in.rows, img_in.cols, ext, uni_kernelData, uni_data_out);
	cudaDeviceSynchronize();
	auto gpu_time_end = std::chrono::high_resolution_clock::now();
	auto gpu_time_cost = gpu_time_end - gpu_time_start;
	std::cout << "[global mem]gpu kernel run time cost(ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_time_cost).count() << std::endl;
	gpu_time_end = std::chrono::high_resolution_clock::now();
	gpu_time_cost = gpu_time_end - gpu_allocmem_time_start;
	std::cout << "[global mem]gpu total run time cost(ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_time_cost).count() << std::endl;
	cv::Mat img_out(originSize.height, originSize.width, CV_8UC1, uni_data_out);
	cv::imwrite(output_fn, img_out);
	cudaFree(uni_data_in);
	cudaFree(uni_data_out);
	cudaFree(uni_kernelData);
	cudaDeviceReset();
}

int main(void)
{
	run_cpu("../../wallhaven-13zl8v.png", "./gaussian_cpu_res.jpg");
	run_gpu_global("../../wallhaven-13zl8v.png", "./gaussian_gpu_global.jpg");
	run_gpu_shared("../../wallhaven-13zl8v.png", "./gaussian_gpu_shared.jpg");
	run_no_cudaMalloc_global("../../wallhaven-13zl8v.png", "./gaussian_no_cudaMalloc_global.jpg");
	return EXIT_SUCCESS;
}
