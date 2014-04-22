#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include "utils.h"

__global__ void distance2048NormalKernel(float* gpuImage, float* gpuTemp, float* gpuResults,
																				 int offX, int offY, int iWidth) {
	int blockIndexX = blockIdx.x / 4;
	offy += blockIdx.x % 4;
	if (offY + 512*blockIdx.x + threadIdx.x < iWidth) {
		float distance
			= gpuTemp[2048*blockIdx.y + 512*blockIndexX + threadIdx.x]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIndexX + threadIdx.x];
		gpuResult[4194304*(blockIdx.x%4) + 2048*blockIdx.y + 512*blockIndexX + threadIdx.x]
			= distance * distance;
	}
	
}


__global__ void distance4096NormalKernel(float* gpuImage, float* gpuTemp, float* gpuResult, float* gpuTest,
																				 int offX, int offY, int iWidth) {
	float distance
		= gpuTemp[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x]
		- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIdx.x + threadIdx.x];
	gpuResult[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x] = distance * distance;
}

__global__ void distance4096ReversedKernel(float* gpuImage, float* gpuTemp, float* gpuResult, float* gpuTest,
																					 int offX, int offY, int iWidth, int tempSize) {
	float distance
		= gpuTemp[tempSize - 4096*blockIdx.y + 512*blockIdx.x + threadIdx.x]
		- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIdx.x + threadIdx.x];
	gpuResult[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x] = distance * distance;
}

__global__ void reduction4096Kernel(float* gpuResult, unsigned int tempSize, unsigned int level) {
	int resultIndex = 2*level*(blockIdx.x*blockDim.x + threadIdx.x);
	if ((resultIndex + level) < tempSize) {
		gpuResult[resultIndex] += gpuResult[resultIndex + level];
	}
}

float calc_min_dist(float *gpu_image, int i_width, int i_height,
										float* gpu_temp, int t_width) {

	float least_distance = FLT_MAX;
	float new_distance = least_distance;

	int trans_height = i_height - t_width + 1;
	int trans_width = i_width - t_width + 1;
	unsigned int temp_size = t_width * t_width;

	if (t_width == 2048) {

		size_t result_size = temp_size*sizeof(float);
		float* gpu_results;
		CUDA_SAFE_CALL(cudaMalloc(&gpu_results, result_size*4));

		int threads_per_block = 512;
		int blocks_per_grid = 65535;

		dim3 dim_threads_per_block(threads_per_block, 1, 1);
		dim3 dim_blocks_per_grid(16, 2048);

		for (int off_x = 0; off_x < trans_height; off_x ++) {
			for (int off_y = 0; off_y < trans_width; off_y += 4) {
				distance2048NormalKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");

				unsigned int level = 1;
				blocks_per_grid = 4 * 4 * 2048;
				while (level < (temp_size/4)) {
					dim3 dim_threads_per_block(threads_per_block, 1, 1);
					dim3 dim_blocks_per_grid(blocks_per_grid, 1);
					reduction2048SumKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
						(gpu_results, temp_size, level);
					cudaThreadSynchronize();
					CUT_CHECK_ERROR("");
					level *= 2;
					blocks_per_grid /= 2;
					if (blocks_per_grid == 0) {
						blocks_per_grid = 1;
					}
				}

			}

		}


	}	else if (t_width == 4096) {

		size_t result_size = temp_size*sizeof(float);
		float* result = (float *)malloc(result_size);
		if (result == NULL) {
			printf("Unable to allocate space for result!\n");
			exit(EXIT_FAILURE);
		}
		float* gpu_result;
		CUDA_SAFE_CALL(cudaMalloc(&gpu_result, result_size));
		CUDA_SAFE_CALL(cudaMemcpy(gpu_result, result, result_size,
															cudaMemcpyHostToDevice));

		size_t test_size = 100*sizeof(float);
		float* test = (float *)malloc(test_size);
		if (result == NULL) {
			printf("Unable to allocate space for test!\n");
			exit(EXIT_FAILURE);
		}
		for (int counter = 0; counter < 100; counter++) {
			test[counter] = -1000;
		}
		float* gpu_test;
		CUDA_SAFE_CALL(cudaMalloc(&gpu_test, test_size));
		CUDA_SAFE_CALL(cudaMemcpy(gpu_test, test, test_size,
															cudaMemcpyHostToDevice));

		int threads_per_block = 512;
		int blocks_per_grid = 65535;

		// [16, 4096]
		dim3 dim_threads_per_block(threads_per_block, 1, 1);
		dim3 dim_blocks_per_grid(8, 4096);

		for (int off_x = 0; off_x < trans_height; off_x++) {
			for (int off_y = 0; off_y < trans_width; off_y++) {
				distance4096NormalKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_image, gpu_temp, gpu_result, gpu_test, off_x , off_y, i_width);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");

				unsigned int level = 1;
				blocks_per_grid = 8 * 4096;
				while (level < temp_size) {
					// printf("%d level reduction with %d blocks\n", level, blocks_per_grid);
					dim3 dim_threads_per_block(threads_per_block, 1, 1);
					dim3 dim_blocks_per_grid(blocks_per_grid, 1);
					reduction4096Kernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
						(gpu_result, temp_size, level);
					cudaThreadSynchronize();
					CUT_CHECK_ERROR("");
					level *= 2;
					blocks_per_grid /= 2;
					if (blocks_per_grid == 0) {
						blocks_per_grid = 1;
					}
				}

				CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_result, sizeof(float),
																	cudaMemcpyDeviceToHost));
				if (new_distance < least_distance) {
					least_distance = new_distance;
				}

			}
		}

		for (int off_x = 0; off_x < trans_height; off_x++) {
			for (int off_y = 0; off_y < trans_width; off_y++) {
				distance4096ReversedKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_image, gpu_temp, gpu_result, gpu_test, off_x , off_y, i_width, temp_size);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");

				unsigned int level = 1;
				blocks_per_grid = 8 * 4096;
				while (level != temp_size) {
					// printf("%d level reduction with %d blocks\n", level, blocks_per_grid);
					dim3 dim_threads_per_block(threads_per_block, 1, 1);
					dim3 dim_blocks_per_grid(blocks_per_grid, 1);
					reduction4096Kernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
						(gpu_result, temp_size, level);
					cudaThreadSynchronize();
					CUT_CHECK_ERROR("");
					level *= 2;
					blocks_per_grid /= 2;
					if (blocks_per_grid == 0) {
						blocks_per_grid = 1;
					}
				}

				CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_result, sizeof(float),
																	cudaMemcpyDeviceToHost));
				if (new_distance < least_distance) {
					least_distance = new_distance;
				}

			}
		}
		CUDA_SAFE_CALL(cudaMemcpy(test, gpu_test, test_size,
																	cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(gpu_result));
		CUDA_SAFE_CALL(cudaFree(gpu_test));
		/*
		for (int counter = 0; counter < 100; counter++) {
			printf("%f\n", test[counter]);
		} */

		free(result);
		free(test);

	}

	return least_distance;

}