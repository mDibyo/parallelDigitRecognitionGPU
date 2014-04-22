#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include "utils.h"

__global__ distance4096Kernel(float* gpuImage, float* gpuTemp,
															int offX, int offY, int iWidth) {
	if (blockIdx.y < 4096) {
		float distance
			= gpuTemp[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x]
			- gpuImage[(iWidth+blockIdx.y)*offX + offY + 512*blockIdx.x + threadIdx.x];
		result[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x] = distance * distance;
	}
}

float calc_min_dist(float *gpu_image, int i_width, int i_height,
										float* gpu_temp, int t_width) {

	float least_distance = UINT_MAX;

	if (t_width == 4096) {



		int trans_height = i_height - t_width + 1;
		int trans_width = i_width - t_width + 1;
		int num_translations = trans_width * trans_height;
		int temp_size = t_width * t_width;

		float new_distance;

		size_t result_size = temp_size*sizeof(float);
		float* result = (float *)malloc(result_size);
		if (result == NULL) {
			printf("Unable to allocate space for result!\n");
			exit(EXIT_FAILURE);
		}
		float* gpu_result;
		CUDA_SAFE_CALL(cudaMalloc(&gpu_result, result_size));

		size_t test_size = 100*sizeof(float);
		float* test = (float *)malloc(test_size);
		if (result == NULL) {
			printf("Unable to allocate space for result!\n");
			exit(EXIT_FAILURE);
		}
		float* gpu_test;
		CUDA_SAFE_CALL(cudaMalloc(&gpu_result, test_size));

		int threads_per_block = 512;
		int blocks_per_grid = 65564;

		// [16, 4096]
		dim3 dim_threads_per_block(threads_per_block, 1, 1);
		dim3 dim_blocks_per_grid(8, 4096);
		for (int off_x = 0; off_x < trans_height; off_x++) {
			for (int off_y = 0; off_y < trans_width; off_y++) {
				distance4096Kernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_image, gpu_temp, off_x , off_y, i_width);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");

				int level = 1;
				blocks_per_grid = 8 * 4096;
				while (level != temp_size) {
					dim3 dim_threads_per_block(threads_per_block, 1, 1);
					dim3 dim_blocks_per_grid(blocks_per_grid, 1);
					reduction4096Kernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
						(gpu_result, level);
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

		CUDA_SAFE_CALL(cudaFree(gpu_result));
		CUDA_SAFE_CALL(cudaFree(gpu_test));

		free(result);
		free(test);

	}

	return least_distance;

}