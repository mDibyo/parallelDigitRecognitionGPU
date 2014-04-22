#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include "utils.h"

float calc_min_dist(float *gpu_image, int i_width, int i_height, float* gpu_temp, int t_width) {

	float least_distance = UINT_MAX;

	if (t_width == 4096) {

		int threads_per_block = 512;
		int blocks_per_grid = 65564;

		int trans_height = i_height - t_width + 1;
		int trans_width = i_width - t_width + 1;
		int num_translations = trans_width * trans_height;

		float new_distance;

		size_t result_size = num_translations*sizeof(float);
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


		CUDA_SAFE_CALL(cudaFree(gpu_result));
		CUDA_SAFE_CALL(cudaFree(gpu_test));

		free(result);
		free(test);

	}

	return least_distance;
	
}