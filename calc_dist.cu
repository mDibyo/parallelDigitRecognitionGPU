#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include "utils.h"





////////////////////////////
/// <= 512 TEMPLATE SIZE ///
////////////////////////////

__global__ void distance512NormalKernel(float* gpuImage, float* gpuTemp, float* gpuResults, int offX,
																				int offY, int iWidth, int tWidth, int tempSize) {
	if ((offY + blockIdx.x + tWidth) <= iWidth) {
		float distance
			= gpuTemp[tWidth*blockIdx.y + threadIdx.x]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + blockIdx.x + threadIdx.x];
		gpuResults[tempSize*blockIdx.x + tWidth*blockIdx.y + threadIdx.x]
			= distance * distance;
	}
}

__global__ void distance512NormalTransposeKernel(float* gpuImage, float* gpuTemp, float* gpuResults, int offX,
																								 int offY, int iWidth, int tWidth, int tempSize) {
	if ((offY + blockIdx.x + tWidth) <= iWidth) {
		float distance
			= gpuTemp[tWidth*threadIdx.x + blockIdx.y]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + blockIdx.x + threadIdx.x];
		gpuResults[tempSize*blockIdx.x + tWidth*blockIdx.y + threadIdx.x]
			= distance * distance;
	}
}

__global__ void distance512NormalFlipKernel(float* gpuImage, float* gpuTemp, float* gpuResults, int offX,
																						int offY, int iWidth, int tWidth, int tempSize) {
	if ((offY + blockIdx.x + tWidth) <= iWidth) {
		float distance
			= gpuTemp[tWidth*(blockIdx.y+1) - threadIdx.x-1]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + blockIdx.x + threadIdx.x];
		gpuResults[tempSize*blockIdx.x + tWidth*blockIdx.y + threadIdx.x]
			= distance * distance;
	}
}

__global__ void distance512NormalTransposeFlipKernel(float* gpuImage, float* gpuTemp, float* gpuResults, int offX,
																										 int offY, int iWidth, int tWidth, int tempSize) {
	if ((offY + blockIdx.x + tWidth) <= iWidth) {
		float distance
			= gpuTemp[tWidth*(threadIdx.x+1) - blockIdx.y-1]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + blockIdx.x + threadIdx.x];
		gpuResults[tempSize*blockIdx.x + tWidth*blockIdx.y + threadIdx.x]
			= distance * distance;
	}
}

__global__ void distance512ReverseKernel(float* gpuImage, float* gpuTemp, float* gpuResults, int offX,
																				 int offY, int iWidth, int tWidth, int tempSize) {
	if ((offY + blockIdx.x + tWidth) <= iWidth) {
		float distance
			= gpuTemp[tempSize - tWidth*blockIdx.y - threadIdx.x - 1]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + blockIdx.x + threadIdx.x];
		gpuResults[tempSize*blockIdx.x + tWidth*blockIdx.y + threadIdx.x]
			= distance * distance;
	}
}

__global__ void distance512ReverseTransposeKernel(float* gpuImage, float* gpuTemp, float* gpuResults, int offX,
																									int offY, int iWidth, int tWidth, int tempSize) {
	if ((offY + blockIdx.x + tWidth) <= iWidth) {
		float distance
			= gpuTemp[tempSize - tWidth*threadIdx.x - blockIdx.y - 1]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + blockIdx.x + threadIdx.x];
		gpuResults[tempSize*blockIdx.x + tWidth*blockIdx.y + threadIdx.x]
			= distance * distance;
	}
}

__global__ void distance512ReverseFlipKernel(float* gpuImage, float* gpuTemp, float* gpuResults, int offX,
																						 int offY, int iWidth, int tWidth, int tempSize) {
	if ((offY + blockIdx.x + tWidth) <= iWidth) {
		float distance
			= gpuTemp[tempSize - tWidth*(blockIdx.y+1) + threadIdx.x]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + blockIdx.x + threadIdx.x];
		gpuResults[tempSize*blockIdx.x + tWidth*blockIdx.y + threadIdx.x]
			= distance * distance;
	}
}

__global__ void distance512ReverseTransposeFlipKernel(float* gpuImage, float* gpuTemp, float* gpuResults, int offX,
																											int offY, int iWidth, int tWidth, int tempSize) {
	if ((offY + blockIdx.x + tWidth) <= iWidth) {
		float distance
			= gpuTemp[tempSize - tWidth*(threadIdx.x+1) + blockIdx.y]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + blockIdx.x + threadIdx.x];
		gpuResults[tempSize*blockIdx.x + tWidth*blockIdx.y + threadIdx.x]
			= distance * distance;
	}
}

__global__ void transpose512Kernel(float* gpuTemp, int tWidth) {
	if (threadIdx.x > blockIdx.x) {
		float temp = gpuTemp[tWidth*blockIdx.x + threadIdx.x];
		gpuTemp[tWidth*blockIdx.x + threadIdx.x] = gpuTemp[tWidth*threadIdx.x + blockIdx.x];
		gpuTemp[tWidth*threadIdx.x + blockIdx.x] = temp;
	}
}

__global__ void reduction512SumKernel(float* gpuResults, unsigned int tempSize, int numResults,
																			unsigned int level) {
	unsigned int resultIndex = 2*level*(blockIdx.x*blockDim.x + threadIdx.x);
	if ((resultIndex + level) < (tempSize*numResults)) {
		gpuResults[resultIndex] += gpuResults[resultIndex + level];
	}
}

__global__ void reduction512MaxKernel(float* gpuResults, unsigned int tempSize, int numResults,
																			unsigned int level) {
	unsigned int resultIndex = 2*level*(blockIdx.x*blockDim.x + threadIdx.x);
	if ((resultIndex + level) < (tempSize*numResults)) {
		if (gpuResults[resultIndex + level] < gpuResults[resultIndex]) {
			gpuResults[resultIndex] = gpuResults[resultIndex + level];
		}
	}
}

float calc_min_dist_512(float* gpu_image, int i_width, int i_height,
												float* gpu_temp, int t_width) {

	float least_distance = FLT_MAX;
	float new_distance = least_distance;

	int trans_height = i_height - t_width + 1;
	int trans_width = i_width - t_width + 1;
	unsigned int temp_size = t_width * t_width;

	int blocks_per_grid = 65535;

	size_t result_size = temp_size*sizeof(float);
	int num_results = blocks_per_grid / t_width;
	if (num_results > (trans_width * trans_height)) {
		num_results = trans_height * trans_width;
	}
	float* gpu_results;
	CUDA_SAFE_CALL(cudaMalloc(&gpu_results, result_size*num_results));

	dim3 dim_blocks_per_grid(num_results, t_width);

	// normal
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += num_results) {
			distance512NormalKernel<<<dim_blocks_per_grid, t_width>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width, t_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = num_results * t_width;
			while (level < temp_size) {
				reduction512SumKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}
			
			while (level < (temp_size*num_results)) {
				reduction512MaxKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// normal flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += num_results) {
			distance512NormalFlipKernel<<<dim_blocks_per_grid, t_width>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width, t_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = num_results * t_width;
			// blocks_per_grid = 65535;
			while (level < temp_size) {
				reduction512SumKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}
			
			while (level < (temp_size*num_results)) {
				reduction512MaxKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += num_results) {
			distance512ReverseKernel<<<dim_blocks_per_grid, t_width>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width, t_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = num_results * t_width;
			// blocks_per_grid = 65535;
			while (level < temp_size) {
				reduction512SumKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}
			
			while (level < (temp_size*num_results)) {
				reduction512MaxKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += num_results) {
			distance512ReverseFlipKernel<<<dim_blocks_per_grid, t_width>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width, t_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = num_results * t_width;
			// blocks_per_grid = 65535;
			while (level < temp_size) {
				reduction512SumKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}
			
			while (level < (temp_size*num_results)) {
				reduction512MaxKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	transpose512Kernel<<<t_width, t_width>>>(gpu_temp, t_width);
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("");

	// normal
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += num_results) {
			distance512NormalKernel<<<dim_blocks_per_grid, t_width>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width, t_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = num_results * t_width;
			// blocks_per_grid = 65535;
			while (level < temp_size) {
				reduction512SumKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}
			
			while (level < (temp_size*num_results)) {
				reduction512MaxKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// normal flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += num_results) {
			distance512NormalFlipKernel<<<dim_blocks_per_grid, t_width>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width, t_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = num_results * t_width;
			// blocks_per_grid = 65535;
			while (level < temp_size) {
				reduction512SumKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}
			
			while (level < (temp_size*num_results)) {
				reduction512MaxKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += num_results) {
			distance512ReverseKernel<<<dim_blocks_per_grid, t_width>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width, t_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = num_results * t_width;
			// blocks_per_grid = 65535;
			while (level < temp_size) {
				reduction512SumKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}
			
			while (level < (temp_size*num_results)) {
				reduction512MaxKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += num_results) {
			distance512ReverseFlipKernel<<<dim_blocks_per_grid, t_width>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width, t_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = num_results * t_width;
			// blocks_per_grid = 65535;
			while (level < temp_size) {
				reduction512SumKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}
			
			while (level < (temp_size*num_results)) {
				reduction512MaxKernel<<<blocks_per_grid, t_width>>>
					(gpu_results, temp_size, num_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	CUDA_SAFE_CALL(cudaFree(gpu_results));

	return least_distance;

}





//////////////////////////
/// 1024 TEMPLATE SIZE ///
//////////////////////////

__global__ void distance1024NormalKernel(float* gpuImage, float* gpuTemp, float* gpuResults,
																				 int offX, int offY, int iWidth) {
	int blockIndexX = blockIdx.x / 16;
	offY += blockIdx.x % 16;
	if ((offY + 512*blockIndexX + 512) <= iWidth) {
		float distance
			= gpuTemp[1024*blockIdx.y + 512*blockIndexX + threadIdx.x]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIndexX + threadIdx.x];
		gpuResults[1048576*(blockIdx.x%16) + 1024*blockIdx.y + 512*blockIndexX + threadIdx.x]
			= distance * distance;
	}
}

__global__ void distance1024NormalFlipKernel(float* gpuImage, float* gpuTemp, float* gpuResults,
																				 		 int offX, int offY, int iWidth) {
	int blockIndexX = blockIdx.x / 16;
	offY += blockIdx.x % 16;
	if ((offY + 512*blockIndexX + 512) <= iWidth) {
		float distance
			= gpuTemp[1024*(blockIdx.y+1) - 512*blockIndexX - threadIdx.x - 1]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIndexX + threadIdx.x];
		gpuResults[1048576*(blockIdx.x%16) + 1024*blockIdx.y + 512*blockIndexX + threadIdx.x]
			= distance * distance;
	}
}

__global__ void distance1024ReverseKernel(float* gpuImage, float* gpuTemp, float* gpuResults,
																					int offX, int offY, int iWidth) {
	int blockIndexX = blockIdx.x / 16;
	offY += blockIdx.x % 16;
	if ((offY + 512*blockIndexX + 512) <= iWidth) {
		float distance
			= gpuTemp[1048576 - 1024*blockIdx.y - 512*blockIndexX - threadIdx.x - 1]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIndexX + threadIdx.x];
		gpuResults[1048576*(blockIdx.x%16) + 1024*blockIdx.y + 512*blockIndexX + threadIdx.x]
			= distance * distance;
	}
}

__global__ void distance1024ReverseFlipKernel(float* gpuImage, float* gpuTemp, float* gpuResults,
																							int offX, int offY, int iWidth) {
	int blockIndexX = blockIdx.x / 16;
	offY += blockIdx.x % 16;
	if ((offY + 512*blockIndexX + 512) <= iWidth) {
		float distance
			= gpuTemp[1048576 - 1024*(blockIdx.y+1) + 512*blockIndexX + threadIdx.x]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIndexX + threadIdx.x];
		gpuResults[1048576*(blockIdx.x%16) + 1024*blockIdx.y + 512*blockIndexX + threadIdx.x]
			= distance * distance;
	}
}

__global__ void transpose1024Kernel(float* gpuTemp) {
	if (512*blockIdx.x + threadIdx.x > blockIdx.y) {
		float temp = gpuTemp[1024*blockIdx.y + 512*blockIdx.x + threadIdx.x];
		gpuTemp[1024*blockIdx.y + 512*blockIdx.x + threadIdx.x]
			= gpuTemp[1024*(512*blockIdx.x + threadIdx.x) + blockIdx.y];
		gpuTemp[1024*(512*blockIdx.x + threadIdx.x) + blockIdx.y] = temp;
	}
}

__global__ void reduction1024SumKernel(float* gpuResults, unsigned int level) {
	int resultIndex = 2*level*(blockIdx.x*blockDim.x + threadIdx.x);
	if ((resultIndex + level) < 16777216) {
		gpuResults[resultIndex] += gpuResults[resultIndex + level];
	}
}

__global__ void reduction1024MaxKernel(float* gpuResults, unsigned int level) {
	unsigned int resultIndex = 2*level*(blockIdx.x*blockDim.x + threadIdx.x);
	if ((resultIndex + level) < 16777216) {
		if (gpuResults[resultIndex + level] < gpuResults[resultIndex]) {
			gpuResults[resultIndex] = gpuResults[resultIndex + level];
		}
	}
}

float calc_min_dist_1024(float* gpu_image, int i_width, int i_height, float* gpu_temp) {

	float least_distance = FLT_MAX;
	float new_distance = least_distance;

	int trans_height = i_height - 1024 + 1;
	int trans_width = i_width - 1024 + 1;
	unsigned int temp_size = 1024 * 1024;

	int threads_per_block = 512;
	int blocks_per_grid = 65535;

	size_t result_size = temp_size*sizeof(float);
	float* gpu_results;
	CUDA_SAFE_CALL(cudaMalloc(&gpu_results, result_size*16));

	dim3 dim_threads_per_block(threads_per_block, 1, 1);
	dim3 dim_blocks_per_grid(32, 1024);

	// normal
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 16) {
			distance1024NormalKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 16 * 2 * 1024;
			while (level < temp_size) {
				reduction1024SumKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			while (level < (temp_size*16)) {
				reduction1024MaxKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// normal flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 16) {
			distance1024NormalFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 16 * 2 * 1024;
			while (level < temp_size) {
				reduction1024SumKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			while (level < (temp_size*16)) {
				reduction1024MaxKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 16) {
			distance1024ReverseKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 16 * 2 * 1024;
			while (level < temp_size) {
				reduction1024SumKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			while (level < (temp_size*16)) {
				reduction1024MaxKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 16) {
			distance1024ReverseFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 16 * 2 * 1024;
			while (level < temp_size) {
				reduction1024SumKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			while (level < (temp_size*16)) {
				reduction1024MaxKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	{
		dim3 dim_blocks_per_grid(2, 1024);
		transpose1024Kernel<<<dim_blocks_per_grid, dim_threads_per_block>>>(gpu_temp);
		cudaThreadSynchronize();
		CUT_CHECK_ERROR("");
	}

	// normal
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 16) {
			distance1024NormalKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 16 * 2 * 1024;
			while (level < temp_size) {
				reduction1024SumKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			while (level < (temp_size*16)) {
				reduction1024MaxKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// normal flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 16) {
			distance1024NormalFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 16 * 2 * 1024;
			while (level < temp_size) {
				reduction1024SumKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			while (level < (temp_size*16)) {
				reduction1024MaxKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 16) {
			distance1024ReverseKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 16 * 2 * 1024;
			while (level < temp_size) {
				reduction1024SumKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			while (level < (temp_size*16)) {
				reduction1024MaxKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 16) {
			distance1024ReverseFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 16 * 2 * 1024;
			while (level < temp_size) {
				reduction1024SumKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			while (level < (temp_size*16)) {
				reduction1024MaxKernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	CUDA_SAFE_CALL(cudaFree(gpu_results));

	return least_distance;

}





//////////////////////////
/// 2048 TEMPLATE SIZE ///
//////////////////////////

__global__ void distance2048NormalKernel(float* gpuImage, float* gpuTemp, float* gpuResults,
																				 int offX, int offY, int iWidth) {
	int blockIndexX = blockIdx.x / 4;
	offY += blockIdx.x % 4;
	if ((offY + 512*blockIndexX + 512) <= iWidth) {
		float distance
			= gpuTemp[2048*blockIdx.y + 512*blockIndexX + threadIdx.x]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIndexX + threadIdx.x];
		gpuResults[4194304*(blockIdx.x%4) + 2048*blockIdx.y + 512*blockIndexX + threadIdx.x]
			= distance * distance;
	}	
}

__global__ void distance2048NormalFlipKernel(float* gpuImage, float* gpuTemp, float* gpuResults,
																						 int offX, int offY, int iWidth) {
	int blockIndexX = blockIdx.x / 4;
	offY += blockIdx.x % 4;
	if ((offY + 512*blockIndexX + 512) <= iWidth) {
		float distance
			= gpuTemp[2048*(blockIdx.y+1) - 512*blockIndexX - threadIdx.x - 1]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIndexX + threadIdx.x];
		gpuResults[4194304*(blockIdx.x%4) + 2048*blockIdx.y + 512*blockIndexX + threadIdx.x]
			= distance * distance;
	}	
}

__global__ void distance2048ReverseKernel(float* gpuImage, float* gpuTemp, float* gpuResults,
																					int offX, int offY, int iWidth) {
	int blockIndexX = blockIdx.x / 4;
	offY += blockIdx.x % 4;
	if ((offY + 512*blockIndexX + 512) < iWidth) {
		float distance
			= gpuTemp[4194304 - 2048*blockIdx.y - 512*blockIndexX - threadIdx.x - 1]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIndexX + threadIdx.x];
		gpuResults[4194304*(blockIdx.x%4) + 2048*blockIdx.y + 512*blockIndexX + threadIdx.x]
			= distance * distance;
	}
}

__global__ void distance2048ReverseFlipKernel(float* gpuImage, float* gpuTemp, float* gpuResults,
																							int offX, int offY, int iWidth) {
	int blockIndexX = blockIdx.x / 4;
	offY += blockIdx.x % 4;
	if ((offY + 512*blockIndexX + 512) < iWidth) {
		float distance
			= gpuTemp[4194304 - 2048*(blockIdx.y+1) + 512*blockIndexX + threadIdx.x]
			- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIndexX + threadIdx.x];
		gpuResults[4194304*(blockIdx.x%4) + 2048*blockIdx.y + 512*blockIndexX + threadIdx.x]
			= distance * distance;
	}
}

__global__ void transpose2048Kernel(float* gpuTemp) {
	if (512*blockIdx.x + threadIdx.x > blockIdx.y) {
		float temp = gpuTemp[2048*blockIdx.y + 512*blockIdx.x + threadIdx.x];
		gpuTemp[2048*blockIdx.y + 512*blockIdx.x + threadIdx.x]
			= gpuTemp[2048*(512*blockIdx.x + threadIdx.x) + blockIdx.y];
		gpuTemp[2048*(512*blockIdx.x + threadIdx.x) + blockIdx.y] = temp;
	}
}

__global__ void reduction2048SumKernel(float* gpuResults, unsigned int level) {
	int resultIndex = 2*level*(blockIdx.x*blockDim.x + threadIdx.x);
	if ((resultIndex + level) < 16777216) {
		gpuResults[resultIndex] += gpuResults[resultIndex + level];
	}
}

__global__ void reduction2048MaxKernel(float* gpuResults) {
	__shared__ float temp[2];
	if (gpuResults[(2*threadIdx.x)*4194304] < gpuResults[(2*threadIdx.x+1)*4194304]) {
		temp[threadIdx.x] = gpuResults[(2*threadIdx.x)*4194304];
	} else {
		temp[threadIdx.x] = gpuResults[(2*threadIdx.x+1)*4194304];
	}
	__syncthreads();
	if (temp[0] < temp[1]) {
		gpuResults[0] = temp[0];
	} else {
		gpuResults[0] = temp[1];
	}
}

float calc_min_dist_2048(float* gpu_image, int i_width, int i_height, float* gpu_temp) {

	float least_distance = FLT_MAX;
	float new_distance = least_distance;

	int trans_height = i_height - 2048 + 1;
	int trans_width = i_width - 2048 + 1;
	unsigned int temp_size = 2048 * 2048;

	int threads_per_block = 512;
	int blocks_per_grid = 65535;

	size_t result_size = temp_size*sizeof(float);
	float* gpu_results;
	CUDA_SAFE_CALL(cudaMalloc(&gpu_results, result_size*4));

	dim3 dim_threads_per_block(threads_per_block, 1, 1);
	dim3 dim_blocks_per_grid(16, 2048);

	// normal
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 4) {
			distance2048NormalKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 4 * 4 * 2048;
			while (level < temp_size) {
				dim3 dim_threads_per_block(threads_per_block, 1, 1);
				dim3 dim_blocks_per_grid(blocks_per_grid, 1);
				reduction2048SumKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			reduction2048MaxKernel<<<1, 2>>>(gpu_results);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// normal flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 4) {
			distance2048NormalFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 4 * 4 * 2048;
			while (level < temp_size) {
				dim3 dim_threads_per_block(threads_per_block, 1, 1);
				dim3 dim_blocks_per_grid(blocks_per_grid, 1);
				reduction2048SumKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			reduction2048MaxKernel<<<1, 2>>>(gpu_results);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 4) {
			distance2048ReverseKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 4 * 4 * 2048;
			while (level < temp_size) {
				dim3 dim_threads_per_block(threads_per_block, 1, 1);
				dim3 dim_blocks_per_grid(blocks_per_grid, 1);
				reduction2048SumKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			reduction2048MaxKernel<<<1, 2>>>(gpu_results);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 4) {
			distance2048ReverseFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 4 * 4 * 2048;
			while (level < temp_size) {
				dim3 dim_threads_per_block(threads_per_block, 1, 1);
				dim3 dim_blocks_per_grid(blocks_per_grid, 1);
				reduction2048SumKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			reduction2048MaxKernel<<<1, 2>>>(gpu_results);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	{
		dim3 dim_blocks_per_grid(4, 2048);
		transpose2048Kernel<<<dim_blocks_per_grid, dim_threads_per_block>>>(gpu_temp);
		cudaThreadSynchronize();
		CUT_CHECK_ERROR("");
	}

	// normal
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 4) {
			distance2048NormalKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 4 * 4 * 2048;
			while (level < temp_size) {
				dim3 dim_threads_per_block(threads_per_block, 1, 1);
				dim3 dim_blocks_per_grid(blocks_per_grid, 1);
				reduction2048SumKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			reduction2048MaxKernel<<<1, 2>>>(gpu_results);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// normal flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 4) {
			distance2048NormalFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 4 * 4 * 2048;
			while (level < temp_size) {
				dim3 dim_threads_per_block(threads_per_block, 1, 1);
				dim3 dim_blocks_per_grid(blocks_per_grid, 1);
				reduction2048SumKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			reduction2048MaxKernel<<<1, 2>>>(gpu_results);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 4) {
			distance2048ReverseKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 4 * 4 * 2048;
			while (level < temp_size) {
				dim3 dim_threads_per_block(threads_per_block, 1, 1);
				dim3 dim_blocks_per_grid(blocks_per_grid, 1);
				reduction2048SumKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			reduction2048MaxKernel<<<1, 2>>>(gpu_results);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	// reverse flip
	for (int off_x = 0; off_x < trans_height; off_x ++) {
		for (int off_y = 0; off_y < trans_width; off_y += 4) {
			distance2048ReverseFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_results, off_x, off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 4 * 4 * 2048;
			while (level < temp_size) {
				dim3 dim_threads_per_block(threads_per_block, 1, 1);
				dim3 dim_blocks_per_grid(blocks_per_grid, 1);
				reduction2048SumKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
					(gpu_results, level);
				cudaThreadSynchronize();
				CUT_CHECK_ERROR("");
				level *= 2;
				blocks_per_grid /= 2;
				if (blocks_per_grid == 0) {
					blocks_per_grid = 1;
				}
			}

			reduction2048MaxKernel<<<1, 2>>>(gpu_results);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_results, sizeof(float),
																cudaMemcpyDeviceToHost));
			if (new_distance < least_distance) {
				least_distance = new_distance;
			}

		}
	}

	CUDA_SAFE_CALL(cudaFree(gpu_results));

	return least_distance;

}





//////////////////////////
/// 4096 TEMPLATE SIZE ///
//////////////////////////

__global__ void distance4096NormalKernel(float* gpuImage, float* gpuTemp, float* gpuResult,
																				 int offX, int offY, int iWidth) {
	float distance
		= gpuTemp[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x]
		- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIdx.x + threadIdx.x];
	gpuResult[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x] = distance * distance;
}

__global__ void distance4096NormalFlipKernel(float* gpuImage, float* gpuTemp, float* gpuResult,
																						 int offX, int offY, int iWidth) {
	float distance
		= gpuTemp[4096*(blockIdx.y+1) - 512*blockIdx.x - threadIdx.x - 1]
		- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIdx.x + threadIdx.x];
	gpuResult[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x] = distance * distance;
}

__global__ void distance4096ReverseKernel(float* gpuImage, float* gpuTemp, float* gpuResult,
																					int offX, int offY, int iWidth, int tempSize) {
	float distance
		= gpuTemp[tempSize - 4096*blockIdx.y - 512*blockIdx.x - threadIdx.x - 1]
		- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIdx.x + threadIdx.x];
	gpuResult[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x] = distance * distance;
}

__global__ void distance4096ReverseFlipKernel(float* gpuImage, float* gpuTemp, float* gpuResult,
																							int offX, int offY, int iWidth, int tempSize) {
	float distance
		= gpuTemp[tempSize - 4096*(blockIdx.y+1) + 512*blockIdx.x + threadIdx.x]
		- gpuImage[(offX+blockIdx.y)*iWidth + offY + 512*blockIdx.x + threadIdx.x];
	gpuResult[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x] = distance * distance;
}

__global__ void transpose4096Kernel(float* gpuTemp) {
	if (512*blockIdx.x + threadIdx.x > blockIdx.y) {
		float temp = gpuTemp[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x];
		gpuTemp[4096*blockIdx.y + 512*blockIdx.x + threadIdx.x]
			= gpuTemp[4096*(512*blockIdx.x + threadIdx.x) + blockIdx.y];
		gpuTemp[4096*(512*blockIdx.x + threadIdx.x) + blockIdx.y] = temp;
	}
}

__global__ void reduction4096Kernel(float* gpuResult, unsigned int level) {
	int resultIndex = 2*level*(blockIdx.x*blockDim.x + threadIdx.x);
	if ((resultIndex + level) < 16777216) {
		gpuResult[resultIndex] += gpuResult[resultIndex + level];
	}
}

float calc_min_dist_4096(float* gpu_image, int i_width, int i_height, float* gpu_temp) {

	float least_distance = FLT_MAX;
	float new_distance = least_distance;

	int trans_height = i_height - 4096 + 1;
	int trans_width = i_width - 4096 + 1;
	unsigned int temp_size = 4096 * 4096;

	int threads_per_block = 512;
	int blocks_per_grid = 65535;

	size_t result_size = temp_size*sizeof(float);
	float* gpu_result;
	CUDA_SAFE_CALL(cudaMalloc(&gpu_result, result_size));

	dim3 dim_threads_per_block(threads_per_block, 1, 1);
	dim3 dim_blocks_per_grid(8, 4096);

	// normal
	for (int off_x = 0; off_x < trans_height; off_x++) {
		for (int off_y = 0; off_y < trans_width; off_y++) {
			distance4096NormalKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_result, off_x , off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 8 * 4096;
			while (level < temp_size) {
				// printf("%d level reduction with %d blocks\n", level, blocks_per_grid);
				reduction4096Kernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_result, level);
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

	// normal flip
	for (int off_x = 0; off_x < trans_height; off_x++) {
		for (int off_y = 0; off_y < trans_width; off_y++) {
			distance4096NormalFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_result, off_x , off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 8 * 4096;
			while (level < temp_size) {
				// printf("%d level reduction with %d blocks\n", level, blocks_per_grid);
				reduction4096Kernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_result, level);
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

	// reverse
	for (int off_x = 0; off_x < trans_height; off_x++) {
		for (int off_y = 0; off_y < trans_width; off_y++) {
			distance4096ReverseKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_result, off_x , off_y, i_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 8 * 4096;
			while (level != temp_size) {
				// printf("%d level reduction with %d blocks\n", level, blocks_per_grid);
				reduction4096Kernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_result, level);
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

	// reverse flip
	for (int off_x = 0; off_x < trans_height; off_x++) {
		for (int off_y = 0; off_y < trans_width; off_y++) {
			distance4096ReverseFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_result, off_x , off_y, i_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 8 * 4096;
			while (level != temp_size) {
				// printf("%d level reduction with %d blocks\n", level, blocks_per_grid);
				reduction4096Kernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_result, level);
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

	transpose4096Kernel<<<dim_blocks_per_grid, dim_threads_per_block>>>(gpu_temp);
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("");

	// normal
	for (int off_x = 0; off_x < trans_height; off_x++) {
		for (int off_y = 0; off_y < trans_width; off_y++) {
			distance4096NormalKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_result, off_x , off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 8 * 4096;
			while (level < temp_size) {
				// printf("%d level reduction with %d blocks\n", level, blocks_per_grid);
				reduction4096Kernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_result, level);
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

	// normal flip
	for (int off_x = 0; off_x < trans_height; off_x++) {
		for (int off_y = 0; off_y < trans_width; off_y++) {
			distance4096NormalFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_result, off_x , off_y, i_width);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 8 * 4096;
			while (level < temp_size) {
				// printf("%d level reduction with %d blocks\n", level, blocks_per_grid);
				reduction4096Kernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_result, level);
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

	// reverse
	for (int off_x = 0; off_x < trans_height; off_x++) {
		for (int off_y = 0; off_y < trans_width; off_y++) {
			distance4096ReverseKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_result, off_x , off_y, i_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 8 * 4096;
			while (level != temp_size) {
				// printf("%d level reduction with %d blocks\n", level, blocks_per_grid);
				reduction4096Kernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_result, level);
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

	// reverse flip
	for (int off_x = 0; off_x < trans_height; off_x++) {
		for (int off_y = 0; off_y < trans_width; off_y++) {
			distance4096ReverseFlipKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
				(gpu_image, gpu_temp, gpu_result, off_x , off_y, i_width, temp_size);
			cudaThreadSynchronize();
			CUT_CHECK_ERROR("");

			unsigned int level = 1;
			blocks_per_grid = 8 * 4096;
			while (level != temp_size) {
				// printf("%d level reduction with %d blocks\n", level, blocks_per_grid);
				reduction4096Kernel<<<blocks_per_grid, threads_per_block>>>
					(gpu_result, level);
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

	CUDA_SAFE_CALL(cudaFree(gpu_result));

	return least_distance;

}





////////////
/// MAIN ///
////////////


float calc_min_dist(float* gpu_image, int i_width, int i_height,
										float* gpu_temp, int t_width) {

	if (t_width <= 512) {

		return calc_min_dist_512(gpu_image, i_width, i_height, gpu_temp, t_width);

	}	else if (t_width == 1024) {

		return calc_min_dist_1024(gpu_image, i_width, i_height, gpu_temp);

	}	else if (t_width == 2048) {

		return calc_min_dist_2048(gpu_image, i_width, i_height, gpu_temp);

	}	else if (t_width == 4096) {

		return calc_min_dist_4096(gpu_image, i_width, i_height, gpu_temp);

	} else {

		return FLT_MAX;

	}

}