/*
 * Proj 3-2 SKELETON
 */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include "utils.h"

/* Does a horizontal flip of the array arr */
void flip_horizontal(float *arr, int width) {
  /* YOU MAY WISH TO IMPLEMENT THIS */

}

/* Transposes the square array ARR. */
void transpose(float *arr, int width) {
  /* YOU MAY WISH TO IMPLEMENT THIS */

}

/* Rotates the square array ARR by 90 degrees counterclockwise. */
void rotate_ccw_90(float *arr, int width) {
  /* YOU MAY WISH TO IMPLEMENT THIS */

}

__global__ void reductionKernel(float* result, int len, int level) {
  int arrayIndex = 2*level*(blockIdx.y*4096 + blockIdx.x*512 + threadIdx.x);
  if (arrayIndex + level < len) {
    result[arrayIndex] += result[arrayIndex + level];
  }
}

__global__ void leastDistanceKernel (float* A, float* B, float* C, int len);

__global__ void leastDistance4096Kernel(float *image, float *temp, float *result, int translation_width, int i, int j) {
  float dist = temp[blockIdx.y*4096 + blockIdx.x*512 + threadIdx.x] - image[blockIdx.y*4096 + blockIdx.x*512 + threadIdx.x + i*translation_width + j];
  result[blockIdx.x*512 + threadIdx.x] = dist * dist;
}

float calc_min_dist(float *image, int i_width, int i_height, float *temp, int t_width) {

  if (t_width == 4096) {

    int threads_per_block = 512;
    int blocks_per_grid = 65535;

    int translation_height = i_height - t_width + 1;
    int translation_width = i_width - t_width + 1;

    float *gpu_image, *gpu_temp;
    CUDA_SAFE_CALL(cudaMalloc(&gpu_image, i_width*i_height));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_image, image, i_width*i_height, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc(&gpu_temp, t_width*t_width));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_temp, temp, t_width*t_width, cudaMemcpyHostToDevice));

    float *gpu_result;
    CUDA_SAFE_CALL(cudaMalloc(&gpu_result, translation_width*translation_height));

    int num_operations = translation_width * translation_height * t_width * t_width;
    int num_per_iter = threads_per_block * blocks_per_grid;.
    int num_iter = num_operations / num_per_iter;
    if (num_iter * num_per_iter < num_operations) {
      num_iter ++;
    }
  
  }


}

/* Returns the squared Euclidean distance between TEMPLATE and IMAGE. The size of IMAGE
 * is I_WIDTH * I_HEIGHT, while TEMPLATE is square with side length T_WIDTH. The template
 * image should be flipped, rotated, and translated across IMAGE.
 */
float calc_min_dist_old(float *image, int i_width, int i_height, float *temp, int t_width) {
  // float* image and float* temp are pointers to GPU addressible memory
  // You MAY NOT copy this data back to CPU addressible memory and you MAY 
  // NOT perform any computation using values from image or temp on the CPU.
  // The only computation you may perform on the CPU directly derived from distance
  // values is selecting the minimum distance value given a calculated distance and a 
  // "min so far"
  
  // Basic units of computation:
  // - one comparison
  // - one eight configuration ie. one translation
  // - one traversal in min(width, height) dimension
  // - all translations

  int threads_per_block = 512; // 2^9
  int blocks_per_grid = 65535; // 2^16
  int translation_width = i_width - t_width + 1;
  int translation_height = i_height - t_width + 1;
  int blocks_per_comparison = 1;
  
  float *gpu_image, *gpu_temp;
  CUDA_SAFE_CALL(cudaMalloc(&gpu_image, i_width*i_height));
  CUDA_SAFE_CALL(cudaMalloc(gpu_image, image, i_width*i_height, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc(&gpu_temp, t_width * t_width));
  CUDA_SAFE_CALL(cudaMalloc(gpu_temp, temp, t_width*t_width, cudaMemcpyHostToDevice));

  if (t_width <= 512) {
    blocks_per_comparison = 512;
    for (int i = 0; i < translation_height; i++) {
      for (int j = 0; j < translation_width; j++) {

      }
    }
  } else if (t_width == 1024) {
    blocks_per_comparison = 2048;
    for (int i = 0; i < translation_height; i++) {
      for (int j = 0; j < translation_width; j++) {
        
      }
    }
  } else if (t_width == 2048) {
    blocks_per_comparison = 8192;
    for (int i = 0; i < translation_height; i++) {
      for (int j = 0; j < translation_width; j++) {
        
      }
    }
  } else if (t_width >= 4096) {
    size_t arraySize = translation_width * translation_height * sizeof(float);
    // float* result = (float *)malloc(arraySize);
    float* gpu_result;
    CUDA_SAFE_CALL(cudaMalloc(gpu_result, arraySize));
    blocks_per_comparison = 32768;
    float least_distance = UINT_MAX;
    for (int i = 0; i < translation_height; i++) {
      for (int j = 0; j < translation_width; j++) {
        dim3 dim_threads_per_block(threads_per_block, 1, 1);
        dim3 dim_blocks_per_grid(8, 4096);
        leastDistance4096Kernel<<<dim_blocks_per_grid, dim_threads_per_block>>>(gpu_image, gpu_temp, gpu_result, translation_width, i, j);
        cudaThreadSynchronize();
        CUT_CHECK_ERROR("");

        int level = 1;
        while (level != (8*4096)) {
          blocks_per_grid = 8*4096;
          dim3 dim_blocks_per_grid(blocks_per_grid, 1);
          reductionKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>(gpu_result, 4096*4096, level);
          cudaThreadSynchronize();
          CUT_CHECK_ERROR("");
          level *= 2;
          blocks_per_grid /= 2;
          if (blocks_per_grid == 0) {
            blocks_per_grid = 1;
        }
        CUDA_SAFE_CALL(cudaMemcpy(&gpu_result, ))
      }
    }


  }


 



  // dim3 dim_threads_per_block(threads_per_block, 1, 1);
  // dim3 dim_blocks_per_grid(blocks_per_grid, 1);



  reductionKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>()

  

  return 0;
}


