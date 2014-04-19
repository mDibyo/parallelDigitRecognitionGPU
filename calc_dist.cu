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

__global__ void reductionKernel(float* A, int len, int level);

__global__ void leastDistanceKernel (float* A, float* B, float* C, int len)

/* Returns the squared Euclidean distance between TEMPLATE and IMAGE. The size of IMAGE
 * is I_WIDTH * I_HEIGHT, while TEMPLATE is square with side length T_WIDTH. The template
 * image should be flipped, rotated, and translated across IMAGE.
 */
float calc_min_dist(float *image, int i_width, int i_height, float *temp, int t_width) {
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

  // given t_width = 4096, i_width = 4196, i_height = 4196
  // given t_width = 512, i_width = 612, i_height = 712 [ 16, 4096 ]
  // 
  // for (int i = 0; i < i_height; i ++) {
  //   for (int j = 0; j < i_width; j++) {}
  // }
  
  if (t_width <= 16) {
    blocks_per_comparison = 1;
  } else if (t_width <= 512) {

  }

  if (t_width <= 16) {
    dim3 dim_threads_per_block(t_width, t_width, 1);
    if (translation_width * translation_height < 8192) {
      // Run all translations and all eight different configurations in one go
    } else if (translation_width * translation_height < 65536) {
      // Run all translations in one go and do translations separately
    } else if (translation_height < 65536 && translation_width < 65536)
  } else if (t_width <= 512) {
    dim3 dim_threads_per_block(t_width, 1, 1);
  } else {
    dim3 dim_threads_per_block(512, 1, 1);
  }
  
  dim3 dim_threads_per_block(8, 8, 8);



  // dim3 dim_threads_per_block(threads_per_block, 1, 1);
  // dim3 dim_blocks_per_grid(blocks_per_grid, 1);



  reductionKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>()

  

  return 0;
}


