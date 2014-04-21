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

__global__ void reductionKernel_old(float* result, int len, int level) {
  int arrayIndex = 2*level*(blockIdx.y*4096 + blockIdx.x*512 + threadIdx.x);
  if (arrayIndex + level < len) {
    result[arrayIndex] += result[arrayIndex + level];
  }
}

__global__ void leastDistanceKernel (float* A, float* B, float* C, int len);

__global__ void leastDistance4096Kernel_old(float *image, float *temp, float *result, int translation_width, int i, int j) {
  float dist = temp[blockIdx.y*4096 + blockIdx.x*512 + threadIdx.x] - image[blockIdx.y*4096 + blockIdx.x*512 + threadIdx.x + i*translation_width + j];
  result[blockIdx.x*512 + threadIdx.x] = dist * dist;
}










__global__ void distance4096Kernel(float* gpu_image, float* gpu_temp, float* gpu_result, int num_translations,
                                   int offset, int t_width, int i_width) {
  int thread_index = offset + blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_index < (num_translations * t_width * t_width)) {
    int pixel_index = thread_index / num_translations;
    int distance = gpu_temp[pixel_index]
                 - gpu_image[thread_index % num_translations + (pixel_index / t_width) * i_width + pixel_index % t_width];
    gpu_result[thread_index % num_translations] += distance * distance;  
  }  
}

__global__ void reductionKernel(float* gpu_result, int num_iterations, int level, int offset) {
  int thread_index = offset + 2 * level * (blockIdx.x * blockDim.x + threadIdx.x);
  if (thread_index + level < num_iterations) {
    if (gpu_result[thread_index + level] < gpu_result[thread_index]) {
      gpu_result[thread_index] = gpu_result[thread_index + level];
    }
  }
}

__global__ void distanceSerialKernel(float* gpu_image, float* gpu_temp, float* gpu_result, float* gpu_test, int num_translations,
                                     int i_width, int t_width, int translation_width, int translations_per_block) {
  int trans_num = threadIdx.x;
  int pixel_num = threadIdx.y + translations_per_block * blockIdx.x;
  if (pixel_num < (num_translations)) {
    float distance = gpu_temp[pixel_num] - gpu_image[((int) (trans_num / translation_width)) * i_width + ((int) (pixel_num / t_width)) * i_width + trans_num % translation_width + pixel_num % t_width];
    if (pixel_num == 0 && trans_num < 100) {
      gpu_test[trans_num] = distance;
    }
    gpu_result[trans_num] += distance * distance;  
  }  
}

float calc_min_dist(float *gpu_image, int i_width, int i_height, float *gpu_temp, int t_width) {

  float least_distance = UINT_MAX;
  
  if (t_width == 4096) {

    int threads_per_block = 512;
    int blocks_per_grid = 65534;

    int translation_height = i_height - t_width + 1;
    int translation_width = i_width - t_width + 1;
    int num_translations = translation_height * translation_width;
    float new_distance;

    float* result = (float *)malloc(num_translations*sizeof(float));
    if (result == NULL) {
      printf("Unable to allocate space for result");
      exit(EXIT_FAILURE);
    }
    for (int counter = 0; counter < num_translations; counter++) {
      result[counter] = 0.0;
    }
    float* gpu_result;
    size_t arraySize = num_translations*sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc(&gpu_result, arraySize));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_result, result, num_translations*sizeof(float),
                              cudaMemcpyHostToDevice));

    float* test = (float *)malloc(100*sizeof(float));
    float* gpu_test;
    size_t test_size = 100*sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc(&gpu_test, test_size));

    printf("%d\n", 3); /*
    ///////////////////
    dim3 dim_threads_per_block(threads_per_block, 1, 1);
    dim3 dim_blocks_per_grid(blocks_per_grid, 1);

    int num_operations = num_translations * t_width * t_width;
    int num_per_iter = threads_per_block * blocks_per_grid;
    int num_iter = num_operations / num_per_iter;
    if (num_iter * num_per_iter < num_operations) {
      num_iter ++;
    }


    for (int counter = 0; counter < num_iter; counter ++) { /*
      distance4096Kernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
        (gpu_image, gpu_temp, gpu_result, num_translations,
         num_operations - counter*num_per_iter, t_width, i_width); 
      distance4096Kernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
        (gpu_image, gpu_temp, gpu_result, num_translations,
         0, t_width, i_width);
      cudaThreadSynchronize();
      CUT_CHECK_ERROR("");
    }
    //////////////////*/
    // int num_operations = num_translations * t_width * t_width;
    // int num_per_iter = threads_per_block * blocks_per_grid;

    if (num_translations < threads_per_block) {
      int translations_per_block = threads_per_block / num_translations;
      int num_blocks = num_translations / translations_per_block + 1;
      while (num_blocks > 0) {
        if (num_blocks > blocks_per_grid) {
          dim3 dim_threads_per_block(num_translations, translations_per_block, 1);
          dim3 dim_blocks_per_grid(blocks_per_grid, 1);
          distanceSerialKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
            (gpu_image, gpu_temp, gpu_result, gpu_test, num_translations, i_width, t_width,
             translation_width, translations_per_block);
          cudaThreadSynchronize();
          CUT_CHECK_ERROR("");
        } else {
          dim3 dim_threads_per_block(num_translations, translations_per_block, 1);
          dim3 dim_blocks_per_grid(num_blocks, 1);
          distanceSerialKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
            (gpu_image, gpu_temp, gpu_result, gpu_test, num_translations, i_width, t_width,
             translation_width, translations_per_block);
          cudaThreadSynchronize();
          CUT_CHECK_ERROR("");
        }
        num_blocks -= blocks_per_grid;

      }
    } else {
      // int 
      // dim3 dim_threads_per_block(threads_per_block, 1, 1);
      // dim3 dim_blocks_per_grid(num_translations / threads_per_block, 1);
      printf("Reached else case of num_translations! \n");
    }

    ///////////////////
    printf("%d\n", 4);

    CUDA_SAFE_CALL(cudaMemcpy(test, gpu_test, test_size,
                              cudaMemcpyDeviceToHost));
    for (int i = 0; i < 100; i++) {
      printf("%f\n", test[i]);
    }


    printf("%d\n", 5);
    CUDA_SAFE_CALL(cudaMemcpy(result, gpu_result, num_translations*sizeof(float),
                              cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_translations; i++) {
      printf("%f\n", result[i]);
    }
 
    int level = 1;
    int num_blocks = 1;
    if (num_translations <= (threads_per_block * blocks_per_grid)) {
      if (num_translations <= threads_per_block) {
        dim3 dim_threads_per_block(num_translations, 1, 1);
        dim3 dim_blocks_per_grid(num_blocks, 1);
        while (level < num_translations) {
          reductionKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
            (gpu_result, num_translations, level, 0);
          cudaThreadSynchronize();
          CUT_CHECK_ERROR("");
          level *= 2;
          num_blocks /= 2;
          if (num_blocks == 0) {
            num_blocks = 1;
          }
        }
      } else {
        num_blocks = num_translations / threads_per_block + 1;
        dim3 dim_threads_per_block(threads_per_block, 1, 1);
        dim3 dim_blocks_per_grid(num_blocks, 1);
        while (level < num_translations) {
          reductionKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>
            (gpu_result, num_translations, level, 0);
          cudaThreadSynchronize();
          CUT_CHECK_ERROR("");
          level *= 2;
          num_blocks /= 2;
          if (num_blocks == 0) {
            num_blocks = 1;
          }
        }
      }
      
    } else {
      printf("Input is too large!");
    }

    printf("%d\n", 6);
   
    CUDA_SAFE_CALL(cudaMemcpy(&new_distance, gpu_result, sizeof(float),
                              cudaMemcpyDeviceToHost));
    if (new_distance < least_distance) {
      least_distance = new_distance;
    }

    printf("%d\n", 7);
   
    // CUDA_SAFE_CALL(cudaFree(gpu_image));
    // CUDA_SAFE_CALL(cudaFree(gpu_temp));
    CUDA_SAFE_CALL(cudaFree(gpu_result));
    CUDA_SAFE_CALL(cudaFree(gpu_test));

    free(result);

  }

  printf("%f\n", least_distance);
  return least_distance;

}

/* Returns the squared Euclidean distance between TEMPLATE and IMAGE. The size of IMAGE
 * is I_WIDTH * I_HEIGHT, while TEMPLATE is square with side length T_WIDTH. The template
 * image should be flipped, rotated, and translated across IMAGE.
 */ /*
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


*/