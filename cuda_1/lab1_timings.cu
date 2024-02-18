#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>

/**
 * CUDA Kernel Device code
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(int *input, int *output, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    output[i] = input[numElements - i - 1];
  }
}

// Simple CPU-based array reversal function
void reverseArrayCPU(int *input, int *output, int numElements) {
  for (int i = 0; i < numElements; ++i) {
    output[i] = input[numElements - i - 1];
  }
}

int main(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
  int maxElements = 100000;
  int step = 1000;
  int numIterations = 100;

  std::ofstream file("timing_data.csv");
  file << "Array Size, GPU Time (ms), CPU Time (ms)\n";

  for (int numElements = 100; numElements <= maxElements; numElements += step) {
    size_t size = numElements * sizeof(int);
    file << numElements << ", ";

    // Allocate host memory
    int *h_input = (int *)malloc(size);
    int *h_output_gpu = (int *)malloc(size);
    int *h_output_cpu = (int *)malloc(size);

    // Initialize host input
    for (int i = 0; i < numElements; ++i) {
      h_input[i] = i;
    }

    double total_gpu_time = 0.0;
    double total_cpu_time = 0.0;

    for (int iter = 0; iter < numIterations; ++iter) {
      // Allocate device memory
      int *d_input = NULL;
      int *d_output = NULL;
      err = cudaMalloc((void **)&d_input, size);
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

      err = cudaMalloc((void **)&d_output, size);
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

      // Copy input data from host to device
      err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

      // Launch GPU kernel
      auto start_gpu = std::chrono::high_resolution_clock::now();
      vectorAdd<<<(numElements + 255) / 256, 256>>>(d_input, d_output, numElements);
      cudaDeviceSynchronize();
      auto end_gpu = std::chrono::high_resolution_clock::now();

      // Copy output data from device to host
      err = cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

      // Measure GPU time
      std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;
      total_gpu_time += gpu_time.count();

      // Measure CPU time
      auto start_cpu = std::chrono::high_resolution_clock::now();
      reverseArrayCPU(h_input, h_output_cpu, numElements);
      auto end_cpu = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
      total_cpu_time += cpu_time.count();

      // Free memory
      err = cudaFree(d_input);
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

      err = cudaFree(d_output);
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
    }

    // Compute average GPU and CPU time
    double avg_gpu_time = total_gpu_time / numIterations;
    double avg_cpu_time = total_cpu_time / numIterations;

    // Write average times to file
    file << avg_gpu_time << ", ";
    file << avg_cpu_time << "\n";

    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
  }

  return 0;
}
