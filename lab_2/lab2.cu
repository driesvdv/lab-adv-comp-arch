#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>

/**
 * CUDA Kernel Device code
 *
 * Computes the sum of all elements in a 2D array using atomic operations
 */
__global__ void atomicOperation(int *input, int *output, int numElementsX, int numElementsY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // y index

    if (idx < numElementsX && idy < numElementsY) {
        atomicAdd(&output[idx], input[idy * numElementsX + idx]);
    }
}

/**
 * CUDA Kernel Device code
 * 
 * Computes the sum of all elements in a 2D array using a reduction algorithm using strides
 
*/
__global__ void reduction(int *input, int *output, int numElementsX, int numElementsY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // y index

    if (idx < numElementsX && idy < numElementsY) {
        int stride = 1;
        while (stride < numElementsX) {
            if (idx % (2 * stride) == 0) {
                input[idx * numElementsX + idy] += input[(idx + stride) * numElementsX + idy];
            }
            __syncthreads();
            stride *= 2;
        }
        if (idx == 0) {
            output[idy] = input[idy];
        }
    }

}

/**
 * CUDA Kernel Device code
 * 
 * Computes the sum of all elements in a 2D array using a reduction algorithm using strides and shared memory
*/

__global__ void reduction_shared(int *input, int *output, int numElementsX, int numElementsY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // y index

    if (idx < numElementsX && idy < numElementsY) {
        __shared__ int shared[1024];
        int stride = 1;
        shared[threadIdx.x] = input[idx * numElementsX + idy];
        __syncthreads();
        while (stride < numElementsX) {
            if (threadIdx.x % (2 * stride) == 0) {
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            }
            __syncthreads();
            stride *= 2;
        }
        if (threadIdx.x == 0) {
            output[idy] = shared[0];
        }
    }
}

/**
 * Host main routine
 */
int main(void) {
    cudaError_t err = cudaSuccess;
    // Error code to check return values for CUDA calls

    // Define the dimensions of the 2D array
    int numElementsX = 100000; // Number of elements along the x-axis
    int numElementsY = 100000; // Number of elements along the y-axis
    size_t sizeX = numElementsX * sizeof(int); // Size of each row

    // Allocate the host input matrix
    int *h_input = (int *)malloc(numElementsX * numElementsY * sizeof(int));

    // Verify that allocation succeeded
    if (h_input == NULL) {
        fprintf(stderr, "Failed to allocate host input matrix!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input matrix
    for (int i = 0; i < numElementsX * numElementsY; ++i) {
        h_input[i] = i+1; // Initialize all elements to 1 for simplicity
    }

    // Open CSV file for writing
    std::ofstream file("timing_data.csv");
    file << "Method,Execution Time (ms)\n";

    // Allocate device memory for input and output matrices
    int *d_input = NULL;
    int *d_output = NULL;
    err = cudaMalloc((void **)&d_input, numElementsX * numElementsY * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device input matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_output, sizeX);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device output vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input matrix to the device input matrix
    err = cudaMemcpy(d_input, h_input, numElementsX * numElementsY * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy host input matrix to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the atomic operation CUDA Kernel and measure execution time
    auto start_atomic = std::chrono::high_resolution_clock::now();
    atomicOperation<<<1, numElementsX>>>(d_input, d_output, numElementsX, numElementsY);
    cudaDeviceSynchronize();
    auto end_atomic = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> atomic_time = end_atomic - start_atomic;

    // Copy the result back to host and verify correctness
    int *h_output_atomic = (int *)malloc(sizeX);
    if (h_output_atomic == NULL) {
        fprintf(stderr, "Failed to allocate host output vector!\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(h_output_atomic, d_output, sizeX, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy device output vector to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print and record execution time for atomic operation
    printf("Atomic Operation - Execution Time: %f ms\n", atomic_time.count());
    file << "Atomic Operation," << atomic_time.count() << "\n";

    // Launch the reduction CUDA Kernel and measure execution time
    auto start_reduction = std::chrono::high_resolution_clock::now();
    reduction<<<1, numElementsX>>>(d_input, d_output, numElementsX, numElementsY);
    cudaDeviceSynchronize();
    auto end_reduction = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> reduction_time = end_reduction - start_reduction;

    // Copy the result back to host and verify correctness
    int *h_output_reduction = (int *)malloc(sizeX);
    if (h_output_reduction == NULL) {
        fprintf(stderr, "Failed to allocate host output vector!\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(h_output_reduction, d_output, sizeX, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy device output vector to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print and record execution time for reduction algorithm
    printf("Reduction Algorithm - Execution Time: %f ms\n", reduction_time.count());
    file << "Reduction Algorithm," << reduction_time.count() << "\n";

    // Launch the shared memory reduction CUDA Kernel and measure execution time
    auto start_reduction_shared = std::chrono::high_resolution_clock::now();
    reduction_shared<<<1, numElementsX>>>(d_input, d_output, numElementsX, numElementsY);
    cudaDeviceSynchronize();
    auto end_reduction_shared = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> reduction_shared_time = end_reduction_shared - start_reduction_shared;

    // Copy the result back to host and verify correctness
    int *h_output_reduction_shared = (int *)malloc(sizeX);
    if (h_output_reduction_shared == NULL) {
        fprintf(stderr, "Failed to allocate host output vector!\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(h_output_reduction_shared, d_output, sizeX, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy device output vector to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print and record execution time for shared memory reduction algorithm
    printf("Shared Memory Reduction Algorithm - Execution Time: %f ms\n", reduction_shared_time.count());
    file << "Shared Memory Reduction Algorithm," << reduction_shared_time.count() << "\n";

    // Free device global memory
    err = cudaFree(d_input);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device input matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_output);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device output vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_input);
    free(h_output_atomic);
    free(h_output_reduction);
    free(h_output_reduction_shared);

    // Close the CSV file
    file.close();

    printf("Done\n");
    return 0;
}
