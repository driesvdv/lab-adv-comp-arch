#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>


/**
 * CPU code
 *
 * Computes the maximum value among all elements in an array
 */
int maxElement(int *input, int numElements)
{
    int max = input[0];
    for (int i = 1; i < numElements; ++i)
    {
        if (input[i] > max)
        {
            max = input[i];
        }
    }
    return max;
}

/**
 * CUDA Kernel Device code
 *
 * Computes the sum of all elements in a 1D array using atomic operations
 */
__global__ void atomicOperation(int *input, int *output, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Index of the element to process

    if (idx < numElements)
    {
        atomicMax(&output[0], input[idx]);
    }
}

__global__ void reduction(int *input, int *output, int numElements, int *largestValue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements)
    {
        int stride = 1;
        while (stride < numElements)
        {
            if (idx % (2 * stride) == 0)
            {
                int val1 = input[idx];
                int val2 = input[idx + stride];
                input[idx] = max(val1, val2);
            }
            __syncthreads();
            stride *= 2;
        }
        if (idx == 0)
        {
            output[0] = input[0];
            atomicMax(largestValue, input[0]); // Update largestValue with the maximum value in this chunk
        }
    }
}

/**
 * CUDA Kernel Device code
 *
 * Computes the maximum value among all elements in a 1D array using reduction
 *
 * This version uses a synchronous approach to the reduction
 * Only 2048 elements are processed at a time, once this is done the next 2048 elements are processed
 */
__global__ void reduction_synchronous(int *input, int *output, int numElements, int *largestValue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = 1;

    for (int i = 0; i < numElements; i += 2048)
    {
        int val1 = input[idx];
        int val2 = input[idx + stride];
        input[idx] = max(val1, val2);
        atomicMax(largestValue, input[idx]);

        __syncthreads();
    }
}

int main(void)
{
    cudaError_t err = cudaSuccess;
    // Error code to check return values for CUDA calls

    // Define the size of the array
    //int numElements = 1000000; // Number of elements in the array
    int numElements = 1000000; // Number of elements in the array
    size_t size = numElements * sizeof(int); // Size of the array

    // Allocate the host input array
    int *h_input = (int *)malloc(size);

    // Verify that allocation succeeded
    if (h_input == NULL)
    {
        fprintf(stderr, "Failed to allocate host input array!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input array
    for (int i = 0; i < numElements; ++i)
    {
        h_input[i] = i + 1; // Initialize all elements to 1 for simplicity
    }

    // Open CSV file for writing
    std::ofstream file("timing_data.csv");
    if (!file.is_open())
    {
        fprintf(stderr, "Failed to open timing_data.csv file for writing!\n");
        exit(EXIT_FAILURE);
    }
    file << "Method,Execution Time (ms)\n";

    // Timing for CPU execution
    auto start_cpu = std::chrono::high_resolution_clock::now();
    int max_cpu = maxElement(h_input, numElements);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // Print and record execution time for CPU execution
    printf("Max element in the array (CPU): %d\n", max_cpu);
    printf("CPU Execution Time: %f ms\n", cpu_time.count());
    file << "CPU," << cpu_time.count() << "\n";

    // Allocate device memory for input and output arrays
    int *d_input = NULL;
    int *d_output = NULL;
    err = cudaMalloc((void **)&d_input, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device input array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_output, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device output variable (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input array to the device input array
    err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy host input array to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Timing for atomic operation
    auto start_atomic = std::chrono::high_resolution_clock::now();
    dim3 blockDim_atomic(1024);                                                     // block dimension (assuming each block has 256 threads)
    dim3 gridDim_atomic((numElements + blockDim_atomic.x - 1) / blockDim_atomic.x); // grid dimension
    atomicOperation<<<gridDim_atomic, blockDim_atomic>>>(d_input, d_output, numElements);
    cudaDeviceSynchronize();
    auto end_atomic = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> atomic_time = end_atomic - start_atomic;

    // Copy the result back to host and verify correctness
    int h_output_atomic;
    err = cudaMemcpy(&h_output_atomic, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy device output variable to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print and record execution time for atomic operation
    printf("Max element in the array (Atomic): %d\n", h_output_atomic);
    printf("Atomic Operation - Execution Time: %f ms\n", atomic_time.count());
    file << "Atomic Operation," << atomic_time.count() << "\n";

    // Reset device memory
    cudaMemset(d_output, 0, sizeof(int));

    // Split array in chunks
    int chunkSize = 2048;
    int numChunks = (int)ceil((float)numElements / chunkSize);

    int largestValue = 0; // Initialize largestValue
    int *d_largestValue; // Device pointer to store largestValue
    cudaMalloc((void **)&d_largestValue, sizeof(int)); // Allocate device memory
    cudaMemcpy(d_largestValue, &largestValue, sizeof(int), cudaMemcpyHostToDevice); // Copy largestValue to device


    auto start_reduction = std::chrono::high_resolution_clock::now();

    for (int chunkIndex = 0; chunkIndex < numChunks; ++chunkIndex)
    {
        int start = chunkIndex * chunkSize;
        int end = min(start + chunkSize - 1, numElements - 1); // End index of the current chunk

        // Calculate the size of the current chunk
        int currentChunkSize = end - start + 1;

        // Timing for reduction
        dim3 blockDim_reduction(1024);                                                                // block dimension (assuming each block has 1024 threads)
        dim3 gridDim_reduction((currentChunkSize + blockDim_reduction.x - 1) / blockDim_reduction.x); // grid dimension
        reduction<<<gridDim_reduction, blockDim_reduction>>>(d_input + start, d_output, currentChunkSize, d_largestValue);        
        cudaDeviceSynchronize();
    }

    auto end_reduction = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> reduction_time = end_reduction - start_reduction;

    cudaMemcpy(&largestValue, d_largestValue, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_largestValue);


    // Print and record execution time for reduction
    printf("Max element in the array (Reduction): %d\n", largestValue);
    printf("Reduction - Execution Time: %f ms\n", reduction_time.count());
    file << "Reduction," << reduction_time.count() << "\n";

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);

    // Close the file
    file.close();
        
    return 0;
}