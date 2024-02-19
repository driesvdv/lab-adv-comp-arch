#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

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

int main()
{
    // Define the size of the array
    int numElements = 2048; // Number of elements in the array
    size_t size = numElements * sizeof(int); // Size of the array

    // Allocate the host input array
    int *h_input = new int[numElements];

    // Initialize the host input array
    for (int i = 0; i < numElements; ++i)
    {
        h_input[i] = i + 1; // Initialize all elements to increasing values for simplicity
    }

    // Allocate device memory for input and output arrays
    int *d_input = NULL;
    int *d_output = NULL;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, sizeof(int));

    // Copy the host input array to the device input array
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Allocate device memory for largestValue
    int *d_largestValue;
    cudaMalloc((void **)&d_largestValue, sizeof(int));
    cudaMemset(d_largestValue, 0, sizeof(int));

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel
    dim3 blockDim(1024); // 256 threads per block
    dim3 gridDim((numElements + blockDim.x - 1) / blockDim.x);
    reduction_synchronous<<<gridDim, blockDim>>>(d_input, d_output, numElements, d_largestValue);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Kernel execution time: " << duration.count() << " milliseconds" << std::endl;

    // Copy the result from the device output array to the host output array
    int h_largestValue;
    cudaMemcpy(&h_largestValue, d_largestValue, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Largest value found: " << h_largestValue << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_largestValue);

    // Free host memory
    delete[] h_input;

    return 0;
}
