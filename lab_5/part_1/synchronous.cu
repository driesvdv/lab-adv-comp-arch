#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <fstream>
#include <iostream>

#define ARR_SIZE_MAX 100000
#define NUM_ITERATIONS 100

/**
 * CUDA device code for performing element-wise summation operation on array
 */
__global__ void summation(int *x, int *y, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements)
    {
        while (numElements > 1)
        {
            if (idx < numElements / 2)
            {
                x[idx] += x[idx + numElements / 2];
            }
            __syncthreads();
            numElements /= 2;
        }
    }
    if (idx == 0)
    {
        *y = x[0];
    }
}

/**
 * CUDA device code for performing element-wise multiplication on array
 */
__global__ void multiplication(int *x, int *y, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements)
    {
        while (numElements > 1)
        {
            if (idx < numElements / 2)
            {
                x[idx] *= x[idx + numElements / 2];
            }
            __syncthreads();
            numElements /= 2;
        }
    }
    if (idx == 0)
    {
        *y = x[0];
    }
}

/**
 * CUDA device code for returning the smallest element from array
 */
__global__ void minimum(int *input, int *result, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements)
    {
        while (numElements > 1)
        {
            if (idx < numElements / 2)
            {
                if (input[idx] > input[idx + numElements / 2])
                {
                    input[idx] = input[idx + numElements / 2];
                }
            }
            __syncthreads();
            numElements /= 2;
        }
    }
    if (idx == 0)
    {
        *result = input[0];
    }
}

/**
 * CUDA device code for returning the largest element from array
 */
__global__ void maximum(int *input, int *result, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements)
    {
        while (numElements > 1)
        {
            if (idx < numElements / 2)
            {
                if (input[idx] < input[idx + numElements / 2])
                {
                    input[idx] = input[idx + numElements / 2];
                }
            }
            __syncthreads();
            numElements /= 2;
        }
    }
    if (idx == 0)
    {
        *result = input[0];
    }
}

int main(void)
{
    // Open CSV file for writing
    std::ofstream csv_file("./../timings/synchronous_cuda_results.csv");
    csv_file << "Array size, GPU execution time (µs), Data Prep time (µs), Total Execution time (µs)";

    // Seed the random number generator
    srand(time(NULL));

    // Warm up the GPU
    int *d_dummy;
    cudaMalloc((void **)&d_dummy, sizeof(int));
    summation<<<1, 1>>>(nullptr, d_dummy, 1);
    multiplication<<<1, 1>>>(nullptr, d_dummy, 1);
    minimum<<<1, 1>>>(nullptr, d_dummy, 1);
    maximum<<<1, 1>>>(nullptr, d_dummy, 1);
    cudaFree(d_dummy);

    // Loop over array sizes from 10 to ARR_SIZE_MAX
    for (int arr_size = 10; arr_size <= ARR_SIZE_MAX; arr_size *= 10)
    {
        // Calculate number of blocks and threads
        int numThreads = (arr_size + 1) / 2;
        int numThreadsMax = 1024;
        numThreads = numThreads > numThreadsMax ? numThreadsMax : numThreads;
        int numBlocks = (arr_size + numThreads - 1) / numThreads;

        if (numThreads > 1024)
        {
            throw std::runtime_error("Array size is too large for current implementation. Maximum array size is 1024 elements.");
        }

        // Perform benchmarking for each operation
        float summation_time = 0, multiplication_time = 0, minimum_time = 0, maximum_time = 0, data_prep_time = 0;
        for (int iter = 0; iter < NUM_ITERATIONS; iter++)
        {
            auto start_data_prep = std::chrono::high_resolution_clock::now();

            // Allocate memory for host arrays
            int *arr_1 = new int[arr_size];
            int *arr_2 = new int[arr_size];
            int *arr_3 = new int[arr_size];
            int *arr_4 = new int[arr_size];

            // Fill host arrays with random numbers
            for (int i = 0; i < arr_size; i++)
            {
                arr_1[i] = rand() % 1000;
                arr_2[i] = rand() % 1000;
                arr_3[i] = rand() % 1000;
                arr_4[i] = rand() % 1000;
            }

            // Allocate memory on the device
            int *d_arr_1, *d_arr_2, *d_arr_3, *d_arr_4;
            cudaMalloc((void **)&d_arr_1, arr_size * sizeof(int));
            cudaMalloc((void **)&d_arr_2, arr_size * sizeof(int));
            cudaMalloc((void **)&d_arr_3, arr_size * sizeof(int));
            cudaMalloc((void **)&d_arr_4, arr_size * sizeof(int));

            // Copy host arrays to device
            cudaMemcpy(d_arr_1, arr_1, arr_size * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_arr_2, arr_2, arr_size * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_arr_3, arr_3, arr_size * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_arr_4, arr_4, arr_size * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            auto end_data_prep = std::chrono::high_resolution_clock::now();
            data_prep_time += std::chrono::duration_cast<std::chrono::microseconds>(end_data_prep - start_data_prep).count();

            // Start the timer for each operation
            auto start_summation = std::chrono::high_resolution_clock::now();
            summation<<<numBlocks, numThreads>>>(d_arr_1, nullptr, arr_size);
            cudaDeviceSynchronize();
            auto end_summation = std::chrono::high_resolution_clock::now();
            summation_time += std::chrono::duration_cast<std::chrono::microseconds>(end_summation - start_summation).count();

            auto start_multiplication = std::chrono::high_resolution_clock::now();
            multiplication<<<numBlocks, numThreads>>>(d_arr_2, nullptr, arr_size);
            cudaDeviceSynchronize();
            auto end_multiplication = std::chrono::high_resolution_clock::now();
            multiplication_time += std::chrono::duration_cast<std::chrono::microseconds>(end_multiplication - start_multiplication).count();

            auto start_minimum = std::chrono::high_resolution_clock::now();
            minimum<<<numBlocks, numThreads>>>(d_arr_3, nullptr, arr_size);
            cudaDeviceSynchronize();
            auto end_minimum = std::chrono::high_resolution_clock::now();
            minimum_time += std::chrono::duration_cast<std::chrono::microseconds>(end_minimum - start_minimum).count();

            auto start_maximum = std::chrono::high_resolution_clock::now();
            maximum<<<numBlocks, numThreads>>>(d_arr_4, nullptr, arr_size);
            cudaDeviceSynchronize();
            auto end_maximum = std::chrono::high_resolution_clock::now();
            maximum_time += std::chrono::duration_cast<std::chrono::microseconds>(end_maximum - start_maximum).count();

            // Free device memory
            cudaFree(d_arr_1);
            cudaFree(d_arr_2);
            cudaFree(d_arr_3);
            cudaFree(d_arr_4);

            // Free host memory
            delete[] arr_1;
            delete[] arr_2;
            delete[] arr_3;
            delete[] arr_4;
        }

        // Compute average time for each operation
        summation_time /= NUM_ITERATIONS;
        multiplication_time /= NUM_ITERATIONS;
        minimum_time /= NUM_ITERATIONS;
        maximum_time /= NUM_ITERATIONS;
        data_prep_time /= NUM_ITERATIONS;

        auto execution_time = summation_time + multiplication_time + minimum_time + maximum_time;
        auto total_time = execution_time + data_prep_time;

        // Write results to CSV
        csv_file << "\n"
                 << arr_size << ", " << execution_time << ", " << data_prep_time << ", " << total_time;
    }

    // Close CSV file
    csv_file.close();

    return 0;
}
