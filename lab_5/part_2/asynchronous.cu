#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <vector>

#define ARR_SIZE_MAX 100000000
#define ARR_SIZE_MIN 10
#define ARR_SIZE_STEP 10
#define NUM_RUNS 4

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
    std::ofstream outputFile("./../timings/asynchronous_cuda_results.csv");
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }

    outputFile << "Array Size,Execution time (µs)" << std::endl;

    for (int arr_size = 10; arr_size <= ARR_SIZE_MAX; arr_size *= 10)
    {
        // Calculate number of blocks and threads
        int numThreads = 1024;
        int numBlocks = (arr_size + numThreads - 1) / numThreads;

        if (numThreads > 1024)
        {
            throw std::runtime_error("Array size is too large for current implementation. Maximum array size is 1024 elements.");
        }

        // Init timers
        float total_time = 0, total_data_prep_time = 0, total_gpu_exec_time = 0;

        // Perform the operations with streams
        for (int run = 0; run < NUM_RUNS; ++run)
        {
            // Start the timer for the operations
            auto start_operations = std::chrono::high_resolution_clock::now();

            // Create CUDA streams
            cudaStream_t stream1, stream2, stream3, stream4;
            cudaStreamCreate(&stream1);
            cudaStreamCreate(&stream2);
            cudaStreamCreate(&stream3);
            cudaStreamCreate(&stream4);

            // Copy data to device asynchronously

            std::vector<int> arr_1(arr_size);
            for (int i = 0; i < arr_size; i++)
            {
                arr_1[i] = rand() % 1000;
            }
            int *d_arr_1;
            cudaMalloc((void **)&d_arr_1, arr_size * sizeof(int));
            int *d_out_1;
            cudaMalloc((void **)&d_out_1, sizeof(int));

            cudaMemcpyAsync(d_arr_1, arr_1.data(), arr_size * sizeof(int), cudaMemcpyHostToDevice, stream1);
            summation<<<numBlocks, numThreads, 0, stream1>>>(d_arr_1, d_out_1, arr_size);

            std::vector<int> arr_2(arr_size);
            for (int i = 0; i < arr_size; i++)
            {
                arr_2[i] = rand() % 1000;
            }
            int *d_arr_2;
            cudaMalloc((void **)&d_arr_2, arr_size * sizeof(int));
            int *d_out_2;
            cudaMalloc((void **)&d_out_2, sizeof(int));

            cudaMemcpyAsync(d_arr_2, arr_2.data(), arr_size * sizeof(int), cudaMemcpyHostToDevice, stream2);
            multiplication<<<numBlocks, numThreads, 0, stream2>>>(d_arr_2, d_out_2, arr_size);

            std::vector<int> arr_3(arr_size);
            for (int i = 0; i < arr_size; i++)
            {
                arr_3[i] = rand() % 1000;
            }
            int *d_arr_3;
            cudaMalloc((void **)&d_arr_3, arr_size * sizeof(int));
            int *d_out_3;
            cudaMalloc((void **)&d_out_3, sizeof(int));

            cudaMemcpyAsync(d_arr_3, arr_3.data(), arr_size * sizeof(int), cudaMemcpyHostToDevice, stream3);
            maximum<<<numBlocks, numThreads, 0, stream3>>>(d_arr_3, d_out_3, arr_size);

            std::vector<int> arr_4(arr_size);
            for (int i = 0; i < arr_size; i++)
            {
                arr_4[i] = rand() % 1000;
            }
            int *d_arr_4;
            cudaMalloc((void **)&d_arr_4, arr_size * sizeof(int));
            int *d_out_4;
            cudaMalloc((void **)&d_out_4, sizeof(int));

            cudaMemcpyAsync(d_arr_4, arr_4.data(), arr_size * sizeof(int), cudaMemcpyHostToDevice, stream4);
            minimum<<<numBlocks, numThreads, 0, stream4>>>(d_arr_4, d_out_4, arr_size);

            // Synchronize streams
            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream2);
            cudaStreamSynchronize(stream3);
            cudaStreamSynchronize(stream4);

            // Destroy streams after use
            cudaStreamDestroy(stream1);
            cudaStreamDestroy(stream2);
            cudaStreamDestroy(stream3);
            cudaStreamDestroy(stream4);

            auto end_operations = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end_operations - start_operations).count();

            // Free device memory
            cudaFree(d_arr_1);
            cudaFree(d_out_1);
            cudaFree(d_arr_2);
            cudaFree(d_out_2);
            cudaFree(d_arr_3);
            cudaFree(d_out_3);
            cudaFree(d_arr_4);
            cudaFree(d_out_4);
        }

        // Calculate the average execution time for each operation
        float average_time_operations = total_time / NUM_RUNS;

        // Write the results to the CSV file
        outputFile << arr_size << ","
                   << average_time_operations << std::endl;
    }

    // Close the output file
    outputFile.close();

    return 0;
}
