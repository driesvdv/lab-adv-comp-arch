#include <chrono>
#include <cstdio>
#include <stdexcept>

#define ARR_SIZE 10

/**
 * CUDA device code for performing element wise summation opperation to array
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
 * CUDA device code for performing element wise multiplication to array
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
 *
 * Reduction algorithm with sequential adressing is used to find the maximum value in the array
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
    // Seed the random number generator
    srand(time(NULL));

    // Initialize the arrays
    int arr_1[ARR_SIZE];
    int arr_2[ARR_SIZE];
    int arr_3[ARR_SIZE];
    int arr_4[ARR_SIZE];

    // Fill the arrays with random numbers
    for (int i = 0; i < ARR_SIZE; i++)
    {
        arr_1[i] = rand() % 1000;
        arr_2[i] = rand() % 1000;
        arr_3[i] = rand() % 1000;
        arr_4[i] = rand() % 1000;
    }

    // Print first 100 elements of arr_3
    for (int i = 0; i < 10; i++)
    {
        fprintf(stdout, "%d ", arr_3[i]);
    }

    // Allocate memory on the device
    int *d_arr_1;
    int *d_arr_2;
    int *d_arr_3;
    int *d_arr_4;

    cudaMalloc((void **)&d_arr_1, ARR_SIZE * sizeof(int));
    cudaMalloc((void **)&d_arr_2, ARR_SIZE * sizeof(int));
    cudaMalloc((void **)&d_arr_3, ARR_SIZE * sizeof(int));
    cudaMalloc((void **)&d_arr_4, ARR_SIZE * sizeof(int));

    int *d_out_1;
    int *d_out_2;
    int *d_out_3;
    int *d_out_4;

    cudaMalloc((void **)&d_out_1, sizeof(int));
    cudaMalloc((void **)&d_out_2, sizeof(int));
    cudaMalloc((void **)&d_out_3, sizeof(int));
    cudaMalloc((void **)&d_out_4, sizeof(int));

    // Copy the arrays to the device
    cudaMemcpy(d_arr_1, arr_1, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_2, arr_2, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_3, arr_3, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_4, arr_4, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate number of blocks and threads
    int numThreads = ARR_SIZE < 2048 ? ARR_SIZE / 2 : 1024;
    int numBlocks = (ARR_SIZE + numThreads - 1) / numThreads;

    if (numThreads > 1024)
    {
        throw std::runtime_error("Array size is too large for current implementation. Maximum array size is 1024 elements.");
    }

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the operations
    summation<<<numBlocks, numThreads>>>(d_arr_1, d_out_1, ARR_SIZE);
    multiplication<<<numBlocks, numThreads>>>(d_arr_2, d_out_2, ARR_SIZE);
    maximum<<<numBlocks, numThreads>>>(d_arr_3, d_out_3, ARR_SIZE);
    minimum<<<numBlocks, numThreads>>>(d_arr_4, d_out_4, ARR_SIZE);

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Print the time
    std::chrono::duration<float, std::milli> duration = end - start;

    fprintf(stdout, "Execution time synchronous approach: %f ms\n", duration.count());

    // Copy the results back to the host
    int out_1;
    int out_2;
    int out_3;
    int out_4;

    cudaMemcpy(&out_1, d_out_1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_2, d_out_2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_3, d_out_3, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_4, d_out_4, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy back input arrays
    cudaMemcpy(arr_1, d_arr_1, ARR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(arr_2, d_arr_2, ARR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(arr_3, d_arr_3, ARR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(arr_4, d_arr_4, ARR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(d_arr_1);
    cudaFree(d_arr_2);
    cudaFree(d_arr_3);
    cudaFree(d_arr_4);

    cudaFree(d_out_1);
    cudaFree(d_out_2);
    cudaFree(d_out_3);
    cudaFree(d_out_4);

    return 0;
}