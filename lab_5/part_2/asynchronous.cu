#include <chrono>
#include <cstdio>
#include <stdexcept>

#define ARR_SIZE 100000

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
    // Warm up the GPU
    int *d_arr_x;
    cudaMalloc((void **)&d_arr_x, 1 * sizeof(int));

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

    // Calculate number of blocks and threads
    int numThreads = (ARR_SIZE + 1) / 2;
    int numThreadsMax = 1024;
    numThreads = numThreads > numThreadsMax ? numThreadsMax : numThreads;
    int numBlocks = (ARR_SIZE + numThreads - 1) / numThreads;

    if (numThreads > 1024)
    {
        throw std::runtime_error("Array size is too large for current implementation. Maximum array size is 1024 elements.");
    }

    /// Create CUDA streams
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the operations with streams
    cudaMemcpyAsync(d_arr_1, arr_1, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream1);
    summation<<<numBlocks, numThreads, 0, stream1>>>(d_arr_1, d_out_1, ARR_SIZE);

    cudaMemcpyAsync(d_arr_2, arr_2, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream2);
    multiplication<<<numBlocks, numThreads, 0, stream2>>>(d_arr_2, d_out_2, ARR_SIZE);

    cudaMemcpyAsync(d_arr_3, arr_3, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream3);
    maximum<<<numBlocks, numThreads, 0, stream3>>>(d_arr_3, d_out_3, ARR_SIZE);

    cudaMemcpyAsync(d_arr_4, arr_4, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream4);
    minimum<<<numBlocks, numThreads, 0, stream4>>>(d_arr_4, d_out_4, ARR_SIZE);

    // Wait for all streams to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Destroy the streams after use
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);

    // Wait for the device to finish
    cudaDeviceSynchronize();

    // Copy the results back to the host asynchronously
    int out_1;
    int out_2;
    int out_3;
    int out_4;

    cudaMemcpyAsync(&out_1, d_out_1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&out_2, d_out_2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&out_3, d_out_3, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&out_4, d_out_4, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the time
    std::chrono::duration<float, std::micro> duration = end - start;

    fprintf(stdout, "Execution time asynchronous approach: %f µs\n", duration.count());

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