#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <fstream>

#define THREADS_PER_BLOCK 16  // Threads per block
#define TILE_SIZE 16  // Size of the shared memory tile

__global__ void matrix_mul_part_shared(int* A, int* B, int* C, int widthA, int heightA, int widthB) {
    // Allocate shared memory for the tiles
    __shared__ int tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ int tile_B[TILE_SIZE][TILE_SIZE];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int row = block_y * TILE_SIZE + thread_y;
    int col = block_x * TILE_SIZE + thread_x;

    int partial_sum = 0;
    for (int tile_index = 0; tile_index < widthA / TILE_SIZE; ++tile_index) {
        // Collaboratively load tiles into shared memory
        tile_A[thread_y][thread_x] = A[row * widthA + tile_index * TILE_SIZE + thread_x];
        tile_B[thread_y][thread_x] = B[(tile_index * TILE_SIZE + thread_y) * widthB + col];

        __syncthreads();

        // Compute the partial result of the block
        for (int k = 0; k < TILE_SIZE; ++k) {
            partial_sum += tile_A[thread_y][k] * tile_B[k][thread_x];
        }

        __syncthreads();
    }

    // Write the final result to device memory
    if (row < heightA && col < widthB) {
        C[row * widthB + col] = partial_sum;
    }
}

// Function to write matrix to a CSV file
void writeMatrixToCSV(int* matrix, int rows, int cols, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        printf("Error: Unable to open the file.\n");
        return;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i * cols + j];
            if (j < cols - 1) {
                file << ",";
            }
        }
        file << std::endl;
    }

    file.close();
}

int main() {
    FILE *file = fopen("./timings/timing_results_shared_2d.csv", "w");
    if (!file) {
        printf("Error: Unable to open the file.\n");
        return 1;
    }
    fprintf(file, "Matrix_Size,Execution_Time(us)\n");

    const int num_runs = 100;

    for (int size = 2; size <= 512; size *= 2) {
        int widthA = size, heightA = size, widthB = size;
        int sizeA = widthA * heightA * sizeof(int);
        int sizeB = widthB * widthA * sizeof(int);
        int sizeC = widthB * heightA * sizeof(int);

        int* host_A = (int*)malloc(sizeA);
        int* host_B = (int*)malloc(sizeB);
        int* host_C = (int*)malloc(sizeC);

        int* device_A, *device_B, *device_C;
        cudaMalloc(&device_A, sizeA);
        cudaMalloc(&device_B, sizeB);
        cudaMalloc(&device_C, sizeC);

        for (int i = 0; i < size * size; ++i) {
            host_A[i] = 1;
            host_B[i] = i;
        }

        cudaMemcpy(device_A, host_A, sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(device_B, host_B, sizeB, cudaMemcpyHostToDevice);

        double total_duration = 0.0;
        for (int run = 0; run < num_runs; ++run) {
            dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
            dim3 numBlocks((widthB + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (heightA + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

            auto start = std::chrono::high_resolution_clock::now();

            matrix_mul_part_shared<<<numBlocks, threadsPerBlock>>>(device_A, device_B, device_C, widthA, heightA, widthB);
            cudaDeviceSynchronize();

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> duration = end - start;

            total_duration += duration.count();
        }

        double average_duration = total_duration / num_runs;
        fprintf(file, "%d,%.2f\n", size, average_duration);

        // Save matrix for size 8x8
        if (size == 8) {
            cudaMemcpy(host_C, device_C, sizeC, cudaMemcpyDeviceToHost);
            writeMatrixToCSV(host_C, size, size, "./matrices/matrix_shared_2d.csv");
        }

        free(host_A);
        free(host_B);
        free(host_C);
        cudaFree(device_A);
        cudaFree(device_B);
        cudaFree(device_C);
    }

    fclose(file);

    return 0;
}
