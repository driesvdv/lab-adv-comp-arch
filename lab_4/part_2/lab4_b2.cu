#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

#define TPB 4  // Thread per block

__global__ void matrix_mul_part_shared(int* A, int* B, int* C, int widthA, int heightA, int widthB) {
    // Define arrays in the shared memory
    __shared__ int sA[TPB][TPB];
    __shared__ int sB[TPB][TPB];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tmp = 0;
    for (int i = 0; i < widthA / TPB; i++) {
        // Preload data into shared memory
        sA[threadIdx.y][threadIdx.x] = 0;
        sB[threadIdx.y][threadIdx.x] = 0;
        if (y < heightA && (threadIdx.x + i * TPB) < widthA) {
            sA[threadIdx.y][threadIdx.x] = A[y * widthA + threadIdx.x + i * TPB];
        }
        if (x < widthB && (threadIdx.y + i * TPB) < heightA) {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + i * TPB) * widthB + x];
        }

        // Synchronize to ensure all threads finish preloading
        __syncthreads();

        // Compute partial product on shared memory
        for (int j = 0; j < TPB; j++) {
            tmp += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }

        // Synchronize to ensure all threads finish computing
        __syncthreads();
    }

    if (y < heightA && x < widthB) {
        C[y * widthB + x] = tmp;
    }
}

int main() {
    FILE *file = fopen("timing_results_sharded_mem.csv", "w");
    if (!file) {
        printf("Error: Unable to open the file.\n");
        return 1;
    }
    fprintf(file, "Matrix_Size,Execution_Time(us)\n");

    // Number of runs for averaging
    const int num_runs = 100;

    // Loop through each matrix size
    for (int size = 2; size <= 512; size *= 2) {
        int widthA = size, heightA = size, widthB = size;
        int sizeA = widthA * heightA * sizeof(int);
        int sizeB = widthB * widthA * sizeof(int);
        int sizeC = widthB * heightA * sizeof(int);

        // Host arrays
        int* h_A = (int*)malloc(sizeA);
        int* h_B = (int*)malloc(sizeB);
        int* h_C = (int*)malloc(sizeC);

        // Device arrays
        int* d_A, *d_B, *d_C;
        cudaMalloc(&d_A, sizeA);
        cudaMalloc(&d_B, sizeB);
        cudaMalloc(&d_C, sizeC);

        // Initialize matrices
        for (int i = 0; i < heightA; i++) {
            for (int j = 0; j < widthA; j++) {
                h_A[i * widthA + j] = 1;
            }
        }

        for (int i = 0; i < widthA; i++) {
            for (int j = 0; j < widthB; j++) {
                h_B[i * widthB + j] = i*widthB + j;
            }
        }

        // Timing accumulator
        double total_duration = 0.0;

        // Loop for averaging
        for (int run = 0; run < num_runs; ++run) {
            // Copy matrices to device
            cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

            // Launch kernel
            dim3 threadsPerBlock(TPB, TPB);
            dim3 numBlocks((widthB + TPB - 1) / TPB, (heightA + TPB - 1) / TPB);

            auto start = std::chrono::high_resolution_clock::now();

            matrix_mul_part_shared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, widthA, heightA, widthB);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> duration = end - start;

            // Accumulate timing results
            total_duration += duration.count();

            // Synchronize to ensure kernel execution is completed
            cudaDeviceSynchronize();
        }

        // Calculate average timing results
        double average_duration = total_duration / num_runs;

        // Write average timing results to CSV file
        fprintf(file, "%d,%.2f\n", size, average_duration);

        // Free memory
        free(h_A);
        free(h_B);
        free(h_C);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    // Close the CSV file
    fclose(file);

    return 0;
}
