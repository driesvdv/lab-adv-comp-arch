#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

/**
 * CUDA kernel matrix multiplication
*/
__global__ void matrixMulKernel(int* d_A, int* d_B, int* d_C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N){
        int sum = 0;
        for(int i = 0; i < N; i++){
            sum += d_A[row * N + i] * d_B[i * N + col];
        }
        d_C[row * N + col] = sum;
    }
}


int main(void){
    // Open CSV file to write timing results
    std::ofstream file("timing_results.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the file." << std::endl;
        return 1;
    }

    // Write CSV header
    file << "Matrix_Size, Execution_Time" << std::endl;

    // Matrix sizes to test
    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512};

    // Number of runs for averaging
    const int num_runs = 100;

    // Loop through each matrix size
    for (int size : sizes) {
        // Matrix size
        int N = size;
        size_t bytes = N * N * sizeof(int);

        // Timing accumulator
        double total_duration = 0.0;

        // Loop for averaging
        for (int run = 0; run < num_runs; ++run) {
            // Host memory
            int* h_A, *h_B, *h_C;
            h_A = (int*)malloc(bytes);
            h_B = (int*)malloc(bytes);
            h_C = (int*)malloc(bytes);

            // Device memory
            int* d_A, *d_B, *d_C;
            cudaMalloc(&d_A, bytes);
            cudaMalloc(&d_B, bytes);
            cudaMalloc(&d_C, bytes);

            // Initialize matrices
            for(int i = 0; i < N * N; i++){
                h_A[i] = 1;
                h_B[i] = i;
            }

            // Transfer data to device
            cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

            // Kernel launch
            dim3 threads(2, 2);
            dim3 blocks(N / threads.x, N / threads.y);

            auto start = std::chrono::high_resolution_clock::now();
            matrixMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate execution time in microseconds
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            // Accumulate timing results
            total_duration += duration;

            // Free memory
            free(h_A);
            free(h_B);
            free(h_C);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        }

        // Calculate average timing results
        double average_duration = total_duration / num_runs;

        // Write average timing results to CSV file
        file << N << ", " << average_duration << std::endl;
    }

    // Close the CSV file
    file.close();

    return 0;
}
