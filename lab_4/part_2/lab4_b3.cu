#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

#define TPB 16  // Thread per block

#define MATRIX_SIZE 16 // Maximum matrix size

__constant__ int d_A_const[MATRIX_SIZE * MATRIX_SIZE]; // Define constant memory for matrix A
__constant__ int d_B_const[MATRIX_SIZE * MATRIX_SIZE]; // Define constant memory for matrix B

__global__ void matrix_mul_part_global(int widthA, int heightA, int widthB, int* d_C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tmp = 0;
    for (int i = 0; i < widthA; i++) {
        int A_val = d_A_const[y * widthA + i];
        int B_val = d_B_const[i * widthB + x];
        tmp += A_val * B_val;
    }

    if (y < heightA && x < widthB) {
        d_C[y * widthB + x] = tmp;
    }
}

int main() {
    FILE *file = fopen("timing_results_constant.csv", "w");
    if (!file) {
        printf("Error: Unable to open the file.\n");
        return 1;
    }
    fprintf(file, "Matrix_Size,Execution_Time(us)\n");

    // Number of runs for averaging
    const int num_runs = 100;

    // Loop through each matrix size
    for (int size = 2; size <= MATRIX_SIZE; size *= 2) {
        int widthA = size, heightA = size, widthB = size;
        int sizeC = widthB * heightA * sizeof(int);

        // Host array for matrix C
        int* h_C = (int*)malloc(sizeC);

        // Device array for matrix C
        int* d_C;
        cudaMalloc(&d_C, sizeC);

        // Initialize matrices A and B dynamically
        int* h_A = (int*)malloc(size * size * sizeof(int));
        int* h_B = (int*)malloc(size * size * sizeof(int));
        for (int i = 0; i < size * size; ++i) {
            h_A[i] = 1;
            h_B[i] = i;
        }

        // Copy matrices A and B to constant memory
        cudaMemcpyToSymbol(d_A_const, h_A, size * size * sizeof(int));
        cudaMemcpyToSymbol(d_B_const, h_B, size * size * sizeof(int));

        // Timing accumulator
        double total_duration = 0.0;

        // Loop for averaging
        for (int run = 0; run < num_runs; ++run) {
            // Launch kernel
            dim3 threadsPerBlock(TPB, TPB);
            dim3 numBlocks((widthB + TPB - 1) / TPB, (heightA + TPB - 1) / TPB);

            auto start = std::chrono::high_resolution_clock::now();

            matrix_mul_part_global<<<numBlocks, threadsPerBlock>>>(widthA, heightA, widthB, d_C);

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

        // Copy the result back from device to host
        cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

        // Save the last result to CSV
        if (size == 8) {
            FILE *last_result_file = fopen("last_result_matrix_const.csv", "w");
            if (!last_result_file) {
                printf("Error: Unable to open the file.\n");
                return 1;
            }

            fprintf(last_result_file, "Result Matrix:\n");
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    fprintf(last_result_file, "%d,", h_C[i * size + j]);
                }
                fprintf(last_result_file, "\n");
            }

            fclose(last_result_file);
        }

        // Free memory
        free(h_A);
        free(h_B);
        free(h_C);
        cudaFree(d_C);
    }

    // Close the CSV file
    fclose(file);

    return 0;
}
