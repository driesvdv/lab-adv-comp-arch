#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <fstream>

#define TPB 16  // Thread per block
#define MATRIX_SIZE 64   // Maximum matrix size

__constant__ int d_A_const[MATRIX_SIZE * MATRIX_SIZE]; // Define constant memory for matrix A
__constant__ int d_B_const[MATRIX_SIZE * MATRIX_SIZE]; // Define constant memory for matrix B

__global__ void matrix_mul_part_global(int widthA, int heightA, int widthB, int* d_C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = bx * TPB + tx;
    int y = by * TPB + ty;

    if (y < heightA && x < widthB) {
        int tmp = 0;
        for (int i = 0; i < widthA; ++i) {
            tmp += d_A_const[y * widthA + i] * d_B_const[i * widthB + x];
        }
        d_C[y * widthB + x] = tmp;
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
    FILE *file = fopen("./timings/timing_results_constant.csv", "w");
    if (!file) {
        printf("Error: Unable to open the file.\n");
        return 1;
    }
    fprintf(file, "Matrix_Size,Execution_Time(us)\n");

    const int num_runs = 100;

    for (int size = 2; size <= MATRIX_SIZE; size *= 2) {
        int widthA = size, heightA = size, widthB = size;
        int sizeC = widthB * heightA * sizeof(int);

        int* h_C = (int*)malloc(sizeC);

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

        double total_duration = 0.0;
        for (int run = 0; run < num_runs; ++run) {
            dim3 threadsPerBlock(TPB, TPB);
            dim3 numBlocks((widthB + TPB - 1) / TPB, (heightA + TPB - 1) / TPB);

            auto start = std::chrono::high_resolution_clock::now();

            matrix_mul_part_global<<<numBlocks, threadsPerBlock>>>(widthA, heightA, widthB, d_C);
            cudaDeviceSynchronize();

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> duration = end - start;

            total_duration += duration.count();
        }

        double average_duration = total_duration / num_runs;
        fprintf(file, "%d,%.2f\n", size, average_duration);

        // Save matrix for size 8x8
        if (size == 8) {
            cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
            writeMatrixToCSV(h_C, size, size, "./matrices/matrix_constant.csv");
        }

        free(h_A);
        free(h_B);
        free(h_C);
        cudaFree(d_C);
    }

    fclose(file);

    return 0;
}
