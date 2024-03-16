#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <fstream>

#define TPB 16  // Thread per block

__global__ void matrix_mul_part_global(int* A, int* B, int* C, int widthA, int heightA, int widthB) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int x = bx * TPB + tx;
    int y = by * TPB + ty;

    if (y < heightA && x < widthB) {
        int tmp = 0;
        for (int i = 0; i < widthA; i++) {
            tmp += A[y * widthA + i] * B[i * widthB + x];
        }
        C[y * widthB + x] = tmp;
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
    FILE *file = fopen("./timings/timing_results_global_2d.csv", "w");
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

        int* h_A = (int*)malloc(sizeA);
        int* h_B = (int*)malloc(sizeB);
        int* h_C = (int*)malloc(sizeC);

        int* d_A, *d_B, *d_C;
        cudaMalloc(&d_A, sizeA);
        cudaMalloc(&d_B, sizeB);
        cudaMalloc(&d_C, sizeC);

        for (int i = 0; i < size * size; ++i) {
            h_A[i] = 1;
            h_B[i] = i;
        }

        cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

        double total_duration = 0.0;
        for (int run = 0; run < num_runs; ++run) {
            dim3 threadsPerBlock(TPB, TPB);
            dim3 numBlocks((widthB + TPB - 1) / TPB, (heightA + TPB - 1) / TPB);

            auto start = std::chrono::high_resolution_clock::now();

            matrix_mul_part_global<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, widthA, heightA, widthB);
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
            writeMatrixToCSV(h_C, size, size, "./matrices/matrix_global_2d.csv");
        }

        free(h_A);
        free(h_B);
        free(h_C);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    fclose(file);

    return 0;
}
