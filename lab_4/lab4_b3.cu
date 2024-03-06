#include <cuda_runtime.h>
#include <stdio.h>

#define TPB 16  // Thread per block

__constant__ int d_A_const[16 * 16]; // Define constant memory for matrix A
__constant__ int d_B_const[16 * 16]; // Define constant memory for matrix B

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
    int widthA = 4, heightA = 4, widthB = 4;
    int sizeC = widthB * heightA * sizeof(int);

    // Host arrays
    int h_A[16 * 16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int h_B[16 * 16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int* h_C = (int*)malloc(sizeC);

    // Device arrays
    int* d_C;
    cudaMalloc(&d_C, sizeC);

    // Copy matrices to constant memory
    cudaMemcpyToSymbol(d_A_const, h_A, 16 * 16 * sizeof(int));
    cudaMemcpyToSymbol(d_B_const, h_B, 16 * 16 * sizeof(int));

    // Launch kernel
    dim3 threadsPerBlock(TPB, TPB);
    dim3 numBlocks((widthB + TPB - 1) / TPB, (heightA + TPB - 1) / TPB);
    matrix_mul_part_global<<<numBlocks, threadsPerBlock>>>(widthA, heightA, widthB, d_C);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print result
    printf("Matrix C:\n");
    for (int i = 0; i < heightA; i++) {
        for (int j = 0; j < widthB; j++) {
            printf("%d ", h_C[i * widthB + j]);
        }
        printf("\n");
    }

    // Free memory
    free(h_C);
    cudaFree(d_C);

    return 0;
}
