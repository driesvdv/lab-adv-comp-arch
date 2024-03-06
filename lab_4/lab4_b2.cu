#include <cuda_runtime.h>
#include <stdio.h>

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
    int widthA = 4, heightA = 4, widthB = 4;
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

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(TPB, TPB);
    dim3 numBlocks((widthB + TPB - 1) / TPB, (heightA + TPB - 1) / TPB);
    matrix_mul_part_shared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, widthA, heightA, widthB);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print input matrices
    printf("Matrix A:\n");
    for (int i = 0; i < heightA; i++) {
        for (int j = 0; j < widthA; j++) {
            printf("%d ", h_A[i * widthA + j]);
        }
        printf("\n");
    }

    printf("Matrix B:\n");
    for (int i = 0; i < widthA; i++) {
        for (int j = 0; j < widthB; j++) {
            printf("%d ", h_B[i * widthB + j]);
        }
        printf("\n");
    }

    // Print result
    printf("Matrix C:\n");
    for (int i = 0; i < heightA; i++) {
        for (int j = 0; j < widthB; j++) {
            printf("%d ", h_C[i * widthB + j]);
        }
        printf("\n");
    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
