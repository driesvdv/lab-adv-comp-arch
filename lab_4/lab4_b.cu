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


template<typename T>
__global__ void naive_matrix_multiply(const T *A, const T *B, T* C, int width, int P, int Q)
{
  int r = blockIdx.y * blockDim.y + threadIdx.y;   
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  // check boundry conditions
  if( r < P && c < Q){
    // do the multiplication for one row and col
    T value = 0;
    for(int k = 0; k < width; k++){
      value += A[r * width + k] * B[k * Q + c];
    }
    // store the result
    C[r * Q + c] = value;
  }}

int main(void){
    // Matrix size
    int N = 4;
    size_t bytes = N * N * sizeof(int);

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

    // Print matrices
    printf("Matrix A\n");
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            printf("%d ", h_A[i * N + j]);
        }
        printf("\n");
    }
    
    printf("\n\n");
    printf("Matrix B\n");
    for(size_t i = 0; i < N; i++){
        for (size_t j = 0; j < N; j++)
        {
            printf("%d ", h_B[i * N + j]);
        }
        printf("\n");
    }
    printf("\n\n");

    // Transfer data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 threads(2, 2);
    dim3 blocks(N / threads.x, N / threads.y);
    matrixMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Transfer data back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Print result
    printf("Matrix C\n");
    for(size_t i = 0; i < N; i++){
        for (size_t j = 0; j < N; j++)
        {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }
    printf("\n\n");


    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}