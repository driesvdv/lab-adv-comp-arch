#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 941       // VR width
#define N 704       // VR height
#define C 3         // Colors
#define OFFSET 15   // Header length

uint8_t* get_image_array(void){
    /*
     * Get the data of an (RGB) image as a 1D array.
     * 
     * Returns: Flattened image array.
     * 
     * Noets:
     *  - Images data is flattened per color, column, row.
     *  - The first 3 data elements are the RGB components
     *  - The first 3*M data elements represent the firts row of the image
     *  - For example, r_{0,0}, g_{0,0}, b_{0,0}, ..., b_{0,M}, r_{1,0}, ..., b_{b,M}, ..., b_{N,M}
     * 
     */        
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./images/input_image.ppm","rb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open input file");
        exit(EXIT_FAILURE);
    }
    
    // Initialize empty image array
    uint8_t* image_array = (uint8_t*)malloc(M*N*C*sizeof(uint8_t)+OFFSET);
    
    // Read the image
    fread(image_array, sizeof(uint8_t), M*N*C*sizeof(uint8_t)+OFFSET, imageFile);
    
    // Close the file
    fclose(imageFile);
        
    // Move the starting pointer and return the flattened image array
    return image_array + OFFSET;
}

void save_image_array(uint8_t* image_array){
    /*
     * Save the data of an (RGB) image as a pixel map.
     * 
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     * 
     */            
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./output_image.ppm","wb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    // Configure the file
    fprintf(imageFile,"P6\n");               // P6 filetype
    fprintf(imageFile,"%d %d\n", M, N);      // dimensions
    fprintf(imageFile,"255\n");              // Max pixel
    
    // Write the image
    fwrite(image_array, 1, M*N*C, imageFile);
    
    // Close the file
    fclose(imageFile);
}

/**
 * CUDA Kernel Device code
 *
 * Inverts the rgb color of an image
 */
__global__ void invert_colors(uint8_t *image, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        int startIdx = idx * C;
        for (int i = 0; i < C; i++) {
            image[startIdx + i] = 255 - image[startIdx + i]; // Invert each color channel
        }
    }
}


/**
 * CUDA Kernel Device code
 *
 * Inverts the rgb color of an image
 */
__global__ void invert_colors_stride(uint8_t *image, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        int startIdx = idx * C;
        for (int i = 0; i < C; i++) {
            image[startIdx + i] = 255 - image[startIdx + i]; // Invert each color channel
        }
    }
}


int main(void)
{
    cudaError_t err = cudaSuccess;

    // Read the image
    uint8_t* h_image_array = get_image_array();

    // Calculate total number of pixels
    int numPixels = M * N;

    // Allocate memory on the GPU for the image
    uint8_t *d_image_array;
    cudaMalloc((void**)&d_image_array, numPixels * C * sizeof(uint8_t));

    // Copy the image data from host to device
    cudaMemcpy(d_image_array, h_image_array, numPixels * C * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int blockSize = 33;
    int numBlocks = (numPixels + blockSize - 1) / blockSize;

    // Init timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel to invert colors
    invert_colors<<<numBlocks, blockSize>>>(d_image_array, numPixels);

    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    // Copy the inverted image data back to host
    cudaMemcpy(h_image_array, d_image_array, numPixels * C * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Save the output image
    save_image_array(h_image_array);

    // Free device memory
    cudaFree(d_image_array);

    // Print results
    int warps = numBlocks * ceil((double)(blockSize) / 32);

    printf("Image size: %d x %d\n", M, N);
    printf("Warps used: %d\n", warps);
    printf("Blocks used: %d\n", numBlocks);
    printf("Threads per block: %d\n", blockSize);
    printf("Time: %f ms\n", duration.count());


    printf("Done\n");
    return 0;
}
