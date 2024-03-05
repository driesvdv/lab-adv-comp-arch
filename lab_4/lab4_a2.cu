#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 3//941     // VR width
#define N 3//704     // VR height
#define C 3       // Colors
#define OFFSET 15 // Header length

uint8_t *get_image_array(void)
{
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
    imageFile = fopen("./images/input_image.ppm", "rb");
    if (imageFile == NULL)
    {
        perror("ERROR: Cannot open input file");
        exit(EXIT_FAILURE);
    }

    // Initialize empty image array
    uint8_t *image_array = (uint8_t *)malloc(M * N * C * sizeof(uint8_t) + OFFSET);

    // Read the image
    fread(image_array, sizeof(uint8_t), M * N * C * sizeof(uint8_t) + OFFSET, imageFile);

    // Close the file
    fclose(imageFile);

    // Move the starting pointer and return the flattened image array
    return image_array + OFFSET;
}

void save_image_array(uint8_t *image_array)
{
    /*
     * Save the data of an (RGB) image as a pixel map.
     *
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     *
     */
    // Try opening the file
    FILE *imageFile;
    imageFile = fopen("./output_image.ppm", "wb");
    if (imageFile == NULL)
    {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Configure the file
    fprintf(imageFile, "P6\n");          // P6 filetype
    fprintf(imageFile, "%d %d\n", M, N); // dimensions
    fprintf(imageFile, "255\n");         // Max pixel

    // Write the image
    fwrite(image_array, 1, M * N * C, imageFile);

    // Close the file
    fclose(imageFile);
}

/**
 * CUDA Kernel Device code
 *
 * Makes the image grayscale by using the average RGB method using non coalesced memory access
 */
__global__ void non_coalesced_memory_acces(uint8_t *image_in, uint8_t *image_out, uint8_t numPixels)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPixels)
    {
        // calculate the average grayscale value
        image_out[idx] = (image_in[idx] + image_in[idx + numPixels] + image_in[idx + numPixels*2]) / 3;
    }
}

// Reorder the image array to have all r values first, then all g values, then all b values instead of r0, g0, b0, r1, g1, b1, ...
void reorder_rgb(uint8_t* image_array_in, uint8_t* image_array_out, int num_pixels) {
    for (int idx = 0; idx < num_pixels; ++idx) {
        int in_idx = idx * 3;
        int out_idx_r = idx;
        int out_idx_g = idx + num_pixels;
        int out_idx_b = idx + 2 * num_pixels;

        image_array_out[out_idx_r] = image_array_in[in_idx];
        image_array_out[out_idx_g] = image_array_in[in_idx + 1];
        image_array_out[out_idx_b] = image_array_in[in_idx + 2];
    }
}

// Recreate full rgb image from the grayscale value array by repeating the grayscale value 3 times
void recreate_rgb(uint8_t* image_array_in, uint8_t* image_array_out, int num_pixels) {
    for (int idx = 0; idx < num_pixels; ++idx) {
        int in_idx = idx;
        int out_idx_r = idx * 3;
        int out_idx_g = idx * 3 + 1;
        int out_idx_b = idx * 3 + 2;

        image_array_out[out_idx_r] = image_array_in[in_idx];
        image_array_out[out_idx_g] = image_array_in[in_idx];
        image_array_out[out_idx_b] = image_array_in[in_idx];
    }
}


int main(void)
{
    // Open CSV file for writing
    std::ofstream file("timing_data_divergence_fix.csv");
    file << "Threads, Execution Time (µs)\n";

    // Read the image
    //uint8_t *h_image_array = get_image_array();
    // Fake 9 long array as image
    uint8_t h_image_array[27] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};

    // Reorder the image array to have all r values first, then all g values, then all b values instead of r0, g0, b0, r1, g1, b1, ...
    uint8_t *h_image_array_reordered = (uint8_t *)malloc(M * N * C * sizeof(uint8_t));
    uint8_t *h_image_array_grayscale = (uint8_t *)malloc(M * N * sizeof(uint8_t));

    reorder_rgb(h_image_array, h_image_array_reordered, M * N);

    // Print reordered image array
    // printf("Reordered image array: ");
    // for (int i = 0; i < 1000; i++) {
    //     printf("%d ", h_image_array_reordered[i]);
    // }
    // printf("\n\n");

    // Calculate total number of pixels
    int numPixels = M * N;

    // Allocate memory on the GPU for the image
    uint8_t *d_image_array_in;
    uint8_t *d_image_array_out;


    // Allocate memory on the GPU for the image
    cudaMalloc((void **)&d_image_array_in, numPixels * C * sizeof(uint8_t));
    cudaMalloc((void **)&d_image_array_out, numPixels * sizeof(uint8_t));    


    // Copy the image data from host to device
    cudaMemcpy(d_image_array_in, h_image_array_reordered, numPixels * C * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int blockSize = 512;
    int numBlocks = ceil((double)(numPixels) / blockSize);
    // int warps = numStrides * ceil((double)(blockSize) / 32);

    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel to grayscale image using non coalesced memory access
    non_coalesced_memory_acces<<<numBlocks, blockSize>>>(d_image_array_in, d_image_array_out, numPixels);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::micro> duration = end - start;
    printf("Coalesced execution Time: %f µs\n", duration.count());

    // Copy the inverted image data back to host
    cudaMemcpy(h_image_array_grayscale, d_image_array_out, numPixels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Print image array
    printf("Grayscale image array: ");
    for (int i = 0; i<numPixels; i++) {
        printf("%d ", h_image_array_grayscale[i]);
    }
    printf("\n\n");

    // Recreate full rgb image from the grayscale value array by repeating the grayscale value 3 times
    uint8_t *h_image_array_recreated = (uint8_t *)malloc(M * N * C * sizeof(uint8_t));
    recreate_rgb(h_image_array_grayscale, h_image_array, M * N);

    // Print recreated image array
    printf("Recreated image array: ");
    for (int i = 0; i < numPixels * C; i++) {
        printf("%d ", h_image_array[i]);
    }
    printf("\n\n");

    // Save the output image
    save_image_array(h_image_array);

    // Free device memory
    cudaFree(d_image_array_in);
    cudaFree(d_image_array_out);

    // Close CSV file
    file.close();

    printf("Done\n");
    return 0;
}



