#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 941     // VR width
#define N 704     // VR height
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
 * Inverts the rgb color of an image
 */
__global__ void invert_colors_stride(uint8_t *image, int numPixels, int stride)
{
    for (int channel = 0; channel < C; ++channel) {    
        for (size_t k = 0; k < stride; ++k)
        {
            int idx = threadIdx.x + k * blockDim.x; // Correct index calculation with stride
            if (idx < numPixels)
            {
                if(channel == 0)
                {
                    image[idx * C] = (image[idx * C] % 25) * 10; // Calculate red channel
                    //image[idx * C] = 255 - image[idx * C]; // Calculate red channel
                }            
                else if(channel == 1)
                {
                    image[idx * C + 1] = 255 - image[idx * C + 1]; // Calculate green channel
                }    
                else if (channel == 2)
                {
                    image[idx * C + 2] = 255 - image[idx * C + 2]; // Calculate blue channel
                }            
            }
        }
    }
}

int main(void)
{
    // Open CSV file for writing
    std::ofstream file("timing_data_divergence_fix.csv");
    file << "Threads, Execution Time (µs)\n";

    // Read the image
    uint8_t *h_image_array = get_image_array();

    // Calculate total number of pixels
    int numPixels = M * N;

    // Allocate memory on the GPU for the image
    uint8_t *d_image_array;
    cudaMalloc((void **)&d_image_array, numPixels * C * sizeof(uint8_t));

    // Copy the image data from host to device
    cudaMemcpy(d_image_array, h_image_array, numPixels * C * sizeof(uint8_t), cudaMemcpyHostToDevice);

       // Loop for different thread counts
    for (int threads = 1; threads <= 1024; threads *= 2)
    {
        // Launch the kernel to invert colors and measure execution time
        auto start = std::chrono::steady_clock::now();

        invert_colors_stride<<<1, threads>>>(d_image_array, numPixels, numPixels / threads);
        
        cudaDeviceSynchronize();

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Write timing data to CSV file
        file << threads << ", " << duration << std::endl;
    }

    // Allocate memory on the GPU for the image
    cudaMalloc((void **)&d_image_array, numPixels * C * sizeof(uint8_t));

    // Copy the image data from host to device
    cudaMemcpy(d_image_array, h_image_array, numPixels * C * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int blockSize = 128;
    int numBlocks = 1;
    int numStrides = ceil((double)(numPixels * C) / blockSize);
    //int warps = numStrides * ceil((double)(blockSize) / 32);

    // Launch the kernel to invert colors
    invert_colors_stride<<<numBlocks, blockSize>>>(d_image_array, numPixels, numStrides);
    

    cudaDeviceSynchronize();

    // Copy the inverted image data back to host
    cudaMemcpy(h_image_array, d_image_array, numPixels * C * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Save the output image
    save_image_array(h_image_array);

    // Free device memory
    cudaFree(d_image_array);

    // Close CSV file
    file.close();

    printf("Done\n");
    return 0;
}
