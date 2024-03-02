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
    for (size_t k = 0; k < stride; ++k)
    {
        int idx = threadIdx.x + k * blockDim.x; // Correct index calculation with stride
        if (idx < numPixels)
        {
            if (idx % 3 == 0) // Recalculate red channel
            {
                image[idx] = (image[idx] % 25) * 10;
            }
            else{ //Invert green and blue channels
                image[idx] = 255 - image[idx];
            }
        }
    }
}


int main(void)
{
    // Open CSV file for writing
    std::ofstream file("timing_data_thread_divergence_problem.csv");
    file << "Threads, Execution Time (µs)\n";

    cudaError_t err = cudaSuccess;

    // Read the image
    uint8_t *h_image_array = get_image_array();
    int numPixels = M * N * C;

    // Allocate memory on the GPU for the image
    uint8_t *d_image_array;
    cudaMalloc((void **)&d_image_array, numPixels * sizeof(uint8_t));
    cudaMemcpy(d_image_array, h_image_array, numPixels * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Define maximum number of threads to test
    int maxThreads = 1024;
    for (int threads = 1; threads <= maxThreads; threads *= 2)
    {
        // Start the timer
        auto start = std::chrono::steady_clock::now();

        // Launch the kernel to invert colors
        invert_colors_stride<<<1, threads>>>(d_image_array, numPixels, numPixels / threads);

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // Stop the timer
        auto end = std::chrono::steady_clock::now();

        // Calculate duration in milliseconds
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Write results to CSV file
        file << threads << ", " << duration << std::endl;
    }


    // Allocate memory on the GPU for the image
    cudaMalloc((void **)&d_image_array, numPixels * sizeof(uint8_t));
    cudaMemcpy(d_image_array, h_image_array, numPixels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    // Call function with 256 threads
    int numStrides = ceil((double)(numPixels) / 256);
    cudaDeviceSynchronize();
    invert_colors_stride<<<1, 256>>>(d_image_array, numPixels, numStrides);
    cudaDeviceSynchronize();

    // Copy the inverted image data back to host if needed
    cudaMemcpy(h_image_array, d_image_array, numPixels * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    // Save the output image if needed
    save_image_array(h_image_array);

    // Free device memory
    cudaFree(d_image_array);

    // Close CSV file
    file.close();

    std::cout << "Done" << std::endl;
    return 0;
}