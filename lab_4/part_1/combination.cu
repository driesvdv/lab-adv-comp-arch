#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

#define M 941     // VR width
#define N 704     // VR height
#define C 3       // Colors
#define OFFSET 15 // Header length

// Function to read image data from file
uint8_t *get_image_array(void)
{
    FILE *imageFile;
    imageFile = fopen("./../images/input_image.ppm", "rb");
    if (imageFile == NULL)
    {
        perror("ERROR: Cannot open input file");
        exit(EXIT_FAILURE);
    }

    uint8_t *image_array = (uint8_t *)malloc(M * N * C * sizeof(uint8_t) + OFFSET);

    fread(image_array, sizeof(uint8_t), M * N * C * sizeof(uint8_t) + OFFSET, imageFile);

    fclose(imageFile);

    return image_array + OFFSET;
}

// Function to save image data to file
void save_image_array(uint8_t *image_array, const std::string& filename)
{
    std::ofstream imageFile("./images/" + filename, std::ios::binary);
    if (!imageFile.is_open())
    {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    imageFile << "P6\n";          // P6 filetype
    imageFile << M << " " << N << "\n"; // dimensions
    imageFile << "255\n";         // Max pixel

    imageFile.write(reinterpret_cast<char*>(image_array), M * N * C);

    imageFile.close();
}

// CUDA kernel for grayscale conversion using coalesced memory access
__global__ void coalesced_memory_access(uint8_t *image, uint8_t * image_out, int numPixels)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPixels)
    {
        image_out[idx] = (image[idx * C] + image[idx * C + 1] + image[idx * C + 2]) / 3;
    }
}

// CUDA kernel for grayscale conversion using non-coalesced memory access
__global__ void non_coalesced_memory_access(uint8_t *image_in, uint8_t *image_out, int numPixels)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPixels)
    {
        image_out[idx] = (image_in[idx] + image_in[idx + numPixels] + image_in[idx + numPixels * 2]) / 3;
    }
}

// Reorder RGB values in array: r0, r1, r2, g1, g2, g3, b1, b2, b3,...
void reorder_rgb(uint8_t *image_array_in, uint8_t *image_array_out, int num_pixels)
{
    for (int idx = 0; idx < num_pixels; ++idx)
    {
        int in_idx = idx * 3;
        int out_idx_r = idx;
        int out_idx_g = idx + num_pixels;
        int out_idx_b = idx + 2 * num_pixels;

        image_array_out[out_idx_r] = image_array_in[in_idx];
        image_array_out[out_idx_g] = image_array_in[in_idx + 1];
        image_array_out[out_idx_b] = image_array_in[in_idx + 2];
    }
}

// Recreate full RGB image from grayscale value array by repeating grayscale value 3 times
void recreate_rgb(uint8_t *image_array_in, uint8_t *image_array_out, int num_pixels)
{
    for (int idx = 0; idx < num_pixels; ++idx)
    {
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
    std::ofstream file("timing_data_comparison.csv");
    file << "Method, Threads, Block Size, Average Execution Time (µs)\n";

    // Allocate memory for host arrays
    uint8_t *h_image_array = get_image_array();
    uint8_t *h_image_array_reordered = (uint8_t *)malloc(M * N * C * sizeof(uint8_t));
    uint8_t *h_image_array_grayscale_coalesced = (uint8_t *)malloc(M * N * sizeof(uint8_t));
    uint8_t *h_image_array_recreated_coalesced = (uint8_t *)malloc(M * N * C * sizeof(uint8_t));
    uint8_t *h_image_array_grayscale = (uint8_t *)malloc(M * N * sizeof(uint8_t));
    uint8_t *h_image_array_recreated = (uint8_t *)malloc(M * N * C * sizeof(uint8_t));

    // Allocate memory for device arrays
    uint8_t *d_image_array_in;
    uint8_t *d_image_array_out;
    cudaMalloc((void **)&d_image_array_in, M * N * C * sizeof(uint8_t));
    cudaMalloc((void **)&d_image_array_out, M * N * sizeof(uint8_t));

    // Perform runs for different block sizes
    for (int blockSize = 32; blockSize <= 1024; blockSize += 2)
    {
        // Calculate grid dimensions
        int numBlocks = ceil((double)(M * N) / blockSize);

        // Variables to accumulate execution times
        float total_duration_coalesced = 0.0f;
        float total_duration_non_coalesced = 0.0f;

        // Perform 100 runs
        for (int i = 0; i < 100; ++i)
        {
            // Copy the image data from host to device and reorder RGB values
            cudaMemset(d_image_array_in, 0, M * N * C * sizeof(uint8_t));
            reorder_rgb(h_image_array, h_image_array_reordered, M * N);
            cudaMemcpy(d_image_array_in, h_image_array_reordered, M * N * C * sizeof(uint8_t), cudaMemcpyHostToDevice);

            // Timing coalesced memory access
            auto start_coalesced = std::chrono::high_resolution_clock::now();
            coalesced_memory_access<<<numBlocks, blockSize>>>(d_image_array_in, d_image_array_out, M * N);
            cudaDeviceSynchronize();
            auto end_coalesced = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::micro> duration_coalesced = end_coalesced - start_coalesced;
            total_duration_coalesced += duration_coalesced.count();

            // Reset device memory and copy original input image to device
            cudaMemset(d_image_array_out, 0, M * N * sizeof(uint8_t));
            cudaMemcpy(d_image_array_in, h_image_array, M * N * C * sizeof(uint8_t), cudaMemcpyHostToDevice);

            // Timing non-coalesced memory access
            auto start_non_coalesced = std::chrono::high_resolution_clock::now();
            non_coalesced_memory_access<<<numBlocks, blockSize>>>(d_image_array_in, d_image_array_out, M * N);
            cudaDeviceSynchronize();
            auto end_non_coalesced = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::micro> duration_non_coalesced = end_non_coalesced - start_non_coalesced;
            total_duration_non_coalesced += duration_non_coalesced.count();
        }

        // Calculate average execution times
        float average_duration_coalesced = total_duration_coalesced / 100.0f;
        float average_duration_non_coalesced = total_duration_non_coalesced / 100.0f;

        // Write average execution times to CSV file
        file << "Coalesced, " << M * N << ", " << blockSize << ", " << average_duration_coalesced << "\n";
        file << "Non-coalesced, " << M * N << ", " << blockSize << ", " << average_duration_non_coalesced << "\n";
    }

    // Free device memory
    cudaFree(d_image_array_in);
    cudaFree(d_image_array_out);

    // Free host memory
    free(h_image_array_reordered);
    free(h_image_array_grayscale_coalesced);
    free(h_image_array_recreated_coalesced);
    free(h_image_array_grayscale);
    free(h_image_array_recreated);

    // Close CSV file
    file.close();

    std::cout << "Done" << std::endl;
    return 0;
}
