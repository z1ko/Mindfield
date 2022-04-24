
#include <random>
#include <iostream>
#include <config.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

// Controlla possibili errori CUDA
#define CUDA_CHECK(f) { cuda_check_error((f), __FILE__, __LINE__); }
inline void cuda_check_error(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        
        std::cerr << "Errore CUDA nel file " << file << " a linea " << line << ":\n"
        << cudaGetErrorString(code) << std::endl;
        
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// Controlla possibile errore dopo l'esecuzione di un kernel
#define CUDA_CHECK_KERNEL(k) { (k); cuda_check_error(cudaGetLastError(), __FILE__, __LINE__); }

// Linearizza l'accesso 2D ad una matrice
__host__ __device__ inline int linear(int x, int y, int stride) {
    return x + y * stride;
}

// Assegna un valore RGB
__host__ __device__ inline int rgb(unsigned char r, unsigned char g, unsigned char b) {
    return (static_cast<unsigned int>(r) << 8 * 3)
         | (static_cast<unsigned int>(g) << 8 * 2)
         | (static_cast<unsigned int>(b) << 8 * 1)
         | 0xFF;
}

const int N = 1600;
const int BLOCK_SIZE = 16;
const int GRID_SIZE = N / BLOCK_SIZE;

// Date due matrici A e B (NxN) ritorna AB (NxN) 
__global__ void multiply(int const * const A, int const * const B, int * R, int N) {

    // Posizione all'interno della tile
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    // Posizione nella matrice
    const int gx = lx + blockIdx.x * blockDim.x;
    const int gy = ly + blockIdx.y * blockDim.y;

    // Iteriamo su tutte le tile necessarie accumulando parzialmente
    int partial_sum = 0;
    for (int s = 0; s < GRID_SIZE; s++) {

        __shared__ int subA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int subB[BLOCK_SIZE][BLOCK_SIZE];

        // Copia in memoria condivisa il contenuto delle tile di A e B
        // che ci interessano immediatamente.
        subA[ly][lx] = A[linear(s * BLOCK_SIZE + lx, gy, N)];
        subB[ly][lx] = B[linear(gx, s * BLOCK_SIZE + ly, N)];
        
        __syncthreads();

        // Somma parziale del loro contributo
        for (int e = 0; e < BLOCK_SIZE; e++)
            partial_sum += subA[ly][e] * subB[e][lx];

        __syncthreads();
    }

    // Scrittura finale sulla memoria globale
    R[linear(gx, gy, N)] = partial_sum;
}

// Genera immagine
__global__ void generate_image(int* result, int N) {

    // Posizione all'interno della tile
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    // Posizione nella matrice
    const int gx = lx + blockIdx.x * blockDim.x;
    const int gy = ly + blockIdx.y * blockDim.y;

    result[linear(gx, gy, N)] = 0xFFFFFFFF;
}

int main(int argc, char** argv) {

    std::cout << "Versione: " 
              << MINDFIELD_VERSION_MAJOR 
              << "." 
              << MINDFIELD_VERSION_MINOR 
              << std::endl;

    int* device_image_buffer;
    CUDA_CHECK(cudaMalloc(&device_image_buffer, sizeof(int) * N * N));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 bgrid(N / block.x, N / block.y);

    std::cout << "Grid[" << bgrid.x << ", " << bgrid.y << "] " 
              << "of " << bgrid.x * bgrid.y << " blocks[" << block.x << ", " << block.y << "] "
              << "of " << block.x * block.y << " threads each.\n";

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    generate_image<<<bgrid, block>>>(device_image_buffer, N);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaGetLastError());

    int* host_image_buffer = new int[N * N];
    CUDA_CHECK(cudaMemcpy(host_image_buffer, device_image_buffer, 
        sizeof(int) * N * N, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Tempo di esecuzione kernel: " << milliseconds << "ms" << std::endl;
    
    CUDA_CHECK(cudaFree(device_image_buffer));
    CUDA_CHECK(cudaDeviceReset());

    stbi_write_bmp("output.bmp", N, N, 4, host_image_buffer);
    delete[] host_image_buffer;

    return EXIT_SUCCESS;
}
