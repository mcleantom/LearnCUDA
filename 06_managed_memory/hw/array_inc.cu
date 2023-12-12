#include <cstdio>
#include <cstdlib>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "CUDA error code: %d\n", __err); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes) {
    ptr = (T)malloc(num_bytes);
}

__global__ void inc(int *array, size_t n) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n) {
        array[idx]++;
        idx += blockDim.x * gridDim.x; // grid-stride loop
    }
}

const size_t ds = 32ULL * 1024ULL * 1024ULL;

int main() {
    int *h_array, *d_array;

    alloc_bytes(h_array, ds * sizeof(h_array[0]));
    cudaMalloc(&d_array, ds * sizeof(d_array[0]));
    cudaCheckErrors("cudaMalloc Error");

    // No need to memset since you are going to overwrite the array with kernel results

    cudaMemcpy(d_array, h_array, ds * sizeof(h_array[0]), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H->D Error");

    inc<<<256, 256>>>(d_array, ds);
    cudaCheckErrors("kernel launch error");

    cudaMemcpy(h_array, d_array, ds * sizeof(h_array[0]), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution or cudaMemcpy D->H Error");

    for (int i = 0; i < ds; i++) {
        if (h_array[i] != 1) {
            printf("mismatch at %d, was: %d, expected: %d\n", i, h_array[i], 1);
            return -1;
        }
    }

    printf("success!\n");

    // Don't forget to free the allocated memory
    free(h_array);
    cudaFree(d_array);

    return 0;
}
