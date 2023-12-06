#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const size_t DSIZE = 16384;
const int block_size = 256;

// matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < ds) {
        float sum = 0.0f;
        for (size_t i = 0; i < ds; ++i) {
            sum += A[idx * ds + i];
        }
        sums[idx] = sum;
    }
}

__global__ void column_sums(const float *A, float *sums, size_t ds) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < ds) {
        float sum = 0.0f;
        for (size_t i = 0; i < ds; ++i) {
            sum += A[i * ds + idx];
        }
        sums[idx] = sum;
    }
}

bool validate(float *data, size_t sz) {
    for (size_t i = 0; i < sz; ++i) {
        if (data[i] != (float)sz) {
            printf("Results mismatch at %lu, was: %f, expected: %f\n", i, data[i], (float)sz);
            return false;
        }
    }
    return true;
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A, *h_sums, *d_A, *d_sums;
    h_A = new float[DSIZE * DSIZE];
    h_sums = new float[DSIZE];

    for (int i = 0; i < DSIZE * DSIZE; ++i) {
        h_A[i] = 1.0f;
    }

    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_sums, DSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");

    cudaEventRecord(start, 0);
    row_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("row sum time: %f ms\n", elapsedTime);

    cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    if (!validate(h_sums, DSIZE)) {
        printf("Validation failed\n");
        return -1;
    }
    printf("row sum correct\n");

    cudaMemset(d_sums, 0, DSIZE * sizeof(float));

    cudaEventRecord(start, 0);
    column_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("column sum time: %f ms\n", elapsedTime);

    cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    if (!validate(h_sums, DSIZE)) {
        printf("Validation failed\n");
        return -1;
    }
    printf("column sum correct\n");

    cudaFree(d_A);
    cudaFree(d_sums);
    delete[] h_A;
    delete[] h_sums;

    return 0;
}