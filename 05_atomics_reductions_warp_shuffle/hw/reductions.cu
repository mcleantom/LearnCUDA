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


const size_t N = 8ULL * 1024ULL * 1024ULL;  // data size
const int BLOCK_SIZE = 256;  // block size

__global__ void atomic_red(const float *gdata, float *out) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        atomicAdd(out, gdata[idx]);
    }
}

__global__ void reduce(float *gdata, float *out) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    while (idx < N) {  // grid stride loop to load data into shared memory
        sdata[tid] += gdata[idx];
        idx += gridDim.x * blockDim.x;
    }

    // parallel sweep reduction
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        /*
        Example:
        1   2       3   4       5   6       7   8
          3           7           11          15
                10                     26
                            36
        */
        __syncthreads();
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_a(float *gdata, float *out) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    while (idx < N) {  // grid stride loop to load data into shared memory
        sdata[tid] += gdata[idx];
        idx += blockDim.x * gridDim.x;
    }

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}


__global__ void reduce_ws(float *gdata, float *out) {
    __shared__ float sdata[32];  // We can only ever have a maximum of 32 warps
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = 0.0f;
    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    while (idx < N) {  // grid stride loop to add to val
        val += gdata[idx];
        idx += blockDim.x * gridDim.x;
    }

    // 1st warp-shuffle reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    if (lane == 0) {
        // We have computed the sum for each warp, now store it in shared memory
        sdata[warpId] = val;
    }
    __syncthreads();

    // hereafter, just warp 0
    if (warpId == 0) {
        // Reload from shared memory if the warp existed
        val = (tid < blockDim.x / warpSize)? sdata[tid] : 0.0f;
        // Final warp-shuffle reduction
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (tid == 0) {
            atomicAdd(out, val);
        }
    }
}


int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float *h_A, *h_sum, *d_A, *d_sum;
    h_A = new float[N];
    h_sum = new float;
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
    }
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheckErrors("cudaMemset failure");
    cudaEventRecord(start);
    atomic_red<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaCheckErrors("kernel launch failure");
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    printf("atomic reduction: %f\n", *h_sum);
    if (*h_sum != (float)N) {
        printf("atomic reduction failed\n");
    }
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("atomic reduction time: %f ms\n", time);
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheckErrors("cudaMemset failure");
    cudaEventRecord(start);
    reduce_a<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaCheckErrors("kernel launch failure");
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    printf("atomic parallel sweep reduction: %f\n", *h_sum);
    if (*h_sum != (float)N) {
        printf("atomic parallel sweep reduction failed\n");
    }
    cudaEventElapsedTime(&time, start, stop);
    printf("atomic parallel sweep reduction time: %f ms\n", time);
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheckErrors("cudaMemset failure");
    cudaEventRecord(start);
    reduce_ws<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaCheckErrors("kernel launch failure");
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    printf("warp shuffle reduction: %f\n", *h_sum);
    if (*h_sum != (float)N) {
        printf("warp shuffle reduction failed\n");
    }
    cudaEventElapsedTime(&time, start, stop);
    printf("warp shuffle reduction time: %f ms\n", time);
    cudaFree(d_A);
    cudaFree(d_sum);
    cudaCheckErrors("cudaFree failure");
    delete[] h_A;
    delete h_sum;
    return 0;
}