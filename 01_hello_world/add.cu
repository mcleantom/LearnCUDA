#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// __global__ executed on device, can be called from host
__global__ void add(int* a, int *b, int *c) {
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void add_parallel(int *a, int *b, int *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

__global__ void add_blocks_and_threads(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

__global__ void add_any_size(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

void random_ints(int* a, int size) {
    for (int i=0; i<size; ++i) {
        a[i] = rand() % 100;
    }
}

void printArray(const int* a, int size) {
    for (int i = 0; i < size; ++i) {
        printf("%d, ", a[i]);
    }
    printf("\n");
}

#define N 32
#define THREADS_PER_BLOCK 4

int main(void) {
  int *a, *b, *c;  // host copies
  int *d_a, *d_b, *d_c; // device copies
  int size = N * sizeof(int);
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  a = (int *)malloc(size);
  random_ints(a, N);
  b = (int *)malloc(size);
  random_ints(b, N);
  c = (int *)malloc(size);
  random_ints(c, N);

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);
  // BLOCK_SIZE, THREAD_SIZE
  add<<<N,1>>>(d_a, d_b, d_c);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
//   printArray(a, N);
//   printArray(b, N);
//   printArray(c, N);

  add_parallel<<<1, N>>>(d_a, d_b, d_c);
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
//   printArray(a, N);
//   printArray(b, N);
//   printArray(c, N);

  add_blocks_and_threads<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
//   printArray(a, N);
//   printArray(b, N);
//   printArray(c, N);

  add_any_size<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  printArray(a, N);
  printArray(b, N);
  printArray(c, N);

  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}
