// #include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

const int BLOCK_SIZE = 256;
const int BLOCK_ELEMENTS = BLOCK_SIZE << 1;
const int BLOCK_SIZE_LOG = 8;

__global__ void scan_kernel(const float* input, float* output, int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int blk_size = blockDim.x;
    const int offset = bid * blk_size * 2;

    // load data into shared memory
    __shared__ float temp[BLOCK_ELEMENTS];
    if (offset + tid < N)
        temp[tid] = input[offset + tid];
    else 
        temp[tid] = 0.0f;
    if (offset + tid + blk_size < N)
        temp[tid + blk_size] = input[offset + tid + blk_size];
    else
        temp[tid + blk_size] = 0.0f;
    __syncthreads();

    // local scan
    for (int d = 0; d <= BLOCK_SIZE_LOG; d++) {
        int right_id = tid + (((tid >> d) + 1) << d);
        int left_id = (right_id ^ (1 << d)) | ((1 << d) - 1);
        temp[right_id] += temp[left_id];
        __syncthreads();
    }

    // store data back to global memory
    if (offset + tid < N)
        output[offset + tid] = temp[tid];
    if (offset + tid + blk_size < N)
        output[offset + tid + blk_size] = temp[tid + blk_size];
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    int blk_num = (N + BLOCK_ELEMENTS - 1) / BLOCK_ELEMENTS;
    printf("blk_num: %d\n", blk_num);
    scan_kernel<<<blk_num, BLOCK_SIZE>>>(input, output, N);
    cudaDeviceSynchronize();
} 

int main() {
    const int N = 1 << 7;
    float* input = (float*)malloc(N * sizeof(float));
    float* output = (float*)malloc(N * sizeof(float));
    float* input_device, *output_device;
    for (int i = 0; i < N; i++) {
        input[i] = 2 * i + 1;
    }
    cudaMalloc(&input_device, N * sizeof(float));
    cudaMalloc(&output_device, N * sizeof(float));
    cudaMemcpy(input_device, input, N * sizeof(float), cudaMemcpyHostToDevice);
    solve(input_device, output_device, N);
    cudaMemcpy(output, output_device, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf("%f ", output[i]);
    }
    puts("");
    cudaFree(input_device);
    cudaFree(output_device);
    free(input);
    free(output);
    return 0;   
}