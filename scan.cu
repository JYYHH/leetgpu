// #include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

template <int blk_size>
__global__ void scan_kernel(const float* input, float* output, float* rst_next_level, int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int offset = bid * blk_size * 2;

    // load data into shared memory
    __shared__ float temp[blk_size << 1];
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
    for (int d = 0; (1 << d) <= blk_size; d++) {
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
    if (tid == 0)
        rst_next_level[bid] = temp[(blk_size << 1) - 1];
}

__global__ void scan_kernel_serial(float* input_output, int N) {
    for (int i = 1; i < N; i++) {
        input_output[i] += input_output[i - 1];
    }
}

__global__ void walk_back_kernel(const int blk_size_2, float* input_output, float* rst_level_1, float* rst_level_2, int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_id = bid * blockDim.x + tid;
    const int bid_2 = bid / blk_size_2;
    const float add_1 = (bid & (blk_size_2 - 1)) ? rst_level_1[bid - 1] : 0.0f;
    const float add_2 = bid_2 ? rst_level_2[bid_2 - 1] : 0.0f;
    
    if (global_id < N)
        input_output[global_id] += add_1 + add_2;
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    // init
    const int BLOCK_SIZE_1 = 256;
    const int BLOCK_ELEMENTS_1 = BLOCK_SIZE_1 << 1;
    const int BLOCK_SIZE_2 = 128;
    const int BLOCK_ELEMENTS_2 = BLOCK_SIZE_2 << 1;
    int blk_num_1s = (N + BLOCK_ELEMENTS_1 - 1) / BLOCK_ELEMENTS_1;
    int blk_num_2s = (blk_num_1s + BLOCK_ELEMENTS_2 - 1) / BLOCK_ELEMENTS_2;
    float* output_1s, *output_2s;
    cudaMalloc(&output_1s, blk_num_1s * sizeof(float));
    cudaMalloc(&output_2s, blk_num_2s * sizeof(float));
    // first level scan
    scan_kernel<BLOCK_SIZE_1><<<blk_num_1s, BLOCK_SIZE_1>>>(input, output, output_1s, N);
    cudaDeviceSynchronize();
    // second level scan
    scan_kernel<BLOCK_SIZE_2><<<blk_num_2s, BLOCK_SIZE_2>>>(output_1s, output_1s, output_2s, blk_num_1s);
    cudaDeviceSynchronize();
    // third level scan
    scan_kernel_serial<<<1, 1>>>(output_2s, blk_num_2s);
    cudaDeviceSynchronize();
    // walk back
    walk_back_kernel<<<blk_num_1s, BLOCK_ELEMENTS_1>>>(BLOCK_ELEMENTS_2, output, output_1s, output_2s, N);
    // cudaDeviceSynchronize();
    // free memory
    cudaFree(output_1s);
    cudaFree(output_2s);
}

int main() {
    const int N = 100000000;
    float* input = (float*)malloc(N * sizeof(float));
    float* output = (float*)malloc(N * sizeof(float));
    float* input_device, *output_device;
    for (int i = 0; i < N; i++) {
        // input[i] = max(0.0, 2 * i - 1.0);
        input[i] = 1.0f;
    }
    cudaMalloc(&input_device, N * sizeof(float));
    cudaMalloc(&output_device, N * sizeof(float));
    cudaMemcpy(input_device, input, N * sizeof(float), cudaMemcpyHostToDevice);
    // timer
    clock_t start = clock();
    solve(input_device, output_device, N);
    clock_t end = clock();
    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    cudaMemcpy(output, output_device, N * sizeof(float), cudaMemcpyDeviceToHost);
    // int cnt = 0;
    // for (int i = 0; i < N; i++) {
    //     if (fabs(output[i] - (i + 1)) > 1e-2) {
    //         printf("(%d, %f) ", i, output[i]);
    //         cnt ++;
    //         if (cnt > 10)
    //             break;
    //     }
    // }
    // puts("");
    cudaFree(input_device);
    cudaFree(output_device);
    free(input);
    free(output);
    return 0;   
}