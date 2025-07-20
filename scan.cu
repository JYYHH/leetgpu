// #include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

// blk_size is actually how many threads in a block
// iteration_per_thread is how many iterations
// thus the true block size (number of elements to handle) is blk_size * iteration_per_thread * 2
template <int blk_size, int iteration_per_thread, bool is_bottom_level>
__global__ void scan_kernel(const float* input, float* output, float* rst_next_level, int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int offset = bid * blk_size * iteration_per_thread * 2;

    // load data into shared memory
    __shared__ float temp[blk_size << 1];
    float temp_sum = 0.0f;

    for (int i = 0; i < iteration_per_thread; i++) {
        if (offset + 2 * i * blk_size + tid < N)
            temp[tid] = input[offset + 2 * i * blk_size + tid];
        else 
            temp[tid] = 0.0f;
        if (offset + (2 * i + 1) * blk_size + tid < N)
            temp[tid + blk_size] = input[offset + (2 * i + 1) * blk_size + tid];
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
        if (offset + 2 * i * blk_size + tid < N)
            output[offset + 2 * i * blk_size + tid] = temp[tid] + temp_sum;
        if (offset + (2 * i + 1) * blk_size + tid < N)
            output[offset + (2 * i + 1) * blk_size + tid] = temp[tid + blk_size] + temp_sum;
        temp_sum += temp[(blk_size << 1) - 1];
        __syncthreads();
    }

    if (tid == 0 && is_bottom_level)
        rst_next_level[bid] = temp_sum;
}

// __global__ void scan_kernel_serial(float* input_output, int N) {
//     for (int i = 1; i < N; i++) {
//         input_output[i] += input_output[i - 1];
//     }
// }

template <int blk_size, int iteration_per_thread>
__global__ void walk_back_kernel(float* input_output, float* rst_level_1, int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int offset = bid * blk_size * iteration_per_thread * 2;
    const float add = bid ? rst_level_1[bid - 1] : 0.0f;
    
    for (int i = 0; i < iteration_per_thread; i++) {
        if (offset + 2 * i * blk_size + tid < N)
            input_output[offset + 2 * i * blk_size + tid] += add;
        if (offset + (2 * i + 1) * blk_size + tid < N)
            input_output[offset + (2 * i + 1) * blk_size + tid] += add;
    }
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    // init
    const int BLOCK_SIZE_1 = 512, BLOCK_ITERATION_1 = 8;
    const int BLOCK_ELEMENTS_1 = (BLOCK_SIZE_1 * BLOCK_ITERATION_1) << 1;
    const int BLOCK_SIZE_2 = 512, BLOCK_ITERATION_2 = 16;
    const int BLOCK_ELEMENTS_2 = (BLOCK_SIZE_2 * BLOCK_ITERATION_2) << 1;
    int blk_num_1s = (N + BLOCK_ELEMENTS_1 - 1) / BLOCK_ELEMENTS_1;
    int blk_num_2s = (blk_num_1s + BLOCK_ELEMENTS_2 - 1) / BLOCK_ELEMENTS_2;
    float* output_1s;
    cudaMalloc(&output_1s, blk_num_1s * sizeof(float));
    // first level scan
    scan_kernel<BLOCK_SIZE_1, BLOCK_ITERATION_1, true><<<blk_num_1s, BLOCK_SIZE_1>>>(input, output, output_1s, N);
    cudaDeviceSynchronize();
    // second level scan
    scan_kernel<BLOCK_SIZE_2, BLOCK_ITERATION_2, false><<<blk_num_2s, BLOCK_SIZE_2>>>(output_1s, output_1s, NULL, blk_num_1s);
    // cudaDeviceSynchronize(); // no need to sync here, since we have at most 1 block
    // walk back
    walk_back_kernel<BLOCK_SIZE_1, BLOCK_ITERATION_1><<<blk_num_1s, BLOCK_SIZE_1>>>(output, output_1s, N);
    // cudaDeviceSynchronize();
    // free memory
    cudaFree(output_1s);
}

int main() {
    const int N = 25000000;
    float* input = (float*)malloc(N * sizeof(float));
    float* output = (float*)malloc(N * sizeof(float));
    float* input_device, *output_device;
    for (int i = 0; i < N; i++) {
        input[i] = (i % 2333 + 251) * (1 / 2333.0f);
        // input[i] = 1.0f;
    }
    cudaMalloc(&input_device, N * sizeof(float));
    cudaMalloc(&output_device, N * sizeof(float));
    cudaMemcpy(input_device, input, N * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 1; i < N; i++) {
        input[i] += input[i - 1];
    }
    // timer
    clock_t start = clock();
    solve(input_device, output_device, N);
    clock_t end = clock();
    printf("Time: %f\n", (float)(end - start) / CLOCKS_PER_SEC);
    cudaMemcpy(output, output_device, N * sizeof(float), cudaMemcpyDeviceToHost);
    int cnt = 0;
    for (int i = 0; i < N; i++) {
        if (fabs(output[i] - input[i]) > 1e-2) {
            printf("(%d, %f, %f) ", i, output[i], input[i]);
            cnt ++;
            if (cnt > 10)
                break;
        }
    }
    puts("");
    cudaFree(input_device);
    cudaFree(output_device);
    free(input);
    free(output);
    return 0;   
}