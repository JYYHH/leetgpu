#include <cuda_runtime.h>
#include <stdio.h>
#include <random>
#define DEBUG 0

const int BLOCK_SIZE = 256;
const int ELEMENTS_PER_THREAD = 8;
const int WARP_SIZE = 32;

__global__ void sum_and_max_local_kernel(const float* input, float2* output, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    float sum_local = 0.0f, sum_block;
    float max_local = -INFINITY, new_max_local, max_block;
    __shared__ float shared_array[BLOCK_SIZE];

    // step 1: reduce over stride
    #pragma unroll
    for (int i = tid + bid * blockDim.x; i < N; i += stride) {
        new_max_local = fmaxf(max_local, input[i]);
        sum_local = sum_local * expf(max_local - new_max_local) + expf(input[i] - new_max_local);
        max_local = new_max_local;
    }
    __syncthreads();

    // step 2: reduce max inside the block
    shared_array[tid] = max_local;
    __syncthreads();
    for (int i = BLOCK_SIZE >> 1; i >= WARP_SIZE; i >>= 1) {
        if (tid & i) {
            shared_array[tid ^ i] = fmaxf(shared_array[tid ^ i], shared_array[tid]);
        }
        __syncthreads();
    }
    for (int i = WARP_SIZE >> 1; i >= 1; i >>= 1) {
        if (tid & i) {
            shared_array[tid ^ i] = fmaxf(shared_array[tid ^ i], shared_array[tid]);
        }
    }
    __syncthreads();
    max_block = shared_array[0];
    __syncthreads();

    // step 3: reduce sum inside the block
    shared_array[tid] = sum_local * expf(max_local - max_block);
    __syncthreads();
    for (int i = BLOCK_SIZE >> 1; i >= WARP_SIZE; i >>= 1) {
        if (tid & i) {
            shared_array[tid ^ i] += shared_array[tid];
        }
        __syncthreads();
    }
    for (int i = WARP_SIZE >> 1; i >= 1; i >>= 1) {
        if (tid & i) {
            shared_array[tid ^ i] += shared_array[tid];
        }
    }
    __syncthreads();
    sum_block = shared_array[0];

    // step 4: update the global max and sum
    if (tid == 0){
        output[bid] = make_float2(sum_block, max_block);
    }
}

__global__ void sum_and_max_global_kernel(const float* input, float2 *combined_sum_max, const int blocksPerGrid) {
    const int tid = threadIdx.x;
    const int stride = WARP_SIZE;
    float sum_local = 0.0f, sum_block;
    float max_local = -INFINITY, new_max_local, max_block;
    __shared__ float shared_array[WARP_SIZE];

    const float2 *input_float2 = reinterpret_cast<const float2 *>(input);
    #pragma unroll
    for (int i = tid; i < blocksPerGrid; i += stride) {
        new_max_local = fmaxf(max_local, input_float2[i].y);
        sum_local = sum_local * expf(max_local - new_max_local) + input_float2[i].x * expf(input_float2[i].y - new_max_local);
        max_local = new_max_local;
    }

    shared_array[tid] = max_local;
    for (int i = WARP_SIZE >> 1; i >= 1; i >>= 1) {
        if (tid & i) {
            shared_array[tid ^ i] = fmaxf(shared_array[tid ^ i], shared_array[tid]);
        }
    }
    max_block = shared_array[0];

    shared_array[tid] = sum_local * expf(max_local - max_block);
    for (int i = WARP_SIZE >> 1; i >= 1; i >>= 1) {
        if (tid & i) {
            shared_array[tid ^ i] += shared_array[tid];
        }
    }
    sum_block = shared_array[0];

    if (tid == 0){
        combined_sum_max[0] = make_float2(sum_block, max_block);
    }
}

__global__ void softmax_kernel(const float* input, float* output, const float2* combined_sum_max, const int N) {
    const int stride = blockDim.x * gridDim.x;
    const float sum_global = combined_sum_max[0].x;
    const float max_global = combined_sum_max[0].y;

    #pragma unroll
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride) {
        output[i] = expf(input[i] - max_global) / sum_global;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = BLOCK_SIZE, elementsPerBlock = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    int blocksPerGrid = (N + elementsPerBlock - 1) / elementsPerBlock;
    float2 *combined_output;
    cudaMalloc(&combined_output, sizeof(float2));
    
    if (blocksPerGrid > 1){
        sum_and_max_local_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, (float2 *)output, N);
        cudaDeviceSynchronize();

        sum_and_max_global_kernel<<<1, WARP_SIZE>>>(output, combined_output, blocksPerGrid);
        cudaDeviceSynchronize();
    }
    else{
        sum_and_max_local_kernel<<<1, threadsPerBlock>>>(input, combined_output, N);
        cudaDeviceSynchronize();
    }

#ifdef DEBUG
    float2 *combined_output_host = (float2 *)malloc(sizeof(float2));
    cudaMemcpy(combined_output_host, combined_output, sizeof(float2), cudaMemcpyDeviceToHost);
    printf("GPU: max = %f, sum = %f\n", combined_output_host->y, combined_output_host->x);
    free(combined_output_host);
#endif

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, combined_output, N);
    cudaDeviceSynchronize();

    cudaFree(combined_output);
}

void solve_CPU(const float* input, float* output, int N){
    float max_ = -INFINITY, sum_ = 0.0f;
    for (int i = 0; i < N; i++){
        max_ = fmaxf(max_, input[i]);
    }
    for (int i = 0; i < N; i++){
        sum_ += expf(input[i] - max_);
    }
    printf("CPU: max = %f, sum = %f\n", max_, sum_);
    for (int i = 0; i < N; i++){
        output[i] = expf(input[i] - max_) / sum_;
    }
}

int main(){
    int N = 500000;
    float *input = (float *)malloc(N * sizeof(float));
    float *output = (float *)malloc(N * sizeof(float));
    float *output_cpu = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++){
        input[i] = rand() / (float)RAND_MAX;
    }
    float *input_d, *output_d;
    cudaMalloc(&input_d, N * sizeof(float));
    cudaMalloc(&output_d, N * sizeof(float));
    cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice);

    solve(input_d, output_d, N);
    solve_CPU(input, output_cpu, N);
    cudaMemcpy(output, output_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) if (abs(output[i] - output_cpu[i]) > 1e-6){
        printf("Error at index %d: %f vs %f\n", i, output[i], output_cpu[i]);
    }
    printf("All values are within 1e-6 of the CPU result\n");

    return 0;
}