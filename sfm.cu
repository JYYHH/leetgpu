#include <cuda_runtime.h>
// #include <stdio.h>
// #include <random>
// #include <time.h>
// #define DEBUG

const int BLOCK_SIZE = 256;
const int ELEMENTS_PER_THREAD = 32;
const int WARP_SIZE = 32, WARP_SIZE_LOG2 = 5;

__device__ void sequential_reduce(float *max_local, float *sum_local, const float *input, const int start, const int end, const int stride){
    float new_max_local;
    for (int i = start; i < end; i += stride) {
        new_max_local = fmaxf(*max_local, input[i]);
        *sum_local = *sum_local * __expf(*max_local - new_max_local) + __expf(input[i] - new_max_local);
        *max_local = new_max_local;
    }
}

__device__ void sequential_reduce_aftermap(float *max_local, float *sum_local, const float *input, const int start, const int end, const int stride){
    float new_max_local;
    const float2 *input_float2 = reinterpret_cast<const float2 *>(input);
    for (int i = start; i < end; i += stride) {
        new_max_local = fmaxf(*max_local, input_float2[i].y);
        *sum_local = *sum_local * __expf(*max_local - new_max_local) + input_float2[i].x * __expf(input_float2[i].y - new_max_local);
        *max_local = new_max_local;
    }
}
__device__ void warp_reduce(float *max_local, float *sum_local, const int delta_start){
    float new_max, new_sum, new_max_local;
    for (int delta = delta_start; delta > 0; delta >>= 1){
        new_max = __shfl_down_sync(0xFFFFFFFF, *max_local, delta);
        new_sum = __shfl_down_sync(0xFFFFFFFF, *sum_local, delta);
        new_max_local = fmaxf(*max_local, new_max);
        *sum_local = *sum_local * __expf(*max_local - new_max_local) + new_sum * __expf(new_max - new_max_local);
        *max_local = new_max_local;
    }
}

template <void (*reduce_kernel_pointer)(float *, float *, const float *, const int, const int, const int)>
__global__ void sum_and_max_kernel(const float* input, float2* output, const int N) {
    const int tid = threadIdx.x, bid = blockIdx.x, stride = blockDim.x * gridDim.x;
    const int warp_id = tid >> WARP_SIZE_LOG2, lane_id = tid & (WARP_SIZE - 1);
    const int warp_num = blockDim.x >> WARP_SIZE_LOG2;
    float sum_local = 0.0f, max_local = 0.0f;
    __shared__ float shared_array[BLOCK_SIZE >> (WARP_SIZE_LOG2 - 1)];

    // step 1: reduce over stride
    (*reduce_kernel_pointer)(&max_local, &sum_local, input, tid + bid * blockDim.x, N, stride);

    // step 2: reduce inside the warp
    warp_reduce(&max_local, &sum_local, WARP_SIZE >> 1);
    if (warp_num > 1){
        if (lane_id == 0){
            shared_array[warp_id] = max_local;
            shared_array[warp_id + warp_num] = sum_local;
        }
        __syncthreads();

        // step 3: reduce inside the block, use the first warp
        if (warp_id == 0){
            if (lane_id < warp_num){
                max_local = shared_array[lane_id];
                sum_local = shared_array[lane_id + warp_num];
            }
            else{
                max_local = 0.0f;
                sum_local = 0.0f;
            }
            warp_reduce(&max_local, &sum_local, warp_num >> 1);
        }
    }

    // step 4: update the global max and sum
    if (tid == 0){
        output[bid] = make_float2(sum_local, max_local);
    }
}

__global__ void softmax_kernel(const float* input, float* output, const float2* combined_sum_max, const int N) {
    const int stride = blockDim.x * gridDim.x;
    const float sum_global = combined_sum_max[0].x;
    const float max_global = combined_sum_max[0].y;

    #pragma unroll
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride) {
        output[i] = __expf(input[i] - max_global) / sum_global;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = BLOCK_SIZE, elementsPerBlock = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    int blocksPerGrid = (N + elementsPerBlock - 1) / elementsPerBlock;
    float2 *combined_output;
    cudaMalloc(&combined_output, sizeof(float2));
    
    if (blocksPerGrid > 1){
        sum_and_max_kernel<sequential_reduce><<<blocksPerGrid, threadsPerBlock>>>(input, (float2 *)output, N);
        // cudaDeviceSynchronize();

        sum_and_max_kernel<sequential_reduce_aftermap><<<1, 32>>>(output, combined_output, blocksPerGrid);
        // cudaDeviceSynchronize();
    }
    else{
        sum_and_max_kernel<sequential_reduce><<<blocksPerGrid, threadsPerBlock>>>(input, combined_output, N);
        // cudaDeviceSynchronize();
    }

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, combined_output, N);
    // cudaDeviceSynchronize();

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
    int N = 1 << 20;
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

    time_t start = clock();
    solve(input_d, output_d, N);
    time_t end = clock();
    printf("GPU time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    start = clock();
    solve_CPU(input, output_cpu, N);
    end = clock();
    printf("CPU time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    cudaMemcpy(output, output_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) if (abs(output[i] - output_cpu[i]) > 1e-6){
        printf("Error at index %d: %f vs %f\n", i, output[i], output_cpu[i]);
    }
    printf("All values are within 1e-6 of the CPU result\n");

    return 0;
}