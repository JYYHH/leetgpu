// #include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

const int THREADS_PER_BLOCK = 256;
const int UNIT_PER_BLOCK = THREADS_PER_BLOCK << 1;
const int THREADS_PER_BLOCK_LOG = 8;

__device__ float _my_get(const float *data, const float *more_data, const int N, const int global_id){
    if (global_id < N)
        return data[global_id];
    else
        return more_data[global_id - N];
}

__device__ void _my_set(float *data, float *more_data, const int N, const int global_id, const float value){
    if (global_id < N)
        data[global_id] = value;
    else
        more_data[global_id - N] = value;
}

__global__ void initialize_and_local_sort_kernel(float* data, float* more_data, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int global_id = bid * UNIT_PER_BLOCK + tid;
    __shared__ float shared_data[UNIT_PER_BLOCK];

    // initialization for the INF data
    if (global_id >= N)
        shared_data[tid] = INFINITY;
    else
        shared_data[tid] = data[global_id];
    if (global_id + block_size >= N)
        shared_data[tid + block_size] = INFINITY;
    else
        shared_data[tid + block_size] = data[global_id + block_size];
    __syncthreads();

    // local bitonic sort
    for (int k = 2; k < (N << 1) && k <= UNIT_PER_BLOCK; k <<= 1){
        int de_or_in = (k == UNIT_PER_BLOCK) ? (bid & 1) : ((tid & (k >> 1)) > 0);
        for (int j = k >> 1; j; j >>= 1){
            int left_id = tid + tid / j * j;
            int right_id = left_id ^ j;
            float left_data = shared_data[left_id];
            float right_data = shared_data[right_id];
            if ((left_data > right_data) ^ de_or_in){
                shared_data[left_id] = right_data;
                shared_data[right_id] = left_data;
            }
        }
        __syncthreads();
    }
                // save data to global memory
    if (global_id >= N)
        more_data[global_id - N] = shared_data[tid];
    else
        data[global_id] = shared_data[tid];
    if (global_id + block_size >= N)
        more_data[global_id + block_size - N] = shared_data[tid + block_size];
    else
        data[global_id + block_size] = shared_data[tid + block_size];
}

__global__ void global_sort_single_iteration_kernel(float* data, float* more_data, const int N, const int k, const int j) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int global_id = bid * block_size + tid;

    int other_id = global_id ^ j;
    if (global_id < other_id){
        float left_data = _my_get(data, more_data, N, global_id);
        float right_data = _my_get(data, more_data, N, other_id);
        if (global_id & k){
            // switch is left < right
            if (left_data < right_data){
                _my_set(data, more_data, N, global_id, right_data);
                _my_set(data, more_data, N, other_id, left_data);
            }
        }
        else{
            // switch is left > right
            if (left_data > right_data){
                _my_set(data, more_data, N, global_id, right_data);
                _my_set(data, more_data, N, other_id, left_data);
            }
        }
    }
}

__global__ void global_sort_multiple_iterations_kernel(float* data, float* more_data, const int N, const int k) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int global_id = bid * block_size + tid;

    int in_or_de = (global_id & k) == 0;
    for (int j = block_size >> 1; j; j >>= 1){
        int other_id = global_id ^ j;
        if (global_id < other_id){
            float left_data = _my_get(data, more_data, N, global_id);
            float right_data = _my_get(data, more_data, N, other_id);
            if ((left_data < right_data) ^ in_or_de){
                _my_set(data, more_data, N, global_id, right_data);
                _my_set(data, more_data, N, other_id, left_data);
            }
        }
        __syncthreads();
    }
}

// data is device pointer
void solve(float* data, int N) {
    const int block_size = THREADS_PER_BLOCK;
    int block_num = 2, block_num_log = 1; // since we change the kernel of local sort
    while ((1 << (THREADS_PER_BLOCK_LOG + block_num_log)) < N) {
        block_num <<= 1;
        block_num_log ++;
    }
    int delta = (1 << (THREADS_PER_BLOCK_LOG + block_num_log)) - N;
    float* more_data = NULL;
    // printf("delta: %d, block_num: %d, block_num_log: %d\n", delta, block_num, block_num_log);
    if (delta)
        cudaMalloc(&more_data, (delta * sizeof(float)));
    initialize_and_local_sort_kernel<<<(block_num >> 1), block_size>>>(data, more_data, N);
    cudaDeviceSynchronize();
    // after that local sort, the first block will be in ascending order, second in descending order, third in ascending order, etc...

    if (block_num_log){
        // We need to sort among blocks
        for (int k = block_size << 2; k < (N << 1); k <<= 1){ // since we change the kernel of local sort
            for (int j = k >> 1; j >= block_size; j >>= 1){
                global_sort_single_iteration_kernel<<<block_num, block_size>>>(data, more_data, N, k, j);
                cudaDeviceSynchronize();
            }
            global_sort_multiple_iterations_kernel<<<block_num, block_size>>>(data, more_data, N, k);
            cudaDeviceSynchronize();
        }
    }
    if (delta)
        cudaFree(more_data);
}

int main(){
    int n;
    // scanf("%d", &n);
    n = 1 << 20;
    float* data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i ++)
        // scanf("%f", &data[i]);
        data[i] = n - i;
    float* data_device;
    cudaMalloc(&data_device, n * sizeof(float));
    cudaMemcpy(data_device, data, n * sizeof(float), cudaMemcpyHostToDevice);
    // C timer
    clock_t start, stop;
    start = clock();
    solve(data_device, n);
    stop = clock();
    printf("time: %f ms\n", (double)(stop - start) / CLOCKS_PER_SEC * 1000);

    cudaMemcpy(data, data_device, n * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < n; i ++)
    //     printf("%f ", data[i]);
    // printf("\n");
    cudaFree(data_device);
    free(data);
    
    return 0;
}

// TODO (JHY): Use half less threads for local sort, since half did not work at all
// TODO (JHY): Optimize the kernel loading and saving data mode
// TODO (JHY): The bottleneck is the multiple Kernel launchs and the global synchronization, 
//      try to reduce the number of kernel launches and the global synchronization