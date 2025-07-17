// #include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

const int BLOCK_SIZE = 256;
const int ELEMENT_PER_THREAD = 8;

template<int bin_size>
__global__ void hist_kernel(const int* input, int* histogram, int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int offset = bid * BLOCK_SIZE * ELEMENT_PER_THREAD + tid * ELEMENT_PER_THREAD;

    __shared__ int s_hist[bin_size];
    // initialize the histogram
    #pragma unroll
    for (int i = tid; i < bin_size; i += BLOCK_SIZE) {
        s_hist[i] = 0;
    }

    __syncthreads();

    // count the histogram
    #pragma unroll
    for (int i = offset; i < offset + ELEMENT_PER_THREAD && i < N; i ++) {
        atomicAdd(s_hist + input[i], 1);
    }

    __syncthreads();

    // add the local histogram to the global histogram
    #pragma unroll
    for (int i = tid; i < bin_size; i += BLOCK_SIZE) {
        atomicAdd(histogram + i, s_hist[i]);
    }
}

// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins) {
    const int blk_elements = BLOCK_SIZE * ELEMENT_PER_THREAD;
    const int block_num = (N + blk_elements - 1) / blk_elements;

    if (num_bins <= 256) {
        hist_kernel<256><<<block_num, BLOCK_SIZE>>>(input, histogram, N);
    } else if (num_bins <= 512) {
        hist_kernel<512><<<block_num, BLOCK_SIZE>>>(input, histogram, N);
    } else {
        hist_kernel<1024><<<block_num, BLOCK_SIZE>>>(input, histogram, N);
    }
    // cudaDeviceSynchronize();
}

int main(){
    int N = 100000000;
    int num_bins = 1024;
    int* input = (int*)malloc(N * sizeof(int));
    int* histogram = (int*)malloc(num_bins * sizeof(int));
    int *input_d, *histogram_d;
    cudaMalloc((void**)&input_d, N * sizeof(int));
    cudaMalloc((void**)&histogram_d, num_bins * sizeof(int));
    for (int i = 0; i < N; i++) {
        input[i] = (i + 19) % (num_bins - 1);
    }
    cudaMemcpy(input_d, input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(histogram_d, 0, num_bins * sizeof(int));
    clock_t start = clock();
    solve(input_d, histogram_d, N, num_bins);
    clock_t end = clock();
    printf("Time: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    cudaMemcpy(histogram, histogram_d, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < num_bins; i++) {
    //     printf("%d ", histogram[i]);
    // }
    // printf("\n");
    free(input);
    free(histogram);
    cudaFree(input_d);
    cudaFree(histogram_d);
    return 0;
}
