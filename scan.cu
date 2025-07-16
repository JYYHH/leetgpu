#include "solve.h"
#include <cuda_runtime.h>

const int BLOCK_SIZE = 256;
const int BLOCK_SIZE_LOG = 8;

__global__ void scan_kernel(const float* input, float* output, int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int blk_size = blockDim.x;
    const int offset = bid * blk_size;

    // load data into shared memory
    __shared__ float temp[BLOCK_SIZE];
    if (offset + tid < N) {
        temp[tid] = input[offset + tid];
    }
    else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // local scan
    for (int d = 0; d < BLOCK_SIZE_LOG; d++) {

    }
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {

} 