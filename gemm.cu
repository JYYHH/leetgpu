// #include "solve.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>

const int N_TILE_SIZE = 32;
const int M_TILE_SIZE = 32;
const int K_TILE_SIZE = 32;

// __global__ void MatrixMultiplyKernel_simple(
//     const half* A,
//     const half* B,
//     half* C,
//     const int M,
//     const int K,
//     const int N,
//     float alpha,
//     float beta
// ){
    
//     // since in a warp, we have 32 threads in different threadIdx.x
//   const int row = blockIdx.y * blockDim.y + threadIdx.y;
//   const int col = blockIdx.x * blockDim.x + threadIdx.x;

//   float ret = 0.0; // one element in out matrix
//   if (row < M && col < N) {
//     for (int i = 0; i < K; i++) {
//         ret += __half2float(A[row * K + i]) * __half2float(B[i * N + col]);
//     }
//     C[row * N + col] = __float2half(ret * alpha + __half2float(C[row * N + col]) * beta);
//   }
// }

// first trivial version, one thread per output element, and for one tile of k, one thread just read one A and one B
__global__ void MatrixMultiplyKernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M,
    const int K,
    const int N,
    float alpha,
    float beta
){

    __shared__ half A_shared[N_TILE_SIZE][K_TILE_SIZE];
    __shared__ half B_shared[K_TILE_SIZE][M_TILE_SIZE];

    // In each block, we will compute a batch of the output matrix
    // All the threads in the block will work together to compute this batch
    const int row_offset = threadIdx.y;
    const int col_offset = threadIdx.x;
    const int row = blockIdx.y * blockDim.y + row_offset;
    const int col = blockIdx.x * blockDim.x + col_offset;
    const int reduce_dim = (K + K_TILE_SIZE - 1) / K_TILE_SIZE;
    float ret = 0.0; // one element in out matrix

    // walk through
    for (int i = 0; i < reduce_dim; i++){
        A_shared[row_offset][col_offset] = (row < M && (col_offset + i * K_TILE_SIZE) < K) ? A[row * K + col_offset + i * K_TILE_SIZE] : __float2half(0.0);
        B_shared[row_offset][col_offset] = ((row_offset + i * K_TILE_SIZE) < K && col < N) ? B[(row_offset + i * K_TILE_SIZE) * N + col] : __float2half(0.0);
        // B_shared[col_offset][row_offset] = ((row_offset + i * K_TILE_SIZE) < K && col < N) ? B[(row_offset + i * K_TILE_SIZE) * N + col] : __float2half(0.0);
        // sync all threads in this block
        __syncthreads();
        // then do the local reduction here
        for (int j = 0; j < K_TILE_SIZE; j++)
        ret += __half2float(A_shared[row_offset][j]) * __half2float(B_shared[j][col_offset]);
        // must have a sync, since next we will load sth. into the shared memory
        __syncthreads();
    }

    // save the result
    if (row < M && col < N)
        C[row * N + col] = __float2half(ret * alpha + __half2float(C[row * N + col]) * beta);
}

// A, B, and C are device pointers
void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 block(M_TILE_SIZE, N_TILE_SIZE);
    dim3 grid((M + M_TILE_SIZE - 1) / M_TILE_SIZE, (N + N_TILE_SIZE - 1) / N_TILE_SIZE);
    // if (K <= 256 && M <= 256 && N <= 256) {
    //     MatrixMultiplyKernel_simple<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
    // } else {
    MatrixMultiplyKernel<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
    // }
    cudaDeviceSynchronize();
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    half* A = (half*)malloc(M * K * sizeof(half));
    half* B = (half*)malloc(K * N * sizeof(half));
    half* C = (half*)malloc(M * N * sizeof(half));
    memset(A, 0, M * K * sizeof(half));
    memset(B, 0, K * N * sizeof(half));
    memset(C, 0, M * N * sizeof(half));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + i] = 1.0f;
        }   
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = 1.0f;
        }
    }
    half* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(half));
    cudaMalloc((void**)&d_B, K * N * sizeof(half));
    cudaMalloc((void**)&d_C, M * N * sizeof(half));
    cudaMemcpy(d_A, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(half), cudaMemcpyHostToDevice);   
    clock_t start = clock();
    solve(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    clock_t end = clock();
    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    cudaMemcpy(C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f ", (float)C[i * N + j]);
    //     }
    //     printf("\n");
    // }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    return 0;
}
