// #include "solve.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>

const int N_TILE_SIZE = 64;
const int M_TILE_SIZE = 64;
const int K_TILE_SIZE = 64;
const int LOCAL_TILE_LENGTH = 4; // must be the divisor of N_TILE_SIZE and M_TILE_SIZE and K_TILE_SIZE

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
    half A_shared_local[LOCAL_TILE_LENGTH];
    half B_shared_local[LOCAL_TILE_LENGTH];
    float matmul_local[LOCAL_TILE_LENGTH * LOCAL_TILE_LENGTH];
    // init the local sum
    for (int i = 0; i < LOCAL_TILE_LENGTH; i++) {
        for (int j = 0; j < LOCAL_TILE_LENGTH; j++) {
            matmul_local[i * LOCAL_TILE_LENGTH + j] = 0.0;
        }
    }

    const int row_offset = threadIdx.y;
    const int col_offset = threadIdx.x;
    const int reduce_dim = (K + K_TILE_SIZE - 1) / K_TILE_SIZE;

    // walk through
    for (int iter = 0; iter < reduce_dim; iter++){
        // loading the shared memory
        for (int i = 0; i < LOCAL_TILE_LENGTH; i++) {
            for (int j = 0; j < LOCAL_TILE_LENGTH; j++) {
                const int row_shared = i * blockDim.y + row_offset;
                const int col_shared = j * blockDim.x + col_offset;
                const int row_A = blockIdx.y * blockDim.y * LOCAL_TILE_LENGTH + row_shared;
                const int col_B = blockIdx.x * blockDim.x * LOCAL_TILE_LENGTH + col_shared;
                const int col_A = iter * K_TILE_SIZE + col_shared;
                const int row_B = iter * K_TILE_SIZE + row_shared;
                A_shared[row_shared][col_shared] = (row_A < M && col_A < K) ? A[row_A * K + col_A] : __float2half(0.0);
                B_shared[row_shared][col_shared] = (row_B < K && col_B < N) ? B[row_B * N + col_B] : __float2half(0.0);
            }
        }
        __syncthreads();

        // then do the local reduction here
        for (int reduct_iter = 0; reduct_iter < K_TILE_SIZE; reduct_iter++){
            for (int i = 0; i < LOCAL_TILE_LENGTH; i++){
                A_shared_local[i] = A_shared[i * blockDim.y + row_offset][reduct_iter];
                B_shared_local[i] = B_shared[reduct_iter][i * blockDim.x + col_offset];
            }
            for (int i = 0; i < LOCAL_TILE_LENGTH; i++){
                for (int j = 0; j < LOCAL_TILE_LENGTH; j++){
                    matmul_local[i * LOCAL_TILE_LENGTH + j] += __half2float(A_shared_local[i]) * __half2float(B_shared_local[j]);
                }
            }
        }
        __syncthreads();
    }

    // save the result
    for (int i = 0; i < LOCAL_TILE_LENGTH; i++){
        for (int j = 0; j < LOCAL_TILE_LENGTH; j++){
            const int row = blockIdx.y * blockDim.y * LOCAL_TILE_LENGTH + i * blockDim.y + row_offset;
            const int col = blockIdx.x * blockDim.x * LOCAL_TILE_LENGTH + j * blockDim.x + col_offset;
            if (row < M && col < N){
                C[row * N + col] = __float2half(matmul_local[i * LOCAL_TILE_LENGTH + j] * alpha + __half2float(C[row * N + col]) * beta);
            }
        }
    }
    // if (row < M && col < N)
    //     C[row * N + col] = __float2half(ret * alpha + __half2float(C[row * N + col]) * beta);
}

// A, B, and C are device pointers
void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 block(M_TILE_SIZE / LOCAL_TILE_LENGTH, N_TILE_SIZE / LOCAL_TILE_LENGTH);
    dim3 grid((M + M_TILE_SIZE - 1) / M_TILE_SIZE, (N + N_TILE_SIZE - 1) / N_TILE_SIZE);
    MatrixMultiplyKernel<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
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
