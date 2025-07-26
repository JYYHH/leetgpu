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
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int K,
    const int N
){

    __shared__ float A_shared[N_TILE_SIZE][K_TILE_SIZE];
    __shared__ float B_shared[K_TILE_SIZE][M_TILE_SIZE];
    float A_shared_local[LOCAL_TILE_LENGTH];
    float B_shared_local[LOCAL_TILE_LENGTH];
    float matmul_local[LOCAL_TILE_LENGTH * LOCAL_TILE_LENGTH];
    // init the local sum
    #pragma unroll
    for (int i = 0; i < LOCAL_TILE_LENGTH; i++) {
        for (int j = 0; j < LOCAL_TILE_LENGTH; j++) {
            matmul_local[i * LOCAL_TILE_LENGTH + j] = 0.0f;
        }
    }

    const int batch_id = blockIdx.z;
    const int row_offset = threadIdx.y;
    const int col_offset = threadIdx.x;
    const int reduce_dim = (K + K_TILE_SIZE - 1) / K_TILE_SIZE;

    // walk through
    for (int iter = 0; iter < reduce_dim; iter++){
        // loading the shared memory
        #pragma unroll
        for (int i = 0; i < LOCAL_TILE_LENGTH; i++) {
            for (int j = 0; j < LOCAL_TILE_LENGTH; j++) {
                const int row_shared = i * blockDim.y + row_offset;
                const int col_shared = j * blockDim.x + col_offset;
                const int row_A = blockIdx.y * blockDim.y * LOCAL_TILE_LENGTH + row_shared;
                const int col_B = blockIdx.x * blockDim.x * LOCAL_TILE_LENGTH + col_shared;
                const int col_A = iter * K_TILE_SIZE + col_shared;
                const int row_B = iter * K_TILE_SIZE + row_shared;
                A_shared[row_shared][col_shared] = (row_A < M && col_A < K) ? A[batch_id * M * K + row_A * K + col_A] : 0.0f;
                B_shared[row_shared][col_shared] = (row_B < K && col_B < N) ? B[batch_id * K * N + row_B * N + col_B] : 0.0f;
            }
        }
        __syncthreads();

        // then do the local reduction here
        for (int reduct_iter = 0; reduct_iter < K_TILE_SIZE; reduct_iter++){
            for (int i = 0; i < LOCAL_TILE_LENGTH; i++){
                A_shared_local[i] = A_shared[i * blockDim.y + row_offset][reduct_iter];
                B_shared_local[i] = B_shared[reduct_iter][i * blockDim.x + col_offset];
            }
            #pragma unroll
            for (int i = 0; i < LOCAL_TILE_LENGTH; i++){
                for (int j = 0; j < LOCAL_TILE_LENGTH; j++){
                    matmul_local[i * LOCAL_TILE_LENGTH + j] += A_shared_local[i] * B_shared_local[j];
                }
            }
        }
        __syncthreads();
    }

    // save the result
    #pragma unroll
    for (int i = 0; i < LOCAL_TILE_LENGTH; i++){
        for (int j = 0; j < LOCAL_TILE_LENGTH; j++){
            const int row = blockIdx.y * blockDim.y * LOCAL_TILE_LENGTH + i * blockDim.y + row_offset;
            const int col = blockIdx.x * blockDim.x * LOCAL_TILE_LENGTH + j * blockDim.x + col_offset;
            if (row < M && col < N){
                C[batch_id * M * N + row * N + col] = matmul_local[i * LOCAL_TILE_LENGTH + j] + C[batch_id * M * N + row * N + col];
            }
        }
    }
}

// A, B, and C are device pointers
void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 block(M_TILE_SIZE / LOCAL_TILE_LENGTH, N_TILE_SIZE / LOCAL_TILE_LENGTH);
    dim3 grid((M + M_TILE_SIZE - 1) / M_TILE_SIZE, (N + N_TILE_SIZE - 1) / N_TILE_SIZE, BATCH);
    cudaMemset(C, 0, BATCH * M * N * sizeof(float));
    MatrixMultiplyKernel<<<grid, block>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}

int main() {
    const int BATCH = 16;
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    float* A = (float*)malloc(BATCH * M * K * sizeof(float));
    float* B = (float*)malloc(BATCH * K * N * sizeof(float));
    float* C = (float*)malloc(BATCH * M * N * sizeof(float));
    memset(A, 0, BATCH * M * K * sizeof(float));
    memset(B, 0, BATCH * K * N * sizeof(float));
    memset(C, 0, BATCH * M * N * sizeof(float));
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
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, BATCH * M * K * sizeof(float));
    cudaMalloc((void**)&d_B, BATCH * K * N * sizeof(float));
    cudaMalloc((void**)&d_C, BATCH * M * N * sizeof(float));
    cudaMemcpy(d_A, A, BATCH * M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, BATCH * K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, BATCH * M * N * sizeof(float), cudaMemcpyHostToDevice);   
    clock_t start = clock();
    solve(d_A, d_B, d_C, BATCH, M, N, K);
    clock_t end = clock();
    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    cudaMemcpy(C, d_C, BATCH * M * N * sizeof(float), cudaMemcpyDeviceToHost);
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
