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
                A_shared[row_shared][col_shared] = (row_A < M && col_A < K) ? A[row_A * K + col_A] : 0.0f;
                B_shared[row_shared][col_shared] = (row_B < K && col_B < N) ? B[row_B * N + col_B] : 0.0f;
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
                C[row * N + col] = matmul_local[i * LOCAL_TILE_LENGTH + j];
            }
        }
    }
}

void solve(const float* input, float* output, int N, int P) {
    // special case
    if (P == 1){
        cudaMemcpy(output, input, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }
    // general case, and P >= 2
    float* mid_matrix;
    cudaMalloc((void**)&mid_matrix, N * N * sizeof(float));
    dim3 block(N_TILE_SIZE / LOCAL_TILE_LENGTH, N_TILE_SIZE / LOCAL_TILE_LENGTH);
    dim3 grid((N + N_TILE_SIZE - 1) / N_TILE_SIZE, (N + N_TILE_SIZE - 1) / N_TILE_SIZE);

    int p = 0, where_data = 0, used = 0; // where_data == 0 means current power is in output
    while ((1 << p) <= P)
        p ++;
    p --;
    // now p >= 1
    for (int i = p - 1; i >= 0; i--){
        if (P & (1 << i)){
            if (used){
                if (where_data){
                    MatrixMultiplyKernel<<<grid, block>>>(input, mid_matrix, output, N, N, N);
                    MatrixMultiplyKernel<<<grid, block>>>(output, output, mid_matrix, N, N, N); 
                }
                else{
                    MatrixMultiplyKernel<<<grid, block>>>(input, output, mid_matrix, N, N, N);
                    MatrixMultiplyKernel<<<grid, block>>>(mid_matrix, mid_matrix, output, N, N, N); 
                }
            }
            else{
                if (where_data){
                    MatrixMultiplyKernel<<<grid, block>>>(input, input, output, N, N, N);
                    MatrixMultiplyKernel<<<grid, block>>>(input, output, mid_matrix, N, N, N);   
                }
                else{
                    MatrixMultiplyKernel<<<grid, block>>>(input, input, mid_matrix, N, N, N);
                    MatrixMultiplyKernel<<<grid, block>>>(input, mid_matrix, output, N, N, N);
                }
                used = 1;
            }
        }
        else{
            if (used){
                if (where_data)
                    MatrixMultiplyKernel<<<grid, block>>>(mid_matrix, mid_matrix, output, N, N, N);
                else
                    MatrixMultiplyKernel<<<grid, block>>>(output, output, mid_matrix, N, N, N);
            }
            else{
                if (where_data)
                    MatrixMultiplyKernel<<<grid, block>>>(input, input, output, N, N, N);
                else
                    MatrixMultiplyKernel<<<grid, block>>>(input, input, mid_matrix, N, N, N);
                used = 1;
            }
            where_data ^= 1;
        }
    }
    if (where_data)
        cudaMemcpy(output, mid_matrix, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(mid_matrix);
    cudaDeviceSynchronize();
} 