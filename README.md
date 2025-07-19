## [LeetGPU GEMM](https://leetgpu.com/challenges/gemm-fp16)

### 1. Comparing the fast and slow trivial kernels (without shared memory)
1. `Slower one`: (35.97ms under `Tesla T4`)
```c
__global__ void MatrixMultiplyKernel_simple(
    const half* A,
    const half* B,
    half* C,
    const int M,
    const int K,
    const int N,
    float alpha,
    float beta
){
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  float ret = 0.0;
  if (row < M && col < N) {
    for (int i = 0; i < K; i++) {
        ret += __half2float(A[row * K + i]) * __half2float(B[i * N + col]);
    }
    C[row * N + col] = __float2half(ret * alpha + __half2float(C[row * N + col]) * beta);
  }
}
```
2. `Faster one`: (3.31ms under `Tesla T4`)
```c
__global__ void MatrixMultiplyKernel_simple(
    const half* A,
    const half* B,
    half* C,
    const int M,
    const int K,
    const int N,
    float alpha,
    float beta
){
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  float ret = 0.0;
  if (row < M && col < N) {
    for (int i = 0; i < K; i++) {
        ret += __half2float(A[row * K + i]) * __half2float(B[i * N + col]);
    }
    C[row * N + col] = __float2half(ret * alpha + __half2float(C[row * N + col]) * beta);
  }
}
```
- Note: Both will be launched like `MatrixMultiplyKernel_simple<<<grid, block>>>` with `dim3 block(M_TILE_SIZE, N_TILE_SIZE);` and `dim3 grid((M + M_TILE_SIZE - 1) / M_TILE_SIZE, (N + N_TILE_SIZE - 1) / N_TILE_SIZE);`
- Solution:
    1. Threads in a block are organized in warp-fashion, with:
        - `tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;`
        - `warpid = tid / 32;`
    2. In our setting, `blockDim.z==1` and `blockDim.x=blockDim.y=32`. Thus we could ensure all threads in a warp `share the same threadIdx.y` while `have different threadIdx.x`
    3. In the slower kernel, globally accessing array `A[]` will be a catastrophe, since we will loading 32 different rows; But in the faster kernel, in each warp (and for each same instruction), all `A[]/B[]/C[]` will __lie in consecutive segments__ (or just even the same element at all)

### 2. Comparing 2x2 shared memory kernels (under A100-80GB, with block size 32x32)

|  Global \ Shared   | 1 | 2 |
| :--------: | :-------: | :-------: | 
| 1  | 0.5802 ms  | 4.1340 ms |
| 2 | 4.7111 ms    | 4.6079 ms |

- Shared:
    1. ```c
        for (int i = 0; i < reduce_dim; i++){ 
            const int col_a = col_offset + i * K_TILE_SIZE;
            const int row_b = row_offset + i * K_TILE_SIZE;
            A_shared[row_offset][col_offset] = (row < M && col_a < K) ? A[row * K + col_a] : (half)0.0;
            B_shared[row_offset][col_offset] = (row_b < K && col < N) ? B[row_b * N + col] : (half)0.0;
            __syncthreads();
            for (int j = 0; j < K_TILE_SIZE; j++)
            ret += __half2float(A_shared[row_offset][j]) * __half2float(B_shared[j][col_offset]);
            __syncthreads();
        }
       ```
    2. ```c
        for (int i = 0; i < reduce_dim; i++){ 
            const int col_a = col_offset + i * K_TILE_SIZE;
            const int row_b = row_offset + i * K_TILE_SIZE;
            A_shared[row_offset][col_offset] = (row < M && col_a < K) ? A[row * K + col_a] : (half)0.0;
            B_shared[col_offset][row_offset] = (row_b < K && col < N) ? B[row_b * N + col] : (half)0.0;
            __syncthreads();
            for (int j = 0; j < K_TILE_SIZE; j++)
            ret += __half2float(A_shared[row_offset][j]) * __half2float(B_shared[col_offset][j]);
            __syncthreads();
        }
       ```

- Global: 
    1. ```c
        const int row_offset = threadIdx.y;
        const int col_offset = threadIdx.x;
        const int row = blockIdx.y * blockDim.y + row_offset;
        const int col = blockIdx.x * blockDim.x + col_offset;
       ```
    2. ```c
        const int row_offset = threadIdx.x;
        const int col_offset = threadIdx.y;
        const int row = blockIdx.x * blockDim.x + row_offset;
        const int col = blockIdx.y * blockDim.y + col_offset;
       ```

- Solution: 
    1. `Shared == 1` && `Global == 1`:
        - Perfect in `Initializing A_shared[]/B_shared[]` (both shared and global)
        - Perfect in `Fetching A_shared[]/B_shared[] in loop`
    2. `Shared == 1` && `Global == 2`:
        - Bad in `Initializing A_shared[]/B_shared[]` (both shared and global)
        - Bad in `Fetching A_shared[]/B_shared[] in loop`
    3. `Shared == 2` && `Global == 1`:
        - Perfect in `Initializing A_shared[]/B_shared[]` of gloabl
        - Bad in one of the shared array in `Initializing A_shared[]/B_shared[]`, but another is ___perfect___
        - Bad in `Fetching A_shared[]/B_shared[] in loop`
    4. `Shared == 2` && `Global == 2`:
        - Bad in `Initializing A_shared[]/B_shared[]` of gloabl
        - Bad in one of the shared array in `Initializing A_shared[]/B_shared[]`, but another is ___perfect___
        - Bad in `Fetching A_shared[]/B_shared[] in loop`
    - Rank the influencer (from higher to lower):
        1. `Fetching A_shared[]/B_shared[] in loop`
        2. `Initializing A_shared[]/B_shared[]` of gloabl
        3. `Initializing A_shared[]/B_shared[]` of shared


