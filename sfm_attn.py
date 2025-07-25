# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def sfm_attn_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    output_ptr,
    M,
    N,
    d,
    scale_factor,
    BLOCK_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    # recast pointers 
    Q_ptr = Q_ptr.to(tl.pointer_type(tl.float32))
    K_ptr = K_ptr.to(tl.pointer_type(tl.float32))
    V_ptr = V_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    
    range_q_M = pid * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    mask_q_M = range_q_M[:, None] < M
    range_d = tl.arange(0, BLOCK_DIM)
    mask_d = range_d[None, :] < d
    range_q = range_q_M[:, None] * d + range_d[None, :]
    q = tl.load(Q_ptr + range_q, mask = mask_q_M & mask_d, other = 0.0)
    o = tl.zeros_like(q)
    l, m, m_new = tl.zeros((BLOCK_SIZE_Q, 1), tl.float32), tl.full((BLOCK_SIZE_Q, 1), -float("inf"), tl.float32), tl.zeros((BLOCK_SIZE_Q, 1), tl.float32)

    for i in range(0, N, BLOCK_SIZE_KV):
        range_kv_N = i + tl.arange(0, BLOCK_SIZE_KV)
        range_kv = range_kv_N[:, None] * d + range_d[None, :]
        mask_kv_N = range_kv_N[:, None] < N
        k = tl.load(K_ptr + range_kv, mask = mask_kv_N & mask_d, other = 0.0)
        p = tl.dot(q, tl.trans(k), allow_tf32 = True) * scale_factor 
        p = tl.where(mask_kv_N.T, p, -float('inf'))   # apply a mask to avoid nonexistent keys/values
        m_new = tl.maximum(tl.max(p, axis = 1, keep_dims = True), m)
        s = tl.exp(p - m_new)
        l = l * tl.exp(m - m_new) + tl.sum(s, axis = 1, keep_dims = True)
        v = tl.load(V_ptr + range_kv, mask = mask_kv_N & mask_d, other = 0.0)
        o = o * tl.exp(m - m_new) + tl.dot(s, v, allow_tf32 = False)
        m = m_new
    o /= l
    tl.store(output_ptr + range_q, o, mask = mask_q_M & mask_d)


# Q_ptr, K_ptr, V_ptr, output_ptr are raw device pointers
def solve(Q_ptr: int, K_ptr: int, V_ptr: int, output_ptr: int, M: int, N: int, d: int):
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_Q']), ) 
    scale_factor = d ** -0.5
    sfm_attn_kernel[grid](Q_ptr, K_ptr, V_ptr, output_ptr, M, N, d, scale_factor, BLOCK_DIM = max(triton.next_power_of_2(d), 16), BLOCK_SIZE_Q = 32, BLOCK_SIZE_KV = 32)

if __name__ == "__main__":
    import torch
    import time
    # M, N, d = 1 << 15, 1 << 15, 128
    # Q = torch.randn(M, d, device = "cuda")
    # K = torch.randn(N, d, device = "cuda")
    # V = torch.randn(N, d, device = "cuda")
    M, N, d = 2, 3, 4
    Q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device = "cuda")
    K = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], device = "cuda")
    V = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], device = "cuda")
    output = torch.empty((M, d), device = "cuda")
    start = time.time()
    solve(Q.data_ptr(), K.data_ptr(), V.data_ptr(), output.data_ptr(), M, N, d)
    torch.cuda.synchronize()
    end = time.time()
    print(f'The time taken is {end - start}')

    output_pytorch = torch.softmax(Q @ K.T / d ** 0.5, dim = 1) @ V
    print(torch.allclose(output, output_pytorch, atol = 1e-4))
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output - output_pytorch))}')
    # print(f'The output is {output}')
    # print(f'The output_pytorch is {output_pytorch}')