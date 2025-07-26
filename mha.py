import triton
import triton.language as tl

@triton.jit
def sfm_attn_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    output_ptr,
    N, head_dim, num_heads,
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

    pid_q, pid_d = tl.program_id(0), tl.program_id(1)
    model_dim = head_dim * num_heads
    
    range_q_N = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    mask_q_N = range_q_N[:, None] < N
    range_d = tl.arange(0, BLOCK_DIM)
    mask_d = range_d[None, :] < head_dim
    range_d += pid_d * head_dim
    range_q = range_q_N[:, None] * model_dim + range_d[None, :] # the address of Q_i
    q = tl.load(Q_ptr + range_q, mask = mask_q_N & mask_d, other = 0.0)
    o = tl.zeros_like(q)
    l, m, m_new = tl.zeros((BLOCK_SIZE_Q, 1), tl.float32), tl.full((BLOCK_SIZE_Q, 1), -float("inf"), tl.float32), tl.zeros((BLOCK_SIZE_Q, 1), tl.float32)

    for i in range(0, N, BLOCK_SIZE_KV):
        range_kv_N = i + tl.arange(0, BLOCK_SIZE_KV)
        range_kv = range_kv_N[:, None] * model_dim + range_d[None, :]
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
    tl.store(output_ptr + range_q, o, mask = mask_q_N & mask_d)


# Assuming the head_dim is power of 2
@triton.jit
def sfm_attn_kernel_head_fusion(
    Q_ptr,
    K_ptr,
    V_ptr,
    output_ptr,
    N, num_heads,
    scale_factor,
    head_dim: tl.constexpr,
    upper_h: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    # recast pointers 
    Q_ptr = Q_ptr.to(tl.pointer_type(tl.float32))
    K_ptr = K_ptr.to(tl.pointer_type(tl.float32))
    V_ptr = V_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

    pid, d = tl.program_id(0), head_dim * num_heads
    range_q_N = pid * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    mask_q_N = range_q_N[:, None] < N
    range_d = tl.arange(0, BLOCK_DIM)
    mask_d = range_d[None, :] < d
    range_q = range_q_N[:, None] * d + range_d[None, :]
    q = tl.load(Q_ptr + range_q, mask = mask_q_N & mask_d, other = 0.0)
    # reshape and transpose q to (upper_h, BLOCK_SIZE_Q, head_dim)
    q = tl.trans(tl.reshape(q, (BLOCK_SIZE_Q, upper_h, head_dim)), (1, 0, 2))
    o = tl.zeros_like(q)
    l = tl.zeros((upper_h, BLOCK_SIZE_Q, 1), tl.float32)
    m = tl.full((upper_h, BLOCK_SIZE_Q, 1), -float("inf"), tl.float32)
    m_new = tl.zeros((upper_h, BLOCK_SIZE_Q, 1), tl.float32)
    
    for i in range(0, N, BLOCK_SIZE_KV):
        range_kv_N = i + tl.arange(0, BLOCK_SIZE_KV)
        range_kv = range_kv_N[:, None] * d + range_d[None, :]
        mask_kv_N = range_kv_N[:, None] < N
        k = tl.load(K_ptr + range_kv, mask = mask_kv_N & mask_d, other = 0.0)
        # reshape and transpose k to (upper_h, head_dim, BLOCK_SIZE_KV)
        k = tl.trans(tl.reshape(k, (BLOCK_SIZE_KV, upper_h, head_dim)), (1, 2, 0))
        p = tl.dot(q, k, allow_tf32 = True) * scale_factor  # (upper_h, BLOCK_SIZE_Q, BLOCK_SIZE_KV)
        p = tl.where(mask_kv_N.T[None, :, :], p, -float('inf'))   # apply a mask to avoid nonexistent keys/values
        m_new = tl.maximum(tl.max(p, axis = -1, keep_dims = True), m)
        s = tl.exp(p - m_new) # (upper_h, BLOCK_SIZE_Q, BLOCK_SIZE_KV)
        l = l * tl.exp(m - m_new) + tl.sum(s, axis = -1, keep_dims = True)
        v = tl.load(V_ptr + range_kv, mask = mask_kv_N & mask_d, other = 0.0)
        # reshape and transpose v to (upper_h, BLOCK_SIZE_KV, head_dim)
        v = tl.trans(tl.reshape(v, (BLOCK_SIZE_KV, upper_h, head_dim)), (1, 0, 2))
        o = o * tl.exp(m - m_new) + tl.dot(s, v, allow_tf32 = False) # (upper_h, BLOCK_SIZE_Q, head_dim)
        m = m_new
    o /= l
    # reshape and transpose o to (BLOCK_SIZE_Q, BLOCK_DIM)
    o = tl.reshape(tl.trans(o, (1, 0, 2)), (BLOCK_SIZE_Q, BLOCK_DIM))
    tl.store(output_ptr + range_q, o, mask = mask_q_N & mask_d)
    


# Q_ptr, K_ptr, V_ptr, output_ptr are raw device pointers
# only support A100, H100, Tesla T4
def solve(Q_ptr: int, K_ptr: int, V_ptr: int, output_ptr: int, N: int, d_model: int, h: int, gpu = "T4"):
    head_dim = d_model // h
    block_dim = triton.next_power_of_2(head_dim)
    scale_factor = head_dim ** -0.5

    if block_dim > head_dim or head_dim < 16:
        BQ = 16 if gpu == "A100" else (16 if gpu == "H100" else 16)
        BKV = 64 if gpu == "A100" else (64 if gpu == "H100" else 64)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_Q']), h, ) 
        sfm_attn_kernel[grid](
            Q_ptr, K_ptr, V_ptr, 
            output_ptr, 
            N, head_dim, h, 
            scale_factor, 
            BLOCK_DIM = max(block_dim, 16), 
            BLOCK_SIZE_Q = BQ, 
            BLOCK_SIZE_KV = BKV,
        )
    else:
        BQ = 16 if gpu == "A100" else (16 if gpu == "H100" else 16)
        BKV = 64 if gpu == "A100" else (64 if gpu == "H100" else 16)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_Q']), ) 
        sfm_attn_kernel_head_fusion[grid](
            Q_ptr, K_ptr, V_ptr, 
            output_ptr, 
            N, h, 
            scale_factor, 
            head_dim = head_dim,
            upper_h = triton.next_power_of_2(h),
            BLOCK_DIM = block_dim * triton.next_power_of_2(h), 
            BLOCK_SIZE_Q = BQ, 
            BLOCK_SIZE_KV = BKV,
        )

if __name__ == "__main__":
    import torch
    from torch import nn
    import time
    N, d_model, h = 1 << 12, 128, 8
    Q = torch.randn(N, d_model, device = "cuda")
    K = torch.randn(N, d_model, device = "cuda")
    V = torch.randn(N, d_model, device = "cuda")
    # N, d_model, h = 2, 4, 2
    # Q = torch.tensor([[1.0, 0.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], device = "cuda")
    # K = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device = "cuda")
    # V = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]], device = "cuda")
    output = torch.empty((N, d_model), device = "cuda")
    start = time.time()
    solve(Q.data_ptr(), K.data_ptr(), V.data_ptr(), output.data_ptr(), N, d_model, h)
    torch.cuda.synchronize()
    end = time.time()
    print(f'The time taken is {end - start}')

    output_pytorch = torch.empty((N, d_model), device = "cuda")
    head_dim = d_model // h
    for i in range(h):
        output_pytorch[:, i * head_dim : (i + 1) * head_dim] = torch.softmax(Q[:, i * head_dim : (i + 1) * head_dim] @ K[:, i * head_dim : (i + 1) * head_dim].T / head_dim ** 0.5, dim = 1) @ V[:, i * head_dim : (i + 1) * head_dim]

    print(torch.allclose(output, output_pytorch, atol = 1e-4))
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output - output_pytorch))}')
    # print(f'The output is {output}')
    # print(f'The output_pytorch is {output_pytorch}')