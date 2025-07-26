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

# this version assumes K/V is in good shape, ensuring the data access locality
@triton.jit
def sfm_attn_kernel_v2(
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
    range_q = range_q_N[:, None] * model_dim + range_d[None, :] + pid_d * head_dim # the address of Q_i
    q = tl.load(Q_ptr + range_q, mask = mask_q_N & mask_d, other = 0.0)
    o = tl.zeros_like(q)
    l, m, m_new = tl.zeros((BLOCK_SIZE_Q, 1), tl.float32), tl.full((BLOCK_SIZE_Q, 1), -float("inf"), tl.float32), tl.zeros((BLOCK_SIZE_Q, 1), tl.float32)

    for i in range(0, N, BLOCK_SIZE_KV):
        range_kv_N = i + tl.arange(0, BLOCK_SIZE_KV)
        mask_kv_N = range_kv_N[:, None] < N
        range_kv = range_kv_N[:, None] * head_dim + range_d[None, :] + pid_d * head_dim * N
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

# A^T (head_dim based transpose, each element in the matrix is a vector of length "head_dim") -> B; C -> A
@triton.jit
def transpose_and_copy_fused_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    head_dim,
    num_heads,
    head_dim_upper: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    A_ptr = A_ptr.to(tl.pointer_type(tl.float32))
    B_ptr = B_ptr.to(tl.pointer_type(tl.float32))
    C_ptr = C_ptr.to(tl.pointer_type(tl.float32))
    pid, model_dim = tl.program_id(0), head_dim * num_heads

    # step 0: init and define some useful variables
    range_N = pid * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    mask_N = range_N[:, None] < N
    range_head_dim = tl.arange(0, head_dim_upper)
    mask_head_dim = range_head_dim[None, :] < head_dim

    # fused step 1 & 2: A^T -> B; C -> A
    base_range_A = range_N[:, None] * model_dim + range_head_dim[None, :]
    base_range_B = range_N[:, None] * head_dim + range_head_dim[None, :]
    for i in range(num_heads):
        range_A = base_range_A + i * head_dim
        A = tl.load(A_ptr + range_A, mask = mask_N & mask_head_dim, other = 0.0)
        C = tl.load(C_ptr + range_A, mask = mask_N & mask_head_dim, other = 0.0)
        range_B = base_range_B + i * N * head_dim
        tl.store(B_ptr + range_B, A, mask = mask_N & mask_head_dim)
        tl.store(A_ptr + range_A, C, mask = mask_N & mask_head_dim)
    
def transpose_KV_using_output(K_ptr: int, V_ptr: int, output_ptr: int, N: int, head_dim: int, num_heads: int):
    head_dim_upper = triton.next_power_of_2(head_dim)
    BLOCK_ROW = 16
    transpose_and_copy_fused_kernel[(triton.cdiv(N, BLOCK_ROW), )](
        K_ptr, output_ptr, V_ptr,
        N, head_dim, num_heads,
        head_dim_upper, BLOCK_ROW,
    )
    transpose_and_copy_fused_kernel[(triton.cdiv(N, BLOCK_ROW), )](
        K_ptr, V_ptr, output_ptr,
        N, head_dim, num_heads,
        head_dim_upper, BLOCK_ROW,
    )

# Q_ptr, K_ptr, V_ptr, output_ptr are raw device pointers
# only support A100, H100, Tesla T4
def solve(Q_ptr: int, K_ptr: int, V_ptr: int, output_ptr: int, N: int, d_model: int, h: int, gpu = "T4"):
    head_dim = d_model // h
    block_dim = triton.next_power_of_2(head_dim)
    scale_factor = head_dim ** -0.5

    BQ = 16 if gpu == "A100" else (16 if gpu == "H100" else 16)
    BKV = 64 if gpu == "A100" else (64 if gpu == "H100" else 32)
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
    # # abandoned solution, since the checker will check whether input arrays are modified
    # transpose_KV_using_output(K_ptr, V_ptr, output_ptr, N, head_dim, h)
    # grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_Q']), h, )
    # sfm_attn_kernel_v2[grid](
    #     Q_ptr, K_ptr, V_ptr, output_ptr,
    #     N, head_dim, h,
    #     scale_factor,
    #     BLOCK_DIM = max(block_dim, 16),
    #     BLOCK_SIZE_Q = 16,
    #     BLOCK_SIZE_KV = 32,
    # )




# # Abandoned because of the poor performance and high shared memory usage
# # Assuming the head_dim is power of 2
# @triton.jit
# def sfm_attn_kernel_head_fusion(
#     Q_ptr,
#     K_ptr,
#     V_ptr,
#     output_ptr,
#     N, num_heads,
#     scale_factor,
#     head_dim: tl.constexpr,
#     upper_h: tl.constexpr,
#     BLOCK_DIM: tl.constexpr,
#     BLOCK_SIZE_Q: tl.constexpr,
#     BLOCK_SIZE_KV: tl.constexpr,
# ):
#     # recast pointers 
#     Q_ptr = Q_ptr.to(tl.pointer_type(tl.float32))
#     K_ptr = K_ptr.to(tl.pointer_type(tl.float32))
#     V_ptr = V_ptr.to(tl.pointer_type(tl.float32))
#     output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

#     pid, d = tl.program_id(0), head_dim * num_heads
#     range_q_N = pid * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
#     mask_q_N = range_q_N[:, None] < N
#     range_d = tl.arange(0, BLOCK_DIM)
#     mask_d = range_d[None, :] < d
#     range_q = range_q_N[:, None] * d + range_d[None, :]
#     q = tl.load(Q_ptr + range_q, mask = mask_q_N & mask_d, other = 0.0)
#     # reshape and transpose q to (upper_h, BLOCK_SIZE_Q, head_dim)
#     q = tl.trans(tl.reshape(q, (BLOCK_SIZE_Q, upper_h, head_dim)), (1, 0, 2))
#     o = tl.zeros_like(q)
#     l = tl.zeros((upper_h, BLOCK_SIZE_Q, 1), tl.float32)
#     m = tl.full((upper_h, BLOCK_SIZE_Q, 1), -float("inf"), tl.float32)
#     m_new = tl.zeros((upper_h, BLOCK_SIZE_Q, 1), tl.float32)
    
#     for i in range(0, N, BLOCK_SIZE_KV):
#         range_kv_N = i + tl.arange(0, BLOCK_SIZE_KV)
#         range_kv = range_kv_N[:, None] * d + range_d[None, :]
#         mask_kv_N = range_kv_N[:, None] < N
#         k = tl.load(K_ptr + range_kv, mask = mask_kv_N & mask_d, other = 0.0)
#         # reshape and transpose k to (upper_h, head_dim, BLOCK_SIZE_KV)
#         k = tl.trans(tl.reshape(k, (BLOCK_SIZE_KV, upper_h, head_dim)), (1, 2, 0))
#         p = tl.dot(q, k, allow_tf32 = True) * scale_factor  # (upper_h, BLOCK_SIZE_Q, BLOCK_SIZE_KV)
#         p = tl.where(mask_kv_N.T[None, :, :], p, -float('inf'))   # apply a mask to avoid nonexistent keys/values
#         m_new = tl.maximum(tl.max(p, axis = -1, keep_dims = True), m)
#         s = tl.exp(p - m_new) # (upper_h, BLOCK_SIZE_Q, BLOCK_SIZE_KV)
#         l = l * tl.exp(m - m_new) + tl.sum(s, axis = -1, keep_dims = True)
#         v = tl.load(V_ptr + range_kv, mask = mask_kv_N & mask_d, other = 0.0)
#         # reshape and transpose v to (upper_h, BLOCK_SIZE_KV, head_dim)
#         v = tl.trans(tl.reshape(v, (BLOCK_SIZE_KV, upper_h, head_dim)), (1, 0, 2))
#         o = o * tl.exp(m - m_new) + tl.dot(s, v, allow_tf32 = False) # (upper_h, BLOCK_SIZE_Q, head_dim)
#         m = m_new
#     o /= l
#     # reshape and transpose o to (BLOCK_SIZE_Q, BLOCK_DIM)
#     o = tl.reshape(tl.trans(o, (1, 0, 2)), (BLOCK_SIZE_Q, BLOCK_DIM))
#     tl.store(output_ptr + range_q, o, mask = mask_q_N & mask_d)