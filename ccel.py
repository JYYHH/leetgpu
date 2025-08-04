import triton
import triton.language as tl

@triton.jit
def init_kernel(loss_ptr):
    loss_ptr = loss_ptr.to(tl.pointer_type(tl.float32))
    tl.store(loss_ptr, 0.0)

@triton.jit
def ccel_kernel(
    logits_ptr,
    true_labels_ptr,
    loss_ptr,
    N, C,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # recast pointers
    logits_ptr = logits_ptr.to(tl.pointer_type(tl.float32))
    true_labels_ptr = true_labels_ptr.to(tl.pointer_type(tl.int32))
    loss_ptr = loss_ptr.to(tl.pointer_type(tl.float32))

    # get the block index
    block_idx = tl.program_id(0)
    range_n = tl.arange(0, BLOCK_SIZE_N) + block_idx * BLOCK_SIZE_N
    mask_n = range_n < N
    range_c = tl.arange(0, BLOCK_SIZE_C)
    mask_c = range_c < C

    # load the logits and true labels
    logits = tl.load(logits_ptr + range_n[:, None] * C + range_c[None, :], mask = mask_n[:, None] & mask_c[None, :], other = -float("inf"))
    true_labels = tl.load(true_labels_ptr + range_n, mask = mask_n, other = BLOCK_SIZE_C) # BLOCK_SIZE_C is a dummy value to avoid nan
    # compute loss
    exp_sum = tl.sum(tl.exp(logits), axis = -1)
    exp_sum = tl.where(mask_n, exp_sum, 1.0) # avoid nan
    loss = tl.log(exp_sum) - tl.sum(tl.where(range_c[None, :] == true_labels[:, None], logits, 0.0), axis = -1)
    # reduce loss
    local_loss = tl.sum(loss) / N
    # store global loss
    tl.atomic_add(loss_ptr, local_loss)

def solve(logits_ptr: int, true_labels_ptr: int, loss_ptr: int, N: int, C: int):
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_C = triton.next_power_of_2(C)
    init_kernel[(1, )](loss_ptr)
    ccel_kernel[(triton.cdiv(N, BLOCK_SIZE_N), )](
        logits_ptr, 
        true_labels_ptr, 
        loss_ptr, N, C, 
        BLOCK_SIZE_N = BLOCK_SIZE_N, 
        BLOCK_SIZE_C = BLOCK_SIZE_C
    )