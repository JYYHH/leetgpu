# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl
import triton.runtime.driver as drv
import torch
import time

@triton.jit
def softmax_kernel_local(
    input_ptr, 
    output_ptr, 
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_THREAD: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    pid, total_programs = tl.program_id(axis = 0), tl.num_programs(axis = 0)
    offset_base = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    stride = total_programs * BLOCK_SIZE

    sum_local = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    max_local = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # sequential reduce
    for i in range(ELEMENTS_PER_THREAD):
        offset = offset_base + i * stride
        data = tl.load(input_ptr + offset, mask = offset < N, other = -float('inf'))
        new_max_local = tl.maximum(max_local, data)
        sum_local = sum_local * tl.exp(max_local - new_max_local) + tl.exp(data - new_max_local)
        max_local = new_max_local
    # block reduce
    max_block = tl.max(max_local, axis = 0)
    sum_block = tl.sum(sum_local * tl.exp(max_local - max_block), axis = 0)

    # save result
    tl.store(output_ptr + pid, sum_block)
    tl.store(output_ptr + pid + total_programs, max_block)

@triton.jit
def softmax_kernel_global(
    input_ptr,
    output_ptr, 
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    offset = tl.arange(0, BLOCK_SIZE)
    sum_global = tl.load(input_ptr + offset, mask = offset < N, other = 0.0)
    max_global = tl.load(input_ptr + offset + N, mask = offset < N, other = 0.0)
    max_ = tl.max(max_global, axis = 0)
    sum_ = tl.sum(sum_global * tl.exp(max_global - max_), axis = 0)
    tl.store(output_ptr, sum_)
    tl.store(output_ptr + 1, max_)

@triton.jit
def update_kernel(
    input_ptr,
    output_ptr, 
    sum_max_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_THREAD: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    pid, total_programs = tl.program_id(axis = 0), tl.num_programs(axis = 0)
    offset_base = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    stride = total_programs * BLOCK_SIZE
    sum_max_ptr = sum_max_ptr.to(tl.pointer_type(tl.float32))
    sum_global = tl.load(sum_max_ptr)
    max_global = tl.load(sum_max_ptr + 1)

    for i in range(ELEMENTS_PER_THREAD):
        offset = offset_base + i * stride
        data = tl.load(input_ptr + offset, mask = offset < N, other = 0.0)
        tl.store(output_ptr + offset, tl.exp(data - max_global) / sum_global, mask = offset < N)

# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int):
    BLOCK_SIZE = 1024
    ELEMENTS_PER_THREAD = 4
    ELEMENTS_PER_BLOCK = BLOCK_SIZE * ELEMENTS_PER_THREAD

    sum_max_global = drv.active.get_device_interface().caching_allocator_alloc(2 * 4)

    if N <= ELEMENTS_PER_BLOCK:
        softmax_kernel_local[(1, )](input_ptr, sum_max_global, N, BLOCK_SIZE, ELEMENTS_PER_THREAD)
    else:
        block_num = triton.cdiv(N, ELEMENTS_PER_BLOCK)
        softmax_kernel_local[(block_num, )](input_ptr, output_ptr, N, BLOCK_SIZE, ELEMENTS_PER_THREAD)
        softmax_kernel_global[(1, )](output_ptr, sum_max_global, block_num, BLOCK_SIZE)
    update_kernel[(triton.cdiv(N, ELEMENTS_PER_BLOCK), )](input_ptr, output_ptr, sum_max_global, N, BLOCK_SIZE, ELEMENTS_PER_THREAD)

    drv.active.get_device_interface().caching_allocator_delete(sum_max_global)

def main():
    N = 3
    input_ = torch.randn(N, dtype=torch.float32, device="cuda")
    # input_ = torch.tensor([1, 2, 3], dtype=torch.float32, device="cuda")
    output_ = torch.empty(N, dtype=torch.float32, device="cuda")
    solve(input_.data_ptr(), output_.data_ptr(), N)
    manual_output = torch.softmax(input_, dim = 0)
    if torch.allclose(output_, manual_output, atol = 1e-4):
        print("Test passed")
        print(output_)
        print(manual_output)
    else:
        print(output_)
        print(manual_output)
        print("Test failed")

if __name__ == "__main__":
    main()