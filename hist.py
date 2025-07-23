# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl
import torch
import time

@triton.jit
def hist_kernel(
    input_ptr,
    histogram_ptr,
    N: tl.constexpr,
    num_bins: tl.constexpr,
    up_num_bins: tl.constexpr,
    B: tl.constexpr,
    E: tl.constexpr,
):
    pid, total_programs = tl.program_id(axis = 0), tl.num_programs(axis = 0)
    bins = tl.zeros([up_num_bins], dtype=tl.int32) # shape: (num_bins,)
    offset_base = tl.arange(0, B) + pid * B
    stride = total_programs * B
    input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.int32))
    histogram_ptr = tl.cast(histogram_ptr, tl.pointer_type(tl.int32))

    for i in range(E):
        offset = offset_base + i * stride
        nums = tl.load(input_ptr + offset, mask = offset < N, other = 0)
        bins += tl.histogram(nums, up_num_bins)

    tl.atomic_add(histogram_ptr + tl.arange(0, up_num_bins), bins, mask = tl.arange(0, up_num_bins) < num_bins)
    if pid == 0:
        tl.atomic_add(histogram_ptr, N - total_programs * B * E)

@triton.jit
def set_zero_kernel(
    histogram_ptr,
    num_bins: tl.constexpr,
    up_num_bins: tl.constexpr,
):
    pid = tl.program_id(axis = 0)
    histogram_ptr = tl.cast(histogram_ptr, tl.pointer_type(tl.int32))
    tl.store(histogram_ptr + tl.arange(0, up_num_bins), tl.zeros([up_num_bins], dtype=tl.int32), mask = tl.arange(0, up_num_bins) < num_bins)

# input_ptr, histogram_ptr are raw device pointers
def solve(input_ptr: int, histogram_ptr: int, N: int, num_bins: int):
    B, E, up_num_bins = 256, 8, triton.next_power_of_2(num_bins)
    set_zero_kernel[(1,)](histogram_ptr, num_bins = num_bins, up_num_bins = up_num_bins)
    grid = (triton.cdiv(N, B * E), )
    hist_kernel[grid](input_ptr, histogram_ptr, N = N, num_bins = num_bins, up_num_bins = up_num_bins, B = B, E = E)
    
# # warmup
warmup_N, warmup_num_bins = 1000000, 1024
warmup_input = torch.randint(0, warmup_num_bins, (warmup_N,), device = 'cuda', dtype = torch.int32)
warmup_histogram = torch.zeros(warmup_num_bins, device = 'cuda', dtype = torch.int32)
# for i in range(10):
solve(warmup_input.data_ptr(), warmup_histogram.data_ptr(), warmup_N, warmup_num_bins)
torch.cuda.synchronize()
# benchmark
N, num_bins = 100000000, 1024   
input = torch.randint(0, num_bins, (N,), device = 'cuda', dtype = torch.int32)
histogram = torch.zeros(num_bins, device = 'cuda', dtype = torch.int32)
time_start = time.time()
solve(input.data_ptr(), histogram.data_ptr(), N, num_bins)
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
print(histogram)
print(torch.sum(histogram))