## [LeetGPU Softmax](https://leetgpu.com/challenges/softmax)

- The idea is simple: We first do a local reduction and then a global one to get the:
    1. max = $m = \max_{i=0}^{N-1} a_i$
    2. sum = $s = \sum_{i=0}^{N-1} e^{a_i - m}$
- Then finally we can compute the softmax result using $s$ and $m$ in another kernel:
    $$
    \text{softmax}(a) = \frac{e^{a_i - m}}{s}
    $$
- And we could use some math tricks to fuse the max and sum reduction into a single kernel. (See `sfm.py` for more details)


