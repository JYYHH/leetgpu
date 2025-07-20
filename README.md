## [LeetGPU Scan](https://leetgpu.com/challenges/prefix-sum)
- Main idea is to implement a 3-level scan, with bottom 2 levels parallel (and local) scan and top level serial scan.

### Parallel Local Scan
- Assume we need to process the local prefix sum within a block ( with size $n=2^k$ ).
- Communication Pattern (tids):
    1. Round 0: $0\rightarrow 1$, $2\rightarrow 3$, ... ( $2t\rightarrow 2t+1$ )
    2. Round 1: $1\rightarrow 2$, $1\rightarrow 3$, ... ( $4t+1\rightarrow (4t+2, 4t+3)$ )
    3. Round 2: ( $8t+3\rightarrow (8t+4, 8t+5, 8t+6, 8t+7)$ )
    4. ...
- In conclusion, the $i$-th (starting from 0) recving side node from left at Round $d$-th is $$r_{d, i}=i + \lfloor\frac{i}{2^d} + 1\rfloor 2^{d}$$ While the correspond left conjugation node is $$l_{d, i} = r_{d, i} - 2^d=i + \lfloor\frac{i}{2^d}\rfloor 2^{d}$$ Finally the true sending node on the left is $$s_{d, i}=\lfloor\frac{l_{d, i}}{2^d}\rfloor 2^d + (2^d - 1)=l_{d, i} + [(-(i + 1))\ mod\ 2^d]$$
- And in the implementation, right_id and left_id are just $r_{d, tid}$ and $s_{d, tid}$, with the high-performance bit-operation for left_id.

### Multi (Triple)-Level Scan
- Assume the block_size of the bottom level is $blk_1=BLOCK\_ELEMENTS\_1$ and $blk_2=BLOCK\_ELEMENTS\_2$ for the level above it, and the input array $a[]$ with size $N$.
    1. After the first parallel local scan (the bottom level scan, which is the `scan_kernel<BLOCK_SIZE_1><<<blk_num_1s, BLOCK_SIZE_1>>>(input, output, output_1s, N);` in the implementation), we have $s[i] = \sum_{j = \lfloor\frac{i}{blk_1}\rfloor blk_1} ^ i a[i]$ and $sum[i] = \sum_{j=i*blk_1}^{(i+1) * blk_1 - 1} a[i] = s[(i+1) * blk_1 - 1]$. Actually, in code, output[] and output1s[] are just $s[]$ and $sum[]$.
    2. Second layer's paralle scan (`scan_kernel<BLOCK_SIZE_2><<<blk_num_2s, BLOCK_SIZE_2>>>(output_1s, output_1s, output_2s, blk_num_1s);`) is just identical, but we could set a different block size. That is (and after that, output_1s[] and output_2s[] are just $sums[]$ and $sumsum[]$):
        - $sums[i] = \sum_{j = \lfloor\frac{i}{blk_2}\rfloor blk_2} ^ i sum[i]$
        - $sumsum[i] = \sum_{j=i*blk_2}^{(i+1) * blk_2 - 1} sum[i] = sums[(i+1) * blk_2 - 1]$
    3. Final layer's serial scan is just trivial, with $sumsums[]$ is the prefix of $sumsum[]$. Since now we only have $\lceil\frac{\lceil\frac{N}{blk_1}\rceil}{blk_2}\rceil = \lceil\frac{N}{blk_1 * blk_2}\rceil$ elements to handle with.
    4. In the end, we need to add back all the contribution: $\forall\ i,\ s[i] + (\lfloor\frac{i}{blk_1}\rfloor > 0\ (mod\ blk_2) \ ?\  sums[\lfloor\frac{i}{blk_1}\rfloor - 1] : 0) + (\lfloor\frac{i}{blk_1 * blk_2}\rfloor > 0 \ ?\  sumsums[\lfloor\frac{i}{blk_1 * blk_2}\rfloor - 1] : 0) = \sum_{j=0}^i a[i]$ is what we want, and it's just what the kernel call `walk_back_kernel<<<blk_num_1s, BLOCK_ELEMENTS_1>>>(BLOCK_ELEMENTS_2, output, output_1s, output_2s, N);` is doing.

### Best Config
#### for NVIDIA TESLA T4
- commit: 85ab17464a6560a8a3ca5e6456078b1ba2557e27
- BLOCK_SIZE_1: 512
- BLOCK_SIZE_2: 512

#### for NVIDIA A100-80GB
- commit: 85ab17464a6560a8a3ca5e6456078b1ba2557e27
- BLOCK_SIZE_1: 256
- BLOCK_SIZE_2: 64
