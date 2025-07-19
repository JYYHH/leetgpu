## [LeetGPU Sorting](https://leetgpu.com/challenges/sorting)
- The main idea is implementing the [Bitonic sorter](https://en.wikipedia.org/wiki/Bitonic_sorter) on CUDA GPUs. The parallel algorithm has work $O(nlog^2\ n)$ and depth $O(log^2\ n)$.
- All the speeds showed below assume data size = $2^{20}\approx 10^6$, and under NVIDIA A5000
### V0: Initial Version (commit 4c76ab513045193f1cbf9172bb441bcb65bf1983)
- speed: 1.470 ms 

### V1: Add shared memory optimizaion for local sort (commit 3f7887e2517e83c4df34e7eaf9f94a983f8dbc83)
- speed: 1.370 ms

### V2: Optimize one logic (pre-calculate the increase or decrease for the current thread) (commit fea3d563bf1431f0dda4a2dd3cb4f5027b57a6c9)
- speed: 1.355 ms

### V3.0: Upgrade the thread usage (now only for local sort) (commit 8ffac865736d242e02cca7b31eeb0f2a101ab1c7) 
- speed: 1.338 ms

### V3.1: Upgrade the thread usage (for all kernels) (commit 13ab4c2abb4ff88ac9d4e26526a455d2cfb3d9d1)
- speed: 1.170 ms

## 1. Bast commit for both NVIDIA TESLA T4 & A100-80GB: 86166bdd0f86a9b0b753a37faa73c0ee7923a6baa

## -1. Conclusion
- (The main bottleneck is the massive kernel launchs and global synchronizations...)




