## [LeetGPU Histogramming](https://leetgpu.com/challenges/histogramming)
- 2025-July-16th, rank 2nd on the Leaderboard!
- Basic Idea: 
    1. each block maintains a local bin array, which will be added to the global one in the end of the kernel.
    2. each thread is responsible for counting `ELEMENT_PER_THREAD` elements in original array.
    3. We use an int4 for each value, to avoid thread conflicts (see `atomicAdd(s_hist + ((input[i] << 2) | (tid & 3)), 1);`)



