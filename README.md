## [LeetGPU Softmax Attention](https://leetgpu.com/challenges/softmax-attention)
- In this branch, I implement the softmax attention triton kernel based on `Flash Attention v2`.

## Flash Attention v2
- The algorithm is shown as follows: (Actually below is the algorithm mapping from $Q_i$ to $O_i$, and since the mappings are totally independent, we can parallelize the algorithm for each $Q_i,\ i\in [0, \lceil N/B_r\rceil)$ on different thread blocks)
- Note:
    1. $N$ is the sequence length, $d$ is the dimension of the query, key, value.
    2. $B_r$ is the block size of the query, $B_c$ is the block size of the key and value.
    3. $T_c$ is the number of blocks of the key and value. 

$
\begin{array}{l}
\hline
\textbf{Algorithm 1: FlashAttention-2 Forward Pass} \\
\hline
\textbf{Require: } \text{Matrices } Q, K, V \in \mathbb{R}^{N \times d} \text{ located in HBM.} \\
\textbf{Parameters: } \text{Block sizes } B_r, B_c. \\
\text{01: } \text{Let a thread block handle block } i \text{ of } Q. \\
\text{02: } Q_i \leftarrow \text{Load a } B_r \times d \text{ block from } Q \text{ into SRAM.} \\
\text{03: } O_i \leftarrow \mathbf{0} \in \mathbb{R}^{B_r \times d}; \quad m_i \leftarrow -\infty \in \mathbb{R}^{B_r}; \quad l_i \leftarrow \mathbf{0} \in \mathbb{R}^{B_r}. \\
\text{04: } T_c \leftarrow \lceil N / B_c \rceil. \\
\text{05: } \textbf{for } j=1 \textbf{ to } T_c \textbf{ do} \\
\text{06: } \quad K_j \leftarrow \text{Load a } B_c \times d \text{ block from } K \text{ into shared memory.} \\
\text{07: } \quad V_j \leftarrow \text{Load a } B_c \times d \text{ block from } V \text{ into shared memory.} \\
\text{08: } \quad S_{ij} \leftarrow \frac{1}{\sqrt{d}} Q_i K_j^T. \quad \textit{// On-chip computation.} \\
\text{09: } \quad m_i^{\text{new}} \leftarrow \max(m_i, \text{rowmax}(S_{ij})). \\
\text{10: } \quad P_{ij} \leftarrow \exp(S_{ij} - m_i^{\text{new}}). \\
\text{11: } \quad l_i \leftarrow e^{m_i - m_i^{\text{new}}} l_i + \text{rowsum}(P_{ij}). \\
\text{12: } \quad O_i \leftarrow \text{diag}(e^{m_i - m_i^{\text{new}}}) O_i + P_{ij} V_j. \\
\text{13: } \quad m_i \leftarrow m_i^{\text{new}}. \\
\text{14: } \textbf{end for} \\
\text{15: } O_i \leftarrow \text{diag}(l_i)^{-1} O_i. \quad \textit{// Final normalization, done in registers.} \\
\text{16: } \text{Write block } O_i \text{ from registers back to HBM.} \\
\hline
\end{array}
$