from ccel import solve
import torch

N = 10000
C = 1000

logits = torch.randn(N, C, device = "cuda")
# logits = torch.arange(N * C, device = "cuda", dtype = torch.float32).reshape(N, C)
true_labels = torch.randint(0, C, (N,), device = "cuda", dtype = torch.int32)
loss = torch.zeros(1, device = "cuda")
solve(logits.data_ptr(), true_labels.data_ptr(), loss.data_ptr(), N, C)
loss_pytorch = torch.nn.functional.cross_entropy(logits, true_labels.long())
# print(logits, true_labels)
print(loss.item(), loss_pytorch.item())