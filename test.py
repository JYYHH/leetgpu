from mha import transpose_KV_using_output, solve
import torch
from torch import nn
import time

def test_transpose_KV_using_output():
    N, d_model, h = 1 << 12, 7 * 17, 7
    K = torch.randn(N, d_model, device = "cuda")
    V = torch.randn(N, d_model, device = "cuda")
    K_transposed = K.reshape(N, h, -1).permute(1, 0, 2).reshape(-1)
    V_transposed = V.reshape(N, h, -1).permute(1, 0, 2).reshape(-1)
    output = torch.empty((N, d_model), device = "cuda")
    transpose_KV_using_output(K.data_ptr(), V.data_ptr(), output.data_ptr(), N, d_model // h, h)
    print(f"If K passed: {torch.allclose(K_transposed, K.reshape(-1))}")
    print(f"If V passed: {torch.allclose(V_transposed, V.reshape(-1))}")

def test_mha_correctness_and_benchmark():
    # define test cases
    N, d_model, h = 1 << 15, 256, 16
    Q = torch.randn(N, d_model, device = "cuda")
    K = torch.randn(N, d_model, device = "cuda")
    V = torch.randn(N, d_model, device = "cuda")
    # N, d_model, h = 2, 4, 2
    # Q = torch.tensor([[1.0, 0.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], device = "cuda")
    # K = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device = "cuda")
    # V = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]], device = "cuda")

    # N, d_model, h = 4, 4, 4
    # Q = torch.tensor([[0.2907358407974243, -0.9037965536117554, -0.7963020205497742, 0.7003895044326782], [-0.28541284799575806, -0.138644278049469, 0.6299468278884888, -0.15690559148788452], [-0.22019022703170776, 0.3157191276550293, 0.21075129508972168, -0.3111276626586914], [0.8423298597335815, -0.3191496729850769, -0.9271408319473267, 0.7470970153808594]], device = "cuda")
    # K = torch.tensor([[0.03441190719604492, 0.9820493459701538, -0.41239070892333984, 0.5119415521621704], [0.651220440864563, 0.95066237449646, 0.5093033313751221, -0.2072264552116394], [0.22628629207611084, 0.9570034742355347, 0.2386014461517334, -0.202925443649292], [0.12150979042053223, -0.7463395595550537, -0.8906869888305664, 0.6165797710418701]], device = "cuda")
    # V = torch.tensor([[-0.9380414485931396, 0.8038491010665894, 0.8862674236297607, 0.3074228763580322], [0.3303344249725342, 0.4225999116897583, -0.8587127327919006, 0.2452183961868286], [0.6728332042694092, 0.06418848037719727, -0.598467230796814, 0.588752269744873], [-0.6382660865783691, 0.12008094787597656, -0.22649502754211426, -0.5234025716781616]], device = "cuda")
    
    # run pytorch benchmark
    output_pytorch = torch.empty((N, d_model), device = "cuda")
    head_dim = d_model // h
    for i in range(h):
        output_pytorch[:, i * head_dim : (i + 1) * head_dim] = torch.softmax(Q[:, i * head_dim : (i + 1) * head_dim] @ K[:, i * head_dim : (i + 1) * head_dim].T / head_dim ** 0.5, dim = 1) @ V[:, i * head_dim : (i + 1) * head_dim]
    
    # run triton benchmark
    output = torch.empty((N, d_model), device = "cuda")
    start = time.time()
    solve(Q.data_ptr(), K.data_ptr(), V.data_ptr(), output.data_ptr(), N, d_model, h)
    torch.cuda.synchronize()
    end = time.time()
    print(f'The time taken is {end - start}')

    # compare results
    print(torch.allclose(output, output_pytorch, atol = 1e-4))
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output - output_pytorch))}')
    # print(f'The output is {output}')
    # print(f'The output_pytorch is {output_pytorch}')

if __name__ == "__main__":
    test_transpose_KV_using_output()
    test_mha_correctness_and_benchmark()