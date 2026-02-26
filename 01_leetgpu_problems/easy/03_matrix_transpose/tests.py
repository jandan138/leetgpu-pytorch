import torch
import solution_pytorch
import solution_triton

def test_matrix_transpose():
    print("Running Matrix Transpose Test...")
    
    # Setup
    rows, cols = 7000, 6000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("Warning: Triton requires GPU, skipping Triton tests")
        return

    torch.manual_seed(0)
    
    # Create random matrix
    input_matrix = torch.randn(rows, cols, device=device)
    output_pytorch = torch.zeros(cols, rows, device=device)
    output_triton = torch.zeros(cols, rows, device=device)
    
    # Run PyTorch solution
    print(f"1. Running PyTorch implementation (rows={rows}, cols={cols})...")
    solution_pytorch.solve(input_matrix, output_pytorch, rows, cols)
    
    # Run Triton solution
    print(f"2. Running Triton implementation (rows={rows}, cols={cols})...")
    solution_triton.solve(input_matrix, output_triton, rows, cols)
    
    # Verify Correctness
    expected_output = input_matrix.t()
    
    # Check PyTorch
    if torch.allclose(output_pytorch, expected_output):
        print("✅ PyTorch Correctness Check Passed!")
    else:
        print("❌ PyTorch Correctness Check Failed!")
        diff = torch.abs(output_pytorch - expected_output).max()
        print(f"Max difference: {diff}")
        
    # Check Triton
    if torch.allclose(output_triton, expected_output):
        print("✅ Triton Correctness Check Passed!")
    else:
        print("❌ Triton Correctness Check Failed!")
        diff = torch.abs(output_triton - expected_output).max()
        print(f"Max difference: {diff}")

if __name__ == "__main__":
    test_matrix_transpose()
