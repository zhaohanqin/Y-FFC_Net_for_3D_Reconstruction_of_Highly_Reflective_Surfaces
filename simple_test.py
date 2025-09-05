import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

try:
    from ffc import FFC_BN_ACT
    print("FFC module imported successfully")
    
    from Y_FFC import YFFC
    print("YFFC network imported successfully")
    
    # 创建模型
    model = YFFC()
    print("Model created successfully")
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    
    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    test_input = torch.randn(1, 1, 512, 512).to(device)
    print("Test input shape:", test_input.shape)
    
    with torch.no_grad():
        output = model(test_input)
    
    print("Forward pass successful")
    print("Output keys:", list(output.keys()))
    print("Diffuse reflection shape:", output['r'].shape)
    print("Gradient shape:", output['g'].shape)
    print("Specular reflection shape:", output['b'].shape)
    
    print("All tests passed!")
    
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc() 