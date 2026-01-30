"""
Test-Script um sicherzustellen dass alles funktioniert
Führe dieses Script aus BEVOR du auf Runpod trainierst!
"""

import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

print("=" * 60)
print("🔍 INPAINTING SETUP TEST")
print("=" * 60)

# Test 1: Imports
print("\n[1/6] Testing imports...")
try:
    import torch
    import numpy as np
    from architecture import MyModel, CombinedLoss
    from datasets import ImageDataset
    from utils import evaluate_model, plot
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: CUDA
print("\n[2/6] Testing CUDA...")
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ CUDA not available - will use CPU (slow!)")

# Test 3: Model creation
print("\n[3/6] Testing model creation...")
try:
    config = {'n_in_channels': 4, 'base_channels': 64}
    model = MyModel(**config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {total_params:,} parameters")
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_input = torch.randn(2, 4, 100, 100).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 3, 100, 100), f"Wrong output shape: {output.shape}"
    print(f"✅ Forward pass successful, output shape: {output.shape}")
except Exception as e:
    print(f"❌ Model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Loss function
print("\n[4/6] Testing loss function...")
try:
    loss_fn = CombinedLoss(mse_weight=1.0, l1_weight=1.0, perceptual_weight=0.1, use_perceptual=True)
    loss_fn.to(device)
    target = torch.randn(2, 3, 100, 100).to(device).clamp(0, 1)
    output = torch.randn(2, 3, 100, 100).to(device).clamp(0, 1)
    loss = loss_fn(output, target)
    print(f"✅ Loss function works: {loss.item():.4f}")
except Exception as e:
    print(f"❌ Loss function error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Dataset path
print("\n[5/6] Testing dataset path...")
data_path = PROJECT_ROOT / "data" / "dataset"
if data_path.exists():
    import glob
    images = glob.glob(str(data_path / "**" / "*.jpg"), recursive=True)
    print(f"✅ Dataset found: {len(images)} images")
    if len(images) == 0:
        print("⚠️ Warning: No images found in dataset folder!")
else:
    print(f"⚠️ Dataset folder not found: {data_path}")
    print("   Make sure to upload your dataset before training!")

# Test 6: Results path
print("\n[6/6] Testing results path...")
results_path = PROJECT_ROOT / "results"
try:
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Results folder ready: {results_path}")
except Exception as e:
    print(f"❌ Cannot create results folder: {e}")

# Summary
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA: {'✅ Available' if torch.cuda.is_available() else '❌ Not available'}")
print(f"Model params: {total_params:,}")

if torch.cuda.is_available():
    # Memory estimation
    # Rough estimate: params * 4 bytes * 3 (model + gradients + optimizer states)
    estimated_memory = total_params * 4 * 3 / 1e9
    batch_memory = 64 * 100 * 100 * (4 + 3) * 4 / 1e9  # batch * H * W * channels * bytes
    total_estimated = estimated_memory + batch_memory * 2
    print(f"Estimated GPU memory: ~{total_estimated:.1f} GB")
    
    available = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_estimated < available * 0.8:
        print(f"✅ Should fit in {available:.0f} GB VRAM")
    else:
        print(f"⚠️ Might be tight for {available:.0f} GB VRAM - reduce batch size if OOM")

print("\n✅ All tests passed! Ready for training.")
print("   Run: python main.py")
