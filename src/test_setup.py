"""
Test-Script um sicherzustellen dass alles funktioniert
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

print("="*50)
print("🔍 SETUP TEST")
print("="*50)

# Test 1: Imports
print("\n[1/5] Imports...")
try:
    import torch
    import numpy as np
    from architecture import MyModel, CombinedLoss
    from datasets import ImageDataset
    print("✅ OK")
except ImportError as e:
    print(f"❌ {e}")
    sys.exit(1)

# Test 2: CUDA
print("\n[2/5] CUDA...")
if torch.cuda.is_available():
    print(f"✅ {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No CUDA - will use CPU")

# Test 3: Model
print("\n[3/5] Model...")
try:
    model = MyModel(n_in_channels=4, base_channels=64)
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ {params:,} parameters")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.randn(2, 4, 100, 100).to(device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 3, 100, 100)
    print(f"✅ Forward pass OK")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# Test 4: Loss
print("\n[4/5] Loss...")
try:
    loss_fn = CombinedLoss().to(device)
    out = torch.rand(2, 3, 100, 100).to(device)
    tgt = torch.rand(2, 3, 100, 100).to(device)
    loss = loss_fn(out, tgt)
    print(f"✅ Loss: {loss.item():.4f}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# Test 5: Dataset
print("\n[5/5] Dataset...")
data_path = PROJECT_ROOT / "data" / "dataset"
if data_path.exists():
    import glob
    images = glob.glob(str(data_path / "**" / "*.jpg"), recursive=True)
    print(f"✅ {len(images)} images")
else:
    print(f"⚠️ Not found: {data_path}")

print("\n" + "="*50)
print("✅ Ready! Run: python main.py")
print("="*50)

print("\n✅ All tests passed! Ready for training.")
print("   Run: python main.py")
