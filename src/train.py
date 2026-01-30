"""
    Author: Niklas Heftberger
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    train.py
    
    Stabiles Training - OHNE AMP für maximale Stabilität
"""

import datasets
from architecture import MyModel, CombinedLoss
from utils import plot, evaluate_model

import torch
import numpy as np
import os
import platform
import time

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(seed, testset_ratio, validset_ratio, data_path, results_path, early_stopping_patience, device, learningrate,
          weight_decay, n_updates, use_wandb, print_train_stats_at, print_stats_at, plot_at, validate_at, batchsize,
          network_config: dict, use_perceptual_loss: bool = False):
    
    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # CUDA Settings
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    # Workers
    is_linux = platform.system() == 'Linux'
    num_workers = 8 if is_linux else 0

    # Paths
    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)
    saved_model_path = os.path.join(results_path, "best_model.pt")

    # Dataset
    image_dataset = datasets.ImageDataset(datafolder=data_path)
    n_total = len(image_dataset)
    n_test = int(n_total * testset_ratio)
    n_valid = int(n_total * validset_ratio)
    n_train = n_total - n_test - n_valid
    
    indices = np.random.permutation(n_total)
    dataset_train = Subset(image_dataset, indices[:n_train])
    dataset_valid = Subset(image_dataset, indices[n_train:n_train + n_valid])
    dataset_test = Subset(image_dataset, indices[n_train + n_valid:])
    
    print(f"Dataset: Train={n_train}, Valid={n_valid}, Test={n_test}")

    # DataLoaders
    dataloader_train = DataLoader(
        dataset_train, batch_size=batchsize, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0
    )
    dataloader_valid = DataLoader(
        dataset_valid, batch_size=batchsize, shuffle=False,
        num_workers=num_workers // 2 if num_workers > 0 else 0, pin_memory=True
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=batchsize, shuffle=False,
        num_workers=num_workers // 2 if num_workers > 0 else 0, pin_memory=True
    )

    # Model
    network = MyModel(**network_config).to(device)
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Model: {total_params:,} parameters")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Loss & Optimizer
    loss_fn = CombinedLoss(mse_weight=1.0, l1_weight=1.0).to(device)
    mse_loss = torch.nn.MSELoss()
    
    optimizer = torch.optim.AdamW(
        network.parameters(), 
        lr=learningrate, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine Annealing - einfach und stabil
    scheduler = CosineAnnealingLR(optimizer, T_max=n_updates, eta_min=learningrate * 0.01)

    print(f"\n{'='*50}")
    print(f"Training auf {device}")
    print(f"LR: {learningrate}, Batch: {batchsize}, Updates: {n_updates:,}")
    print(f"{'='*50}\n")

    # Training Loop
    network.train()
    i = 0
    counter = 0
    best_val_loss = float('inf')
    running_loss = 0.0
    start_time = time.time()

    while i < n_updates:
        for inputs, targets in dataloader_train:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = loss_fn(outputs, targets)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"⚠️ NaN at step {i+1}, skipping...")
                i += 1
                continue
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            
            # Logging
            if (i + 1) % print_train_stats_at == 0:
                avg_loss = running_loss / print_train_stats_at
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                eta = (n_updates - i - 1) / speed / 60
                print(f'Step {i+1:>6}/{n_updates} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | {speed:.1f} it/s | ETA: {eta:.0f}min')
                running_loss = 0.0

            # Plot
            if (i + 1) % plot_at == 0:
                plot(inputs.cpu().numpy(), targets.cpu().numpy(), outputs.detach().cpu().numpy(), plotpath, i)

            # Validation
            if (i + 1) % validate_at == 0:
                val_loss, val_rmse = evaluate_model(network, dataloader_valid, mse_loss, device)
                print(f"  📋 Val Loss: {val_loss:.6f} | RMSE: {val_rmse:.2f}", end="")
                
                if val_loss < best_val_loss:
                    improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 0
                    best_val_loss = val_loss
                    torch.save(network.state_dict(), saved_model_path)
                    print(f" ✅ Best! ({improvement:.1f}%)")
                    counter = 0
                else:
                    counter += 1
                    print(f" ⏳ {counter}/{early_stopping_patience}")
                
                network.train()

            # Early Stopping
            if counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping at step {i+1}")
                i = n_updates
                break

            i += 1
            if i >= n_updates:
                break

    # Final Evaluation
    print(f"\n{'='*50}")
    print("🏁 FINAL EVALUATION")
    print(f"{'='*50}")
    
    network.load_state_dict(torch.load(saved_model_path, weights_only=True))
    network.eval()
    
    test_loss, test_rmse = evaluate_model(network, dataloader_test, mse_loss, device)
    total_time = (time.time() - start_time) / 60
    
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Training Time: {total_time:.1f} min")
    print(f"{'='*50}")
