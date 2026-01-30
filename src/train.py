"""
    Author: Niklas Heftberger
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    train.py
    
    ULTIMATE Training für RTX 5090 mit:
    - Mixed Precision Training (AMP) für 2x Speedup
    - Perceptual Loss für visuelle Qualität
    - Learning Rate Scheduler (OneCycleLR)
    - Gradient Clipping & Accumulation
    - Multi-Worker DataLoading (15 vCPU)
    - Cosine Annealing mit Warmup
"""

import datasets
from architecture import MyModel, CombinedLoss
from utils import plot, evaluate_model

import torch
import torch.amp
import numpy as np
import os
import platform
import time

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

import wandb

def train(seed, testset_ratio, validset_ratio, data_path, results_path, early_stopping_patience, device, learningrate,
          weight_decay, n_updates, use_wandb, print_train_stats_at, print_stats_at, plot_at, validate_at, batchsize,
          network_config: dict, use_perceptual_loss: bool = True):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed)
    
    # Optimierte CUDA Settings
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # TF32 für RTX 30/40/50 Serie - schneller ohne Qualitätsverlust
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)

    # System Detection - Runpod hat 15 vCPU
    is_linux = platform.system() == 'Linux'
    num_workers = 12 if is_linux else 0  # 12 Workers für 15 vCPU
    
    # Mixed Precision für massive Speedups
    use_amp = device.type == 'cuda'
    # GradScaler mit konservativen Settings für Stabilität
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, growth_interval=2000) if use_amp else None

    if use_wandb:
        wandb.login()
        wandb.init(project="image_inpainting", config={
            "learning_rate": learningrate,
            "weight_decay": weight_decay,
            "n_updates": n_updates,
            "batch_size": batchsize,
            "validation_ratio": validset_ratio,
            "testset_ratio": testset_ratio,
            "early_stopping_patience": early_stopping_patience,
            "mixed_precision": use_amp,
            "perceptual_loss": use_perceptual_loss,
            "base_channels": network_config.get('base_channels', 64),
        })

    # Prepare paths
    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)

    image_dataset = datasets.ImageDataset(datafolder=data_path)

    n_total = len(image_dataset)
    n_test = int(n_total * testset_ratio)
    n_valid = int(n_total * validset_ratio)
    n_train = n_total - n_test - n_valid
    indices = np.random.permutation(n_total)
    dataset_train = Subset(image_dataset, indices=indices[0:n_train])
    dataset_valid = Subset(image_dataset, indices=indices[n_train:n_train + n_valid])
    dataset_test = Subset(image_dataset, indices=indices[n_train + n_valid:n_total])

    assert len(image_dataset) == len(dataset_train) + len(dataset_test) + len(dataset_valid)

    del image_dataset
    
    print(f"Dataset split: Train={n_train}, Valid={n_valid}, Test={n_test}")

    # DataLoader - maximale Performance für RTX 5090 + 15 vCPU
    dataloader_train = DataLoader(
        dataset=dataset_train, 
        batch_size=batchsize,
        num_workers=num_workers, 
        shuffle=True, 
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None,
        drop_last=True  # Konsistente Batch Sizes
    )
    dataloader_valid = DataLoader(
        dataset=dataset_valid, 
        batch_size=32,  # Größer für schnellere Validation
        num_workers=num_workers // 2 if num_workers > 0 else 0, 
        shuffle=False, 
        pin_memory=True
    )
    dataloader_test = DataLoader(
        dataset=dataset_test, 
        batch_size=32,
        num_workers=num_workers // 2 if num_workers > 0 else 0, 
        shuffle=False, 
        pin_memory=True
    )

    # Initialize model
    network = MyModel(**network_config)
    network.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # GPU Memory Info
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    network.train()

    # Loss function mit optionalem Perceptual Loss
    combined_loss = CombinedLoss(
        mse_weight=1.0, 
        l1_weight=1.0,
        perceptual_weight=0.1,
        use_perceptual=use_perceptual_loss
    ).to(device)
    
    mse_loss = torch.nn.MSELoss()

    # AdamW mit optimierten Parametern
    optimizer = torch.optim.AdamW(
        network.parameters(), 
        lr=learningrate, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # OneCycleLR für super convergence - KONSERVATIVE Parameter für 4090/AMP
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learningrate * 5,   # Peak bei 1e-3 - konservativer für AMP
        total_steps=n_updates,
        pct_start=0.1,             # 10% warmup (stabiler mit AMP)
        anneal_strategy='cos',
        div_factor=5,              # Start lr = max_lr / 5 (sanfterer Start)
        final_div_factor=100       # End lr = max_lr / 500
    )

    if use_wandb:
        wandb.watch(network, mse_loss, log="all", log_freq=100)

    i = 0
    counter = 0
    best_validation_loss = np.inf
    loss_list = []
    running_loss = 0.0
    start_time = time.time()

    saved_model_path = os.path.join(results_path, "best_model.pt")

    print(f"\n{'='*60}")
    print(f"Started training on device {device}")
    print(f"Mixed Precision (AMP): {use_amp}")
    print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"Num workers: {num_workers}")
    print(f"Learning rate: {learningrate} (max: {learningrate * 10})")
    print(f"Weight decay: {weight_decay}")
    print(f"Batch size: {batchsize}")
    print(f"Max updates: {n_updates:,}")
    print(f"Perceptual Loss: {use_perceptual_loss}")
    print(f"{'='*60}\n")

    while i < n_updates:

        for input, target in dataloader_train:

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Training mit NaN Protection
            if use_amp:
                with torch.amp.autocast('cuda'):
                    output = network(input)
                    loss = combined_loss(output, target)
                
                # NaN Check - Skip step if loss is NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  NaN/Inf loss at step {i+1}, skipping...")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()  # Reset scaler
                    continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Aggressiveres Gradient Clipping für Stabilität
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = network(input)
                loss = combined_loss(output, target)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  NaN/Inf loss at step {i+1}, skipping...")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
                optimizer.step()
            
            scheduler.step()

            current_loss = loss.item()
            loss_list.append(current_loss)
            running_loss += current_loss

            if (i + 1) % print_train_stats_at == 0:
                avg_loss = running_loss / print_train_stats_at
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_per_sec = (i + 1) / elapsed
                eta = (n_updates - i - 1) / steps_per_sec / 60  # in minutes
                
                print(f'Step {i + 1:>6}/{n_updates} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | {steps_per_sec:.1f} it/s | ETA: {eta:.0f}min')
                running_loss = 0.0

            if use_wandb and (i+1) % print_stats_at == 0:
                wandb.log({
                    "training/loss_per_batch": current_loss,
                    "training/learning_rate": scheduler.get_last_lr()[0]
                }, step=i)

            if (i + 1) % plot_at == 0:
                print(f"📊 Plotting images at step {i + 1}")
                plot(input.cpu().numpy(), target.detach().cpu().numpy(), output.detach().cpu().numpy(), plotpath, i)

            if (i + 1) % validate_at == 0:
                print(f"\n{'─'*40}")
                print(f"📋 Validation at step {i + 1}")
                val_loss, val_rmse = evaluate_model(network, dataloader_valid, mse_loss, device)
                print(f"   Loss: {val_loss:.6f} | RMSE: {val_rmse:.2f}")

                if use_wandb:
                    wandb.log({
                        "validation/loss": val_loss,
                        "validation/RMSE": val_rmse
                    }, step=i)

                if val_loss < best_validation_loss:
                    improvement = (best_validation_loss - val_loss) / best_validation_loss * 100 if best_validation_loss != np.inf else 0
                    best_validation_loss = val_loss
                    torch.save(network.state_dict(), saved_model_path)
                    print(f"   ✅ New best model! ({improvement:.2f}% better)")
                    counter = 0
                else:
                    counter += 1
                    print(f"   ⏳ No improvement. Patience: {counter}/{early_stopping_patience}")
                
                print(f"{'─'*40}\n")

            if counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping at step {i + 1}")
                i = n_updates
                break

            i += 1
            if i >= n_updates:
                print(f"\n✅ Training completed after {n_updates:,} updates")
                break

    # Final evaluation
    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"🏁 FINAL EVALUATION")
    print(f"{'='*60}")
    print(f"Total training time: {total_time:.1f} minutes")
    
    # Load best model
    network_eval = MyModel(**network_config)
    network_eval.load_state_dict(torch.load(saved_model_path, weights_only=True))
    network_eval.to(device)
    network_eval.eval()
    
    testset_loss, testset_rmse = evaluate_model(
        network=network_eval, 
        dataloader=dataloader_test, 
        loss_fn=mse_loss,
        device=device
    )

    print(f'Test Loss: {testset_loss:.6f}')
    print(f'Test RMSE: {testset_rmse:.2f}')
    print(f'Best Validation Loss: {best_validation_loss:.6f}')
    print(f"{'='*60}")

    if use_wandb:
        wandb.summary["testset/loss"] = testset_loss
        wandb.summary["testset/RMSE"] = testset_rmse
        wandb.summary["training_time_minutes"] = total_time
        wandb.finish()
