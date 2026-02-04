"""
    Author: Niklas Heftberger
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    main.py
    
    Image Inpainting Training
"""

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

from utils import create_predictions
from train import train


if __name__ == '__main__':
    
    # ============================================
    # KONFIGURATION
    # ============================================
    config = {
        # Reproducibility
        'seed': 42,
        
        # Data
        'testset_ratio': 0.1,
        'validset_ratio': 0.1,
        'data_path': str(PROJECT_ROOT / "data" / "dataset"),
        'results_path': str(PROJECT_ROOT / "results"),
        
        # Device
        'device': None,  # Auto-detect
        
        # Training - OPTIMIERTE Parameter
        'learningrate': 2e-4,       # Leicht erhöht da stabil
        'weight_decay': 1e-5,       # Leichte Regularisierung
        'n_updates': 75000,         # Mehr Training für bessere Ergebnisse
        'batchsize': 32,            # Sicher für alle GPUs
        
        # Early Stopping
        'early_stopping_patience': 7,
        
        # Logging
        'use_wandb': False,
        'use_perceptual_loss': False,
        'print_train_stats_at': 100,
        'print_stats_at': 200,
        'plot_at': 2000,
        'validate_at': 500,
        
        # Network
        'network_config': {
            'n_in_channels': 4,
            'base_channels': 64,
        }
    }
    
    os.makedirs(config['results_path'], exist_ok=True)

    # Info
    print("="*50)
    print("🎨 IMAGE INPAINTING TRAINING")
    print("="*50)
    print(f"Data: {config['data_path']}")
    print(f"LR: {config['learningrate']}, Batch: {config['batchsize']}")
    print(f"Updates: {config['n_updates']:,}")
    print("="*50)
    
    # Training
    train(**config)
    
    # Predictions
    testset_path = str(PROJECT_ROOT / "data" / "challenge_testset.npz")
    model_path = os.path.join(config['results_path'], "best_model.pt")
    save_path = os.path.join(config['results_path'], "testset", "my_submission.npz")
    plot_path = os.path.join(config['results_path'], "testset", "plots")

    if os.path.exists(testset_path) and os.path.exists(model_path):
        print("\n" + "="*50)
        print("📊 GENERATING PREDICTIONS")
        print("="*50)
        create_predictions(config['network_config'], model_path, testset_path, None, save_path, plot_path)
        print(f"✅ Saved to: {save_path}")
