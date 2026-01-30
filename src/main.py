"""
    Author: Niklas Heftberger
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    main.py
    
    ULTIMATE Konfiguration für RTX 5090 auf Runpod
    32GB VRAM | 117GB RAM | 15 vCPU
"""

import os
import sys
from pathlib import Path

# Automatische Pfaderkennung - funktioniert überall
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Imports nach Pfad-Setup
from utils import create_predictions
from train import train


if __name__ == '__main__':
    config_dict = dict()

    # Reproducibility
    config_dict['seed'] = 42
    
    # Data splits
    config_dict['testset_ratio'] = 0.1
    config_dict['validset_ratio'] = 0.1
    
    # ============================================
    # PFADE - Automatisch relativ zum Projekt
    # ============================================
    config_dict['results_path'] = str(PROJECT_ROOT / "results")
    config_dict['data_path'] = str(PROJECT_ROOT / "data" / "dataset")
    
    # Erstelle results Ordner falls nicht vorhanden
    os.makedirs(config_dict['results_path'], exist_ok=True)
    
    # Device
    config_dict['device'] = None  # Auto-detect (CUDA > MPS > CPU)
    
    # ============================================
    # ULTIMATE HYPERPARAMETER für RTX 5090 (32GB VRAM)
    # Getestet und optimiert für beste Ergebnisse!
    # ============================================
    
    # Learning rate - optimal für große Batch Size mit OneCycleLR
    config_dict['learningrate'] = 1e-3  # Konservativerer Start, OneCycleLR geht bis 1e-2
    
    # Weight decay - moderate Regularisierung
    config_dict['weight_decay'] = 5e-5  # Leicht reduziert für bessere Generalisierung
    
    # Training duration - ausreichend für Konvergenz
    config_dict['n_updates'] = 100000  # 100k reicht für gute Ergebnisse
    
    # Batch size - RTX 5090 mit 32GB VRAM
    # base_channels=64 mit batch=128 ist sicher, 96 braucht kleinere batch
    config_dict['batchsize'] = 64  # Sicher für base_channels=64
    
    # Early stopping - genug Geduld aber nicht zu viel
    config_dict['early_stopping_patience'] = 20
    
    # Wandb logging
    config_dict['use_wandb'] = False 
    
    # Perceptual Loss - verbessert visuelle Qualität
    config_dict['use_perceptual_loss'] = True

    # Logging intervals
    config_dict['print_train_stats_at'] = 100
    config_dict['print_stats_at'] = 200
    config_dict['plot_at'] = 2000
    config_dict['validate_at'] = 500

    # ============================================
    # NETZWERK - Optimiert für RTX 5090
    # base_channels=64 ist der sweet spot für Qualität/Speed
    # ============================================
    network_config = {
        'n_in_channels': 4,      # 3 RGB + 1 Mask
        'base_channels': 64,     # ~40M Parameter - sehr gute Qualität
    }
    
    config_dict['network_config'] = network_config

    # Training starten
    print("="*60)
    print("🚀 IMAGE INPAINTING - RTX 5090 ULTIMATE EDITION 🚀")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data path: {config_dict['data_path']}")
    print(f"Results path: {config_dict['results_path']}")
    print("-"*60)
    print(f"Network config: {network_config}")
    print(f"Updates: {config_dict['n_updates']:,}")
    print(f"Batch size: {config_dict['batchsize']}")
    print(f"Learning rate: {config_dict['learningrate']}")
    print(f"Perceptual Loss: {config_dict['use_perceptual_loss']}")
    print("="*60)
    
    train(**config_dict)
    
    # ============================================
    # PREDICTIONS AUF TESTSET
    # ============================================
    testset_path = str(PROJECT_ROOT / "data" / "challenge_testset.npz")
    state_dict_path = os.path.join(config_dict['results_path'], "best_model.pt")
    save_path = os.path.join(config_dict['results_path'], "testset", "my_submission.npz")
    plot_path = os.path.join(config_dict['results_path'], "testset", "plots")

    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    create_predictions(
        config_dict['network_config'], 
        state_dict_path, 
        testset_path, 
        None, 
        save_path, 
        plot_path, 
        plot_at=20
    )
    
    print(f"\n✅ Predictions saved to: {save_path}")
