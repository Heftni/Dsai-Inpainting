"""
    Author: Niklas Heftberger
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    utils.py
"""

import torch
import numpy as np
import os
from matplotlib import pyplot as plt

from architecture import MyModel


def plot(inputs, targets, predictions, path, update):
    """Plot input, target and prediction"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for i in range(min(len(inputs), 4)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            img = np.transpose(data[i, :3], (1, 2, 0))
            ax.imshow(np.clip(img, 0, 1))
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update+1:07d}_{i+1:02d}.jpg"))
    plt.close(fig)


def testset_plot(input_array, output_array, path, index):
    """Plot testset predictions"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

    for ax, data, title in zip(axes, [input_array, output_array], ["Input", "Prediction"]):
        ax.clear()
        ax.set_title(title)
        img = np.transpose(data[:3], (1, 2, 0))
        ax.imshow(np.clip(img, 0, 1))
        ax.set_axis_off()
    fig.savefig(os.path.join(path, f"testset_{index+1:07d}.jpg"))
    plt.close(fig)


def evaluate_model(network, dataloader, loss_fn, device):
    """Evaluate model - returns MSE loss and RMSE"""
    network.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = network(inputs)
            loss = loss_fn(outputs, targets)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')
    rmse = 255.0 * np.sqrt(avg_loss) if not np.isnan(avg_loss) else float('nan')
    
    return avg_loss, rmse


def read_compressed_file(file_path: str):
    with np.load(file_path) as data:
        return data['input_arrays'], data['known_arrays']


def create_predictions(model_config, state_dict_path, testset_path, device, save_path, plot_path, plot_at=20):
    """Generate predictions for testset"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    model = MyModel(**model_config)
    model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    model.to(device)
    model.eval()

    input_arrays, known_arrays = read_compressed_file(testset_path)
    input_arrays = input_arrays.astype(np.float32) / 255.0
    known_arrays = known_arrays.astype(np.float32)
    input_arrays = np.concatenate((input_arrays, known_arrays), axis=1)

    predictions = []
    print(f"Processing {len(input_arrays)} images...")

    with torch.no_grad():
        for i in range(len(input_arrays)):
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(input_arrays)}")
            
            x = torch.from_numpy(input_arrays[i]).unsqueeze(0).to(device)
            out = model(x).squeeze(0).cpu().numpy()
            predictions.append(out)

            if (i + 1) % plot_at == 0:
                testset_plot(input_arrays[i], out, plot_path, i)

    predictions = np.stack(predictions, axis=0)
    predictions = (np.clip(predictions, 0, 1) * 255.0).astype(np.uint8)

    data = {
        "predictions": predictions
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **data)

    print(f"Predictions saved at {save_path}")
