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
    """Plotting the inputs, targets and predictions to file `path`"""

    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            img = data[i:i + 1:, 0:3, :, :]
            img = np.squeeze(img)
            img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update + 1:07d}_{i + 1:02d}.jpg"))

    plt.close(fig)


def testset_plot(input_array, output_array, path, index):
    """Plotting the inputs, targets and predictions to file `path` for testset (no targets available)"""

    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

    for ax, data, title in zip(axes, [input_array, output_array], ["Input", "Prediction"]):
        ax.clear()
        ax.set_title(title)
        img = data[0:3, :, :]
        img = np.squeeze(img)
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_axis_off()
    fig.savefig(os.path.join(path, f"testset_{index + 1:07d}.jpg"))

    plt.close(fig)


def evaluate_model(network: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Returns MSE and RMSE of the model on the provided dataloader"""
    network.eval()
    loss = 0.0
    n_batches = 0
    
    # Device-agnostic: nur auf CUDA autocast verwenden
    use_amp = device.type == 'cuda'
    
    with torch.no_grad():
        for data in dataloader:
            input_array, target = data
            input_array = input_array.to(device)
            target = target.to(device)

            # Validation immer in FP32 für stabile Metriken
            if use_amp:
                with torch.amp.autocast('cuda', enabled=False):
                    outputs = network(input_array).float()
                    batch_loss = loss_fn(outputs, target.float())
            else:
                outputs = network(input_array).float()
                batch_loss = loss_fn(outputs, target.float())
            
            # Skip NaN losses in validation
            if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                loss += batch_loss.item()
                n_batches += 1

        # Protect against division by zero
        if n_batches > 0:
            loss = loss / n_batches
        else:
            loss = float('nan')

        network.train()

        return loss, 255.0 * np.sqrt(loss) if not np.isnan(loss) else float('nan')


def read_compressed_file(file_path: str):
    with np.load(file_path) as data:
        input_arrays = data['input_arrays']
        known_arrays = data['known_arrays']
    return input_arrays, known_arrays


def create_predictions(model_config, state_dict_path, testset_path, device, save_path, plot_path, plot_at=20):
    """
    Here, one might needs to adjust the code based on the used preprocessing
    """

    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)

    model = MyModel(**model_config)
    model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    model.to(device)
    model.eval()

    input_arrays, known_arrays = read_compressed_file(testset_path)

    known_arrays = known_arrays.astype(np.float32)

    input_arrays = input_arrays.astype(np.float32) / 255.0

    input_arrays = np.concatenate((input_arrays, known_arrays), axis=1)

    predictions = list()
    
    print(f"Processing {len(input_arrays)} images...")

    with torch.no_grad():
        for i in range(len(input_arrays)):
            if (i + 1) % 50 == 0:
                print(f"Processing image {i + 1}/{len(input_arrays)}")
            
            # Add batch dimension if needed
            input_array = torch.from_numpy(input_arrays[i]).unsqueeze(0).to(device)
            output = model(input_array)
            output = output.squeeze(0).cpu().numpy()
            predictions.append(output)

            if (i + 1) % plot_at == 0:
                testset_plot(input_arrays[i], output, plot_path, i)

    predictions = np.stack(predictions, axis=0)

    predictions = (np.clip(predictions, 0, 1) * 255.0).astype(np.uint8)

    data = {
        "predictions": predictions
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **data)

    print(f"Predictions saved at {save_path}")
