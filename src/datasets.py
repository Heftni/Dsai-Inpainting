"""
    Author: Niklas Heftberger
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    datasets.py
    
    Optimiertes Dataset mit Data Augmentation für bessere Generalisierung
"""

import os
import random
import glob
from torchvision import transforms
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


IMAGE_DIMENSION = 100


def create_arrays_from_image(image_array: np.ndarray, offset: tuple, spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    image_array = np.transpose(image_array, (2, 0, 1))  # makes (3, 100, 100) from (100, 100, 3)
    
    known_array = np.zeros_like(image_array) # (3, 100, 100)
    known_array[:, offset[1]::spacing[1], offset[0]::spacing[0]] = 1

    # Korrigierte Zeile: Nur die Werte im Array auf 0 setzen, nicht die Variable überschreiben
    image_array[known_array == 0] = 0

    known_array = known_array[0:1] # creates  (1, 100, 100) from (3, 100, 100)

    return image_array, known_array # (3, 100, 100), (1, 100, 100)


def resize_image(img: Image.Image) -> Image.Image:
    """Resize and crop image to target dimension"""
    resize_transforms = transforms.Compose([
        transforms.Resize((IMAGE_DIMENSION, IMAGE_DIMENSION)),
        transforms.CenterCrop((IMAGE_DIMENSION, IMAGE_DIMENSION))
    ])
    return resize_transforms(img)


def augment_image(img: Image.Image) -> Image.Image:
    """
    Apply random augmentations to improve generalization:
    - Random horizontal/vertical flip
    - Random rotation
    - Random brightness/contrast adjustments
    - Random color jitter
    """
    # Random horizontal flip (50% chance)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Random vertical flip (30% chance)
    if random.random() > 0.7:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Random rotation (90, 180, 270 degrees) - 30% chance
    if random.random() > 0.7:
        angle = random.choice([90, 180, 270])
        img = img.rotate(angle)
    
    # Random brightness adjustment (20% chance)
    if random.random() > 0.8:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    
    # Random contrast adjustment (20% chance)
    if random.random() > 0.8:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    
    # Random saturation adjustment (15% chance)
    if random.random() > 0.85:
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    
    return img


def preprocess(input_array: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range"""
    return np.array(input_array, dtype=np.float32) / 255.0


class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images from a folder with optional augmentation
    """

    def __init__(self, datafolder: str, augment: bool = True):
        self.imagefiles = sorted(glob.glob(os.path.join(datafolder, "**", "*.jpg"), recursive=True))
        self.augment = augment
        print(f"Loaded {len(self.imagefiles)} images from {datafolder}")

    def __len__(self):
        return len(self.imagefiles)

    def __getitem__(self, idx: int):
        index = int(idx)

        # Load image
        image = Image.open(self.imagefiles[index]).convert('RGB')
        
        # Resize
        image = resize_image(image)
        
        # Apply augmentation during training
        if self.augment:
            image = augment_image(image)
        
        # Convert to numpy and preprocess
        image = np.asarray(image)
        image = preprocess(image)

        # Random spacing and offset for mask generation
        # Varied range for better generalization
        spacing_x = random.randint(2, 6)
        spacing_y = random.randint(2, 6)
        offset_x = random.randint(0, spacing_x - 1)
        offset_y = random.randint(0, spacing_y - 1)

        spacing = (spacing_x, spacing_y)
        offset = (offset_x, offset_y)

        # Create input and known arrays
        input_array, known_array = create_arrays_from_image(image.copy(), offset, spacing)

        # Convert to target tensor
        target_image = torch.from_numpy(np.transpose(image, (2, 0, 1)))  # (3, 100, 100)
        
        # Convert input and known to tensors
        input_array = torch.from_numpy(input_array)
        known_array = torch.from_numpy(known_array)

        # Concatenate input image with known mask
        input_array = torch.cat((input_array, known_array), dim=0)  # (4, 100, 100)

        return input_array, target_image
