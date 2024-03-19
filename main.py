"""
Created on: 2024, March 11th
Author: 'Cecover' on GitHub

Title: Stable Diffusion using Transformer for U-net replacement
Framework used: Pytorch
Code reference: https://github.com/facebookresearch/DiT/blob/main/train.py

Difference between this and referenced code:

1. Written from scratch
2. Optimized for single GPU training
"""

# ===== General Imports =====
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from diffusers import AutoencoderKL

# ===== Floating variables =====
dataset_path = ""
batch_size = 128
num_workers = 1

# ===== Stable Diffusion =====
"""
We will be using HuggingFace's diffuser library.

The model pipeline closely follows https://github.com/facebookresearch/DiT/blob/main/train.py
"""

vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")


# ===== Data loading =====
"""
For testing purposes, we will be using Imagenet's subset: Imagenette. 

This means that we will be using Imagenet's setting for the data loading, mainly for the standard deviation and the 
mean. 

This code heavily follows:

1. https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
2. https://github.com/facebookresearch/DiT/blob/main/train.py

Center cropping implementation taken from Guided Diffusion.
https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
"""

# Mean and standard deviation of the dataset
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


# Defining image transformations
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def patchify(image, patch_size, flatten_channels=True):
    B, H, W, C = image.shape

    image = image.reshape(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    image = image.transpose(0, 1, 3, 2, 4, 5)  # [B, H', W', p_H, p_W, C]
    patches = image.reshape(B, -1, *image.shape[3:])  # [B, H'*W', p_H, p_W, C]

    if flatten_channels:
        patches = patches.reshape(B, patches.shape[1], -1)

    return patches


transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


# Since stable diffusion is self-supervised, we do not need to worry about having multiple dataloaders.
# We can just use one and be done with it lol.
dataset = ImageFolder(dataset_path, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True
)
