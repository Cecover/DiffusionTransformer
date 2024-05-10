"""
Created on: 2024, March 11th
Author: 'Cecover' on GitHub

Title: Scalable Diffusion Models with Transformers
Framework used: Pytorch
Code reference: - https://github.com/facebookresearch/DiT/blob/main/train.py (original)
                - https://github.com/chuanyangjin/fast-DiT/blob/main/train.py (re-written)

Difference between this and referenced code:

1. Written from scratch
2. Optimized for single GPU training
"""

# ===== General Imports =====
import os
import logging
import numpy as np
from glob import glob
from PIL import Image
from copy import deepcopy
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL
from accelerate import Accelerator
from model import DiTNet

# ===== Floating variables and functions =====
dataset_path = ""
batch_size = 128
num_workers = 1

# Using pre-used flags to increase training speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def requires_grad(model, flag=True):
    """
    Set 'requires_grad' flag to all parameters in a model.
    """

    for p in model.parameters():
        p.requires_grad = flag


def update_EMA(ema_model, model, decay=0.9999):
    """
    Update the exponential moving average model (EMA) towards the current model
    """

    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, parameters in model_params.items():
        name = name.replace("module.", "")
        if ema_params[name].requires_grad:  # We only apply to params that require grad
            ema_params[name].mul_(decay).add_(parameters.data, alpha=1 - decay)


def create_logger(logging_dir: str):
    """
    Creates a logger that writes into a log file and a standard output.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


class CustomDataset(Dataset):
    """
    Creates a custom dataset, which can be loaded using PyTorch dataloaders.
    """

    def __init__(self, features_dir: str, labels_dir: str):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(
            self.labels_files
        ), "Number of features and labels must be the same!"

        return len(self.features_files)  # Simply returns the length of the dataset

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = torch.load(os.path.join(self.features_dir, feature_file))
        labels = torch.load(os.path.join(self.labels_dir, label_file))

        return features, labels


# ===== VAE =====
"""
We will be using HuggingFace's diffuser library.

This is simply us loading the encoder and decoder of the LDM model, considering it uses a pretrained VAE model, which
acts as an encoder and the decoder for the images and latent image z_1.

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
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


transform = transforms.Compose(
    [
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
)

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
    persistent_workers=True,
)


def main(args):
    """
    The main trainer. Only works with a GPU!
    """

    assert torch.cuda.is_available(), "Training REQUIRES at least 1 (one) Nvidia GPU!"

    # Setting up the accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Setting up the experiment folders
    # TODO: Connect to either tensorboard or Wandb
    if accelerator.is_main_process:
        os.makedirs(args.result_dir, exist_ok=True)  # To make the results directory
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = (
            f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        )
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Setting up the model and dataset
    assert (
        args.image_size % 8 == 0
    ), "Image size must be divisible by 8 for the VAE encoder!"

    latent_size = args.image_size // 8

    # Loading the transformer model (parameterization is already done inside the model)
    model = DiTNet[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    ema_model = deepcopy(model).to(device)
    requires_grad(ema_model, False)





