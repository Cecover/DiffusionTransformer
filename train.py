"""
Created on: 2024, March 11th
Author: 'Cecover' on GitHub

Title: Scalable Diffusion Models with Transformers
Framework used: Pytorch
Code reference: - https://github.com/facebookresearch/DiT/blob/main/train.py (original)
                - https://github.com/chuanyangjin/fast-DiT/blob/main/train.py (re-written)
                - https://github.com/chuanyangjin/fast-DiT/blob/main/train_options/train_original.py (re-written)
                - https://github.com/crowsonkb/k-diffusion

Difference between this and referenced code:
1. Written from scratch
2. General code cleanup
3. Added documentation
4. Complete move to HuggingFace Accelerate

Current TODO's:
1. Clean up and restructure code
2. Add better model logging system
3. Make it more "verbose"

Some comments:
1. This code would only use ImageNet dataset and its variants
"""

# ===== Imports =====

# General imports (loggers, argparse, tqdm)
import os
import logging  # Creates a log file just in case the code went into flames
import argparse
import numpy as np
from glob import glob
from PIL import Image
from time import time
from tqdm import tqdm  # Main logging for training performance
from copy import deepcopy
from collections import OrderedDict

# Model related imports (PyTorch, Accelerate, Diffusers)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL
from accelerate import Accelerator

# Created module imports
from model import DiTNet
from diffusion import create_diffusion

# ===== Floating variables and helper functions =====

# Multiprocessing
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    torch._dynamo.config.automatic_dynamic_shapes = False
except AttributeError:
    pass

# Mean and standard deviation of the dataset (Imagenet)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())


def requires_grad(model, flag=True):
    """
    Set 'requires_grad' flag to all parameters in a model.
    This allows all the model parameter gradients to be computed.
    """
    for p in model.parameters():
        p.requires_grad = flag


def update_EMA(ema_model, model, decay=0.9999):
    """
    To update the Exponential Moving Average model (EMA model) towards the current model.
    More about this here: https://timm.fast.ai/training_modelEMA
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, parameters in model_params.items():
        name = name.replace("module.", "")
        if ema_params[name].requires_grad:  # We only apply to params that require grad
            ema_params[name].mul_(decay).add_(parameters.data, alpha=1 - decay)


def create_logger(logging_dir: str):
    """
    Creates a logger that writes into a log file and standard output.
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


# Dataset transformations
class center_crop_array(torch.nn.Module):
    # We need to do this because somehow turning it into a function wouldn't work when using multiprocessing.

    def __init__(self, image_size):
        super().__init__()

        self.image_size = image_size

    def forward(self, pil_image):
        """
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126

        I modified this function, so it would work with multiprocessing dataloader.
        Now we don't need to fix the value manually/editing the source code.
        """
        while min(*pil_image.size) >= 2 * self.image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - self.image_size) // 2
        crop_x = (arr.shape[1] - self.image_size) // 2

        return Image.fromarray(
            arr[crop_y: crop_y + self.image_size, crop_x: crop_x + self.image_size]
        )


# ===== Main Training Loop =====
def main(args):
    """
    The main trainer. Only works with a GPU/MPS!
    The trainer now uses HF's Accelerate instead of DDP, which means I wrote the same code twice.

    But again, it can run on my MacBook so whatever.

    !!UNTESTED FOR NVIDIA GPU!!
    """

    # Setting up the accelerator
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=args.precision)
    ensure_distributed()
    device = accelerator.device
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(args.experiments_path, exist_ok=True)  # To make the results directory
        experiment_index = len(glob(f"{args.experiments_path}/*"))
        model_string_name = args.experiment_name
        experiment_dir = f"{args.experiments_path}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)

        # Wall of loggers
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Checkpoint directory created at {checkpoint_dir}")
        logger.info(f"Process {accelerator.process_index} using device: {device}")
        logger.info(
            f"World size: {accelerator.num_processes} and Batch size: {args.batch_size * accelerator.num_processes}")

    # Setting up the model and dataset
    # Latent image sizing
    assert (args.image_size % 8 == 0), "Image size must be divisible by 8 for the VAE encoder!"
    latent_size = args.image_size // 8

    # Loading the transformer and EMA model (parameterization is already done inside the model)
    # model = DiTNet(input_size=latent_size, num_classes=args.num_classes).to(device)
    model = DiTNet(
        input_size=latent_size,
        patch_size=8,
        in_channels=4,
        num_classes=args.num_classes,
        global_hidden_dimension=1152,
        transformer_depth=28,
        transformer_attn_heads=8,
        transformer_mlp_ratio=4.0,
        dropout_prob=0.1,
        learn_sigma=True,
    ).to(device)

    ema_model = deepcopy(model).to(device)
    requires_grad(ema_model, False)

    # Loading the diffusion model along with the latent encoder
    diffusion = create_diffusion(timestep_respacing="")  # Default would be 1000 steps, with linear noise scheduling
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setting up the optimizer
    # On paper, they use Adam with betas=(0.9, 0.999), and a constant lr of 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

    # Dataset loading and transformations
    center_crop = center_crop_array(args.image_size)
    transform = transforms.Compose(
        [
            center_crop,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std, inplace=True),
        ]
    )

    # We're using Imagenet, thus we don't need to make a Custom DataLoader or such.
    features_dir = f"{args.dataset_path}"
    dataset = ImageFolder(features_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size // accelerator.num_processes),
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    if accelerator.is_main_process:
        try:
            logger.info(f"Used dataset: {args.dataset_path} with length of {len(dataset):,}")
        except TypeError:
            pass

    # Preparing the model for training
    update_EMA(ema_model, model, decay=0)
    model.train()  # IMPORTANT
    ema_model.eval()
    model, ema_model, optimizer, loader = accelerator.prepare(model, ema_model, optimizer, loader)

    # Variables for monitoring/logging purposes
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Starting training for {args.num_epochs} epochs...")

    for epoch in range(args.num_epochs):
        logger.info(f"Beginning epoch {epoch}...")

        for i, data in enumerate(tqdm(loader)):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                # Map the input images into the latent space + normalizing latents
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            update_EMA(ema_model, model)

            # Update values
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % len(dataset) == 0:
                # Measuring training speed and such

                if device == "cuda":
                    torch.cuda.synchronize()

                end_time = time()
                steps_per_second = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes

                logger.info(
                    f"Train step: {train_steps}, Train loss: {avg_loss:.4f}, Train steps/second: {steps_per_second:.2f}"
                )

                # Resetting the monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Checkpointing
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "Model": model.module.state_dict(),
                    "EMA_model": ema_model.state_dict(),
                    "Optimizer": optimizer.state_dict(),
                    "args": args,
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()
    logger.info("Done training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="DiffusionTransformer")
    parser.add_argument("--dataset_path", type=str, default="features")
    parser.add_argument("--experiments_path", type=str, default="results")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=3043)  # Easter egg
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50000)
    args = parser.parse_args()
    main(args)
