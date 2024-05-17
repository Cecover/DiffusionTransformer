"""
Created on: 2024, March 11th
Author: 'Cecover' on GitHub

Title: Scalable Diffusion Models with Transformers
Framework used: Pytorch
Code reference: - https://github.com/facebookresearch/DiT/blob/main/train.py (original)
                - https://github.com/chuanyangjin/fast-DiT/blob/main/train.py (re-written)
                - https://github.com/chuanyangjin/fast-DiT/blob/main/train_options/train_original.py (re-written)

Difference between this and referenced code:
1. Written from scratch
2. General code cleanup
3. Added documentation
4. Complete move to HuggingFace Accelerate

Current TODO's:
1. Finishing the rest of the code
2. Clean up and restructure code
3. Test code

Some comments:
1. This code would only use ImageNet dataset and its variants
"""

# ===== General Imports =====
import os
import wandb
import logging
import argparse
import numpy as np
from glob import glob
from PIL import Image
from time import time
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL
from accelerate import Accelerator

# Created imports
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


# Creating a custom dataset
class CustomDataset(Dataset):
    """
    Creates a custom dataset, which can be loaded using PyTorch dataloaders.

    The current implementation takes imagenette (https://github.com/fastai/imagenette), thus it would be 'kinda'
    specialized, but it can be replaced with other datasets.
    """

    def __init__(self, features_dir: str, labels_dir: str, transforms):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))
        self.transforms = transforms

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

        if self.transforms:
            features = self.transforms(features)

        return features, labels


# Dataset transformations

class center_crop_array(torch.nn.Module):
    def forward(self, pil_image):
        """
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126

        I modified this function, so it would work with multiprocessing dataloader.
        Due to the nature of the code, this needs to be set MANUALLY until I can fix this.
        But again, it WILL work with multiprocessing.
        """
        while min(*pil_image.size) >= 2 * 256:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = 256 / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - 256) // 2
        crop_x = (arr.shape[1] - 256) // 2
        return Image.fromarray(
            arr[crop_y: crop_y + 256, crop_x: crop_x + 256]
        )


# ===== Main Training Loop =====
def main(args):
    """
    The main trainer. Only works with a GPU!
    The trainer now uses HF's Accelerate instead of DDP, which means I wrote the same code twice.
    """

    # Setting up the accelerator
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=args.precision)
    ensure_distributed()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(args.experiments_path, exist_ok=True)  # To make the results directory
        experiment_index = len(glob(f"{args.experiments_path}/*"))
        model_string_name = args.experiment_name
        experiment_dir = (
            f"{args.experiments_path}/{experiment_index:03d}-{model_string_name}"
        )
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        print(f'World size: {accelerator.num_processes}', flush=True)
        print(f'Batch size: {args.batch_size * accelerator.num_processes}', flush=True)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Setting up the model and dataset
    # Latent image sizing
    assert (
            args.image_size % 8 == 0
    ), "Image size must be divisible by 8 for the VAE encoder!"
    latent_size = args.image_size // 8

    # Loading the transformer and EMA model (parameterization is already done inside the model)
    # model = DiTNet(input_size=latent_size, num_classes=args.num_classes).to(device)
    model = DiTNet(
        input_size=32,
        patch_size=8,
        in_channels=4,
        num_classes=1000,
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
    diffusion = create_diffusion(
        timestep_respacing=""
    )  # Default would be 1000 steps, with linear noise scheduling
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

    # Initializing wandb, so we can actually see what is going on during training.
    # It will be ON by default.
    log_config = vars(args)
    log_config["parameters"] = sum(p.numel() for p in model.parameters())
    wandb.init(project="DiffusionTransformer",
               config=log_config,
               save_code=True)
    wandb.watch(model)

    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setting up the optimizer
    # On paper, they use Adam with betas=(0.9, 0.999), and a constant lr of 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

    # Dataset loading and transformations
    transform = transforms.Compose(
        [
            center_crop_array(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std, inplace=True),
        ]
    )

    features_dir = f"{args.dataset_path}"
    # labels_dir = f"{args.dataset_label_path}"
    # dataset = CustomDataset(features_dir, labels_dir, transform)
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
            print(f"Used dataset: {args.dataset_path} with length of {len(dataset):,}")
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

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                # Map the input images into the latent space + normalizing latents
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_step(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            update_EMA(ema_model, model)

            # Log-loss values
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed
                torch.cuda.synchronize()
                end_time = time()
                steps_per_second = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes

                logger.info(
                    f"(Train step={train_steps:07d}) "
                    f"Train loss={avg_loss:.4f}, "
                    f"Train steps/second: {steps_per_second:.2f}"
                )

                # Resetting the monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Checkpointing
            if train_steps % args.save_every == 0 and train_steps > 0:
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
