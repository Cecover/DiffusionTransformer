"""
Created on: 2024, May 10th
Author: 'Cecover' on GitHub

Code credits:
    - https://github.com/chuanyangjin/fast-DiT/blob/main/diffusion/__init__.py
    - https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
    - https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
    - https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
"""

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear",
    use_kl_divergence=False,
    sigma_small=False,
    predict_x_start=False,
    learn_sigma=True,
    rescale_learned_sigma=False,
    steps=1000,
):
    betas = gd.get_named_beta_schedules(noise_schedule, steps, alpha_bar=None)

    if use_kl_divergence:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigma:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON
            if not predict_x_start
            else gd.ModelMeanType.START_X
        ),
        model_variance_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
    )
