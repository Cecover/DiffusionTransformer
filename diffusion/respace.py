"""
Created on: 2024, May 12th
Author: 'Cecover' on GitHub

Code credits:
    - https://github.com/chuanyangjin/fast-DiT/blob/main/diffusion/respace.py
    - https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
    - https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
    - https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
"""

import torch
import numpy as np

from .gaussian_diffusion import Diffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Creates a list of timesteps to use from an original diffusion process, given the number of
    timesteps we want to take equally sized portions of the original process.
    """

    start_idx = 0
    all_steps = []

    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))

            raise ValueError(
                f"Cannot create exactly {num_timesteps} steps with an integer stride!"
            )
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)

    for i, section_count in enumerate(section_counts):

        curr_idx = 0.0
        taken_steps = []

        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"Cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)

        for _ in range(section_count):
            taken_steps.append(start_idx + round(curr_idx))
            curr_idx += frac_stride

        all_steps += taken_steps
        start_idx += size

    return set(all_steps)


class _WrappedModel:
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.original_num_steps = original_num_steps

    def __call__(self, x, t, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=t.device)
        new_t = map_tensor[t]

        return self.model(x, new_t, **kwargs)


class SpacedDiffusion(Diffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    (I need to read more papers)
    """

    def __init__(self, use_timesteps, **kwargs):

        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = Diffusion(**kwargs)
        last_alpha_cumulative_product = 1.0
        new_betas = []

        for i, alpha_cumulative_product in enumerate(base_diffusion.alphas_cumproduct):
            if i in self.use_timesteps:
                new_betas.append(
                    1 - alpha_cumulative_product / last_alpha_cumulative_product
                )
                last_alpha_cumulative_product = alpha_cumulative_product
                self.timestep_map.append(i)

        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model

        return _WrappedModel(model, self.timestep_map, self.original_num_steps)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)
