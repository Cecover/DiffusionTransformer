"""
Created on: 2024, May 10th
Author: 'Cecover' on GitHub

Title: Scalable Diffusion Models with Transformers
Framework used: PyTorch
Code credits:
    - https://github.com/chuanyangjin/fast-DiT/blob/main/diffusion/gaussian_diffusion.py
    - https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
    - https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
    - https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
    - https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
"""

import enum
import math
import torch
import numpy as np
from typing import Optional

from .diffusion_utils import discrete_gaussian_likelihood, kl_score
from tqdm.auto import tqdm


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """
    Take the mean over all non-batch dimensions.
    """

    return x.mean(dim=list(range(1, len(x.shape))))


class ModelMeanType(enum.Enum):
    """
    To specify the type of output, the model predicts.
    """

    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    LEARNED_RANGE has been added to allow the model predict values between FIXED_SMALL and FIXED_LARGE.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()  # Uses raw MSE with RESCALED_KL
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):  # ELBO optimization (L_simple + lambda * L_vb)
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_timesteps, warmup_fraction):
    betas = beta_end * np.ones(num_timesteps, dtype=np.float64)
    warmup_time = int(num_timesteps * warmup_fraction)
    betas[:warmup_time] = np.linspace(
        beta_start, beta_end, warmup_time, dtype=np.float64
    )

    return betas


def get_named_beta_schedules(schedule_name, num_timesteps, alpha_bar: Optional, max_beta=0.999, ):
    """
    Get a pre-defined beta schedule for the given name.

    This code combines both 'get_beta_schedule', 'get_named_beta_schedule', and 'betas_for_alpha_bar', considering it
    just there for the returns.
    """

    if schedule_name == "linear":
        scale = 1000 / num_timesteps
        betas = np.linspace(
            scale * 0.0001, scale * 0.02, num_timesteps, dtype=np.float64
        )

        assert betas.shape == (num_timesteps,)
        return betas

    elif schedule_name == "squaredcos_cap_v2":
        betas = []

        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps

            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

        return np.array(betas)


def _extract_into_tensor(array, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """

    res = torch.from_numpy(array).to(device=timesteps.device)[timesteps].float()

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]

    return res + torch.zeros(broadcast_shape, device=timesteps.device)


class Diffusion:
    """
    Utilities for training and sampling diffusion models.
    """

    def __init__(self, *, betas, model_mean_type, model_variance_type, loss_type):
        self.model_mean_type = model_mean_type
        self.model_variance_type = model_variance_type
        self.loss_type = loss_type

        # Using float64 for accuracy

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        assert len(betas.shape) == 1, "Betas must be 1-dimensional!"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumproduct = np.cumprod(alphas, axis=0)
        self.alphas_prev_cumproduct = np.append(1.0, self.alphas_cumproduct[:-1])
        self.alphas_next_cumproduct = np.append(self.alphas_cumproduct[1:], 0.0)
        assert self.alphas_prev_cumproduct.shape == (self.num_timesteps,)

        # Calculations for diffusion steps/posterior (q(x_t | x_{t-1})) and others
        self.sqrt_alphas_cumproduct = np.sqrt(self.alphas_cumproduct)
        self.sqrt_minus_one_alphas_cumproduct = np.sqrt(1.0 - self.alphas_cumproduct)
        self.log_minus_one_alphas_cumproduct = np.log(1.0 - self.alphas_cumproduct)
        self.sqrt_recip_alphas_cumproduct = np.sqrt(1.0 / self.alphas_cumproduct)
        self.sqrt_recip_minus_one_alphas_cumproduct = np.sqrt(
            1.0 / self.alphas_cumproduct - 1.0
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_prev_cumproduct) / (1.0 - self.alphas_cumproduct)
        )

        # Log calculation is clipped due to posterior variance is 0 at the start of diffusion chain
        self.posterior_clipped_log_variance = (
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
            if len(self.posterior_variance) > 1
            else np.array([])
        )

        self.posterior_mean_coefficient1 = (
                betas
                * np.sqrt(self.alphas_prev_cumproduct)
                / (1.0 - self.alphas_cumproduct)
        )

        self.posterior_mean_coefficient2 = (
                (1.0 - self.alphas_prev_cumproduct)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumproduct)
        )

    def _predict_x_start_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape

        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumproduct, t, x_t.shape) * x_t
                - _extract_into_tensor(
            self.sqrt_recip_minus_one_alphas_cumproduct, t, x_t.shape
        )
                * eps
        )

    def _predict_eps_from_x_start(self, x_t, t, pred_x_start):

        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumproduct, t, x_t.shape) * x_t
                - pred_x_start
        ) / _extract_into_tensor(
            self.sqrt_recip_minus_one_alphas_cumproduct, t, x_t.shape
        )

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Computing the mean for the previous step, given a conditioning function, cond_fn, that computes the gradient
        of a conditional log probability, w.r.t x.

        Since cond_fn computes grad(log(p(y|x))), and we want to condition on y.
        """

        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = (
                p_mean_var["mean"].float() * p_mean_var["variance"] * gradient.float()
        )

        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what p_mean_variance should've been like, assuming the model's score function be conditioned by cond_fn.
        """

        alpha_bar = _extract_into_tensor(self.alphas_cumproduct, t, x.shape)

        eps = self._predict_eps_from_x_start(x, t, p_mean_var["pred_x_start"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)
        out = p_mean_var.copy()

        out["pred_x_start"] = self._predict_eps_from_x_start(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(out["pred_x_start"], x, t)

        return out

    def q_mean_variance(self, x_start, timestep):
        """
        To get the distribution of the posterior.
        """

        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumproduct, timestep, x_start.shape)
                * x_start
        )
        variance = _extract_into_tensor(
            1.0 - self.alphas_prev_cumproduct, timestep, x_start.shape
        )
        log_variance = _extract_into_tensor(
            self.log_minus_one_alphas_cumproduct, timestep, x_start.shape
        )

        return mean, variance, log_variance

    def q_sample(self, x_start, timestep, noise=None):
        """
        Diffusing the data w.r.t to the timestep, i.e., q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        return (
                _extract_into_tensor(self.sqrt_alphas_cumproduct, timestep, x_start.shape)
                * x_start
                + _extract_into_tensor(
            self.sqrt_minus_one_alphas_cumproduct, timestep, x_start.shape
        )
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, timestep):
        """
        Compute the mean and variance of the diffusion posterior, or q(x_{t-1} | x_t, x_0)
        """

        assert x_start.shape == x_t.shape

        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coefficient1, timestep, x_t.shape)
                * x_start
                + _extract_into_tensor(
            self.posterior_mean_coefficient2, timestep, x_t.shape
        )
                * x_t
        )

        posterior_variance = _extract_into_tensor(
            self.posterior_variance, timestep, x_t.shape
        )
        posterior_clipped_log_variance = _extract_into_tensor(
            self.posterior_clipped_log_variance, timestep, x_t.shape
        )

        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_clipped_log_variance.shape[0]
                == x_start.shape[0]
        )

        return posterior_mean, posterior_variance, posterior_clipped_log_variance

    def p_mean_variance(
            self,
            model,
            x,
            timestep,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
    ):
        """
        Applying the model to get p(x_{t-1}|x_t), as well as a prediction of the initial x; x_0.
        """

        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert timestep.shape == (B,)

        model_output = model(x, timestep, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_variance_type in [
            ModelVarType.LEARNED,
            ModelVarType.LEARNED_RANGE,
        ]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)

            min_log = _extract_into_tensor(
                self.posterior_clipped_log_variance, timestep, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), timestep, x.shape)

            # Model variance values would be around [-1, 1] for min_var and max_var, respectively
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)

        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_clipped_log_variance,
                ),
            }[self.model_variance_type]

            model_variance = _extract_into_tensor(model_variance, timestep, x.shape)
            model_log_variance = _extract_into_tensor(
                model_log_variance, timestep, x.shape
            )

        def process_x_start(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
                return x
            if clip_denoised:
                return x.clamp(-1, 1)

        if self.model_mean_type == ModelMeanType.START_X:
            pred_x_start = process_x_start(model_output)
        else:
            pred_x_start = process_x_start(
                self._predict_x_start_from_eps(x, timestep, model_output)
            )

        model_mean, _, _ = self.q_posterior_mean_variance(pred_x_start, x, timestep)

        assert (
                model_mean.shape
                == model_log_variance.shape
                == pred_x_start.shape
                == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x_start": pred_x_start,
            "extra": extra,
        }

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
    ):
        """
        Samples x_{t-1} from the model at given timestep.
        """

        output = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        noise = torch.randn_like(x)
        non_zero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # No noise on timestep 0

        if cond_fn is not None:
            output["mean"] = self.condition_mean(cond_fn, output, x, t, model_kwargs)
        sample = (
                output["mean"]
                + non_zero_mask * torch.exp(0.5 * output["log_variance"]) * noise
        )

        return {"sample": sample, "pred_x_start": output["pred_x_start"]}

    def p_sample_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            verbose=False,
    ):

        device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            image = noise
        else:
            image = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if verbose:
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)

            with torch.no_grad():
                out = self.p_sample(
                    model,
                    image,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                image = out["sample"]

    def p_sample_progressive_looped(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            verbose=False,
    ):
        """
        Generate samples from the model.
        """

        final = None

        for sample in self.p_sample_progressive(
                model,
                shape,
                noise,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
                verbose=False,
        ):
            final = sample

        return final["sample"]
