"""
Created on: 2024, March 11th
Author: 'Cecover' on GitHub
Framework used: PyTorch

Code credits (sorted, probably randomly ordered IDK):
    - https://github.com/facebookresearch/DiT/blob/main/models.py
    - https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/helpers.py
    - https://github.com/pprp/timm/blob/master/timm/layers/patch_embed.py
    - https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
    - https://github.com/sooftware/attentions/blob/master/attentions.py
    - https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py

Light disclaimer: Some of the documentation would not reflect what the code does (I am too lazy to check everything,
                  but it works very well)
"""

# ===== Imports =====
# General imports
import math
import numpy as np
import collections.abc
from itertools import repeat
from typing import Optional, Tuple

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _assert


# ===== Helper functions =====
def modulate(tensor, shift, scale):
    return tensor * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def n_tuples(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = n_tuples(2)  # Used for lower modules


def pos_embed_2D(
    hidden_dimension: int,  # Hidden dimension size (needs to match the global hidden dimension size)
    grid_size: int,  # Halved number of patches; int(num_patches**0.5)
    use_class_token: bool = False,  # Defaults to False on official implementation
    num_extra_tokens: int = 0  # Defaults to zero on official implementation
) -> torch.Tensor:
    """
    Adds Sine-Cosine positional encoding (Sinusoidal) positional embeddings.

    Note: This code combines all three functions into one single function.

    Code is also reusable for any other tasks.

    Use case:

        # Positional parameter initialization
        self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, hidden_size),
                requires_grad=False
            )

        # Positional embedding initialization
        Positional_embedding = pos_embed_2D(
                hidden_dimension=self.pos_embed.shape[-1],
                grid_size=int(self.x_embedder.num_patches**0.5),
                class_token=False,
                num_extra_tokens=0
            )

        # Adding the positional embedding and freezing the layer since it was not learned
        Self.pos_embed.data.copy_(
            torch.from_numpy(positional_embedding).float().unsqueeze(0)  # This is to freeze the layers
        )

    Input: Positional encoding parameter of zeros with shape [1, num_patches, hidden_size] (Input shape w/ batch of 1)
    Returns: a Sinusoidal positional embedding tensor of [grid_size * grid_size, embed_dim] w/token, or
                                                         [1 + grid_size * grid_size, embed_dim] w/o token
    """

    # Initialize grids
    grid_height = np.arange(grid_size, dtype=np.float32)
    grid_width = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_width, grid_height), axis=0)  # Width goes first
    reshaped_grid = grid.reshape([2, 1, grid_size, grid_size])

    assert hidden_dimension % 2 == 0

    # Where the magic happens
    halved_embed_dim = hidden_dimension // 2  # Using half dimension to encode grid
    omega = np.arange(halved_embed_dim // 2, dtype=np.float32)
    omega /= halved_embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    grid_0 = reshaped_grid[0].reshape(-1)  # (M, )
    grid_1 = reshaped_grid[1].reshape(-1)  # (M, )
    out_0 = np.einsum("m, d -> md", grid_0, omega)  # (M, D/2), outer product
    out_1 = np.einsum("m, d -> md", grid_1, omega)  # (M, D/2), outer product

    sin_0 = np.sin(out_0)
    sin_1 = np.sin(out_1)
    cos_0 = np.cos(out_0)
    cos_1 = np.cos(out_1)

    # Concatenate both embeddings
    embedding_0 = np.concatenate([sin_0, cos_0], axis=1)
    embedding_1 = np.concatenate([sin_1, cos_1], axis=1)
    final_embedding = np.concatenate([embedding_0, embedding_1], axis=1)

    # Adding extra tokens
    if use_class_token and num_extra_tokens > 0:
        final_embedding = np.concatenate(
            [np.zeros([num_extra_tokens, hidden_dimension]), final_embedding], axis=0
        )

    return final_embedding


class PatchEmbed(nn.Module):
    """
    Turns a 2D image into patched images.

    The current implementation simply turns the images into patches, not using things such as sliding windows or any
    other image patching methods.

    The function is also reusable for non-Diffusion use-cases.

    Use case:

        self.x_embedder = PatchEmbed(
                input_size=32,
                patch_size=8,
                in_channels=4,
                hidden_dimension=768,
                use_norm=False,
                use_flatten=True,
                use_bias=True
            )

    Input: A tensor of [batch_size, channels, height, width]
    Returns: A tensor of [batch_size, patches_count, hidden_dimension]
    """

    def __init__(
        self,
        image_size: int,  # Input image size
        patch_size: int,  # Desired patch size
        in_channels: int,  # Defaults to 4 on paper, but differ from use cases
        hidden_dimension: int,  # Hidden dimension for the layer normalization
        use_norm: bool = False,  # Defaults to False on official implementation
        use_flatten: bool = True,  # Defaults to True on official implementation
        use_bias: bool = True  # Defaults to True on official implementation
    ):
        super().__init__()

        # Images init
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Other functions init
        self.use_flatten = use_flatten
        self.projection = nn.Conv2d(
            in_channels,
            hidden_dimension,
            kernel_size=patch_size,
            stride=patch_size,
            bias=use_bias
        )
        self.norm = nn.LayerNorm(hidden_dimension) if use_norm else nn.Identity()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = (
            image.shape
        )  # Image shape assumed to be [Batch, Channel, Height, Weight]

        # Size assertion
        _assert(
            H == self.image_size[0],
            f"Input image height ({H}) does not match ({self.image_size[0]}!",
        )
        _assert(
            W == self.image_size[0],
            f"Input image height ({W}) does not match ({self.image_size[1]}!",
        )

        image = self.projection(image)
        # [Batch, Channels, Height, Width] -> [Batch, Hidden_Dimension, H_Patch_Count, W_Patch_Count]

        if self.use_flatten:
            image = image.flatten(2).transpose(1, 2)
            # [Batch, Hidden_Dimension, H_Patch_Count, W_Patch_Count] -> [Batch, H*W_Patch_Count, Hidden_Dimension]

        image = self.norm(image)  # No shape changes here

        return image


# ===== Main functions =====
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timestep into a vector representation.

    Or in simple terms, turning the timestep into a vector representation, which later can be used to condition the
    denoising model along with class labels representations.

    Use case:

        self.t_embedder = TimestepEmbedder(hidden_dimension=1152,
                                           frequency_max_period=10000,
                                           frequency_embedding_size=256,
                                           use_perceptron_bias=True)

    Input: A tensor of [batch_size, ]
    Returns: A tensor of [batch_size, hidden_size]
    """

    def __init__(
        self,
        hidden_dimension: int,  # Hidden size, follows global hidden size
        frequency_max_period: int,  # Frequency max period, defaults to 10000
        frequency_embedding_size: int,  # Does not follow global hidden size
        use_perceptron_bias: bool  # Enables/disables bias on the perceptron model
    ):
        super().__init__()

        self.frequency_embedding_size = frequency_embedding_size
        self.frequency_max_period = frequency_max_period

        self.perceptron = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, hidden_dimension, bias=use_perceptron_bias
            ),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension, bias=use_perceptron_bias),
        )

    @staticmethod
    def time_stepper(
        tensor: torch.Tensor, embedding_size: int, max_period: int
    ) -> torch.Tensor:
        half = embedding_size // 2
        frequency = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=tensor.device)  # MUST
        args = tensor[:, None].to(torch.float) * frequency[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep_frequency = self.time_stepper(
            timestep, self.frequency_embedding_size, self.frequency_max_period
        )
        timestep_embedding = self.perceptron(timestep_frequency)

        return timestep_embedding


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representation.

    Just like TimestepEmbedder, this module embeds class labels into vector representation for denoising conditioning.

    Code also handles label dropout for classifier-free guidance.

    Use case:

        self.y_embedder = LabelEmbedder(num_classes = 1000,
                                        hidden_dimension = 1152,
                                        class_dropout_prob = 0.1)

    Input: A tensor of [batch_size, ]
    Returns: A tensor of [batch_size, hidden_size]

    Please note that the output must be the same as the output of TimestepEmbedder module as it would be concatenated
    together.
    """

    def __init__(
        self,
        num_classes: int,  # Number of classes of the dataset, defaults to 1000 since ImageNet
        hidden_dimension: int,  # Follows global hidden size
        dropout_prob: float  # Dropout probability, defaults to 0.1
    ):
        super().__init__()

        use_config_embedding = dropout_prob > 0

        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding_table = nn.Embedding(
            num_classes + use_config_embedding, hidden_dimension
        )

    def token_dropper(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Randomly drops labels to enable classifier-free guidance using dropout.

        This code removes 'force_drop_ids' since it was unused in the original implementation.
        """

        drop_id = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        new_labels = torch.where(drop_id, self.num_classes, labels)

        return new_labels

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0

        if use_dropout:
            labels = self.token_dropper(labels)  # Classifier-free conditioning

        embeddings = self.embedding_table(labels)

        return embeddings


# ===== Core Diffusion Model Model =====
class Attention(nn.Module):
    """
    Modularized Attention mechanism, so it is easier to be modified and reused on later uses.

    Implementation: Multi-head scaled dot product attention - Vaswani et al. in 2017.

    Use case:

        self.attention = Attention(hidden_dimension=1152,
                                   num_attention_heads=8)

    Input: Three tensors of [batch size, token size, hidden dimension] and an optional mask tensor
    Returns: A tuple tensor of [batch size, token size, hidden dimension]

    The output of this function requires having [0] on the back since it was a tuple object.
        - [0] returns the actual end value
        - [1] returns the attention matrix
    """

    def __init__(
        self,
        hidden_dimension: int,  # Follows global hidden size
        num_attention_heads: int  # Number of attention heads
    ):
        super().__init__()

        assert (
            hidden_dimension % num_attention_heads == 0
        ), "Attention hidden size and number of attention heads must be divisible!"

        self.num_heads = num_attention_heads
        self.div_dimension = int(hidden_dimension / num_attention_heads)

        self.query_projection = nn.Linear(
            hidden_dimension, self.num_heads * self.div_dimension
        )
        self.key_projection = nn.Linear(
            hidden_dimension, self.num_heads * self.div_dimension
        )
        self.value_projection = nn.Linear(
            hidden_dimension, self.num_heads * self.div_dimension
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = value.size(0)

        # Q, K, V projections
        query = self.query_projection(query).view(
            batch_size, -1, self.num_heads, self.div_dimension
        )  # B * Q_len * H * D

        key = self.query_projection(key).view(
            batch_size, -1, self.num_heads, self.div_dimension
        )  # B * K_len * H * D

        value = self.query_projection(value).view(
            batch_size, -1, self.num_heads, self.div_dimension
        )  # B * V_len * H * D

        # Permutations
        query = (
            query.permute(2, 0, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.div_dimension)
        )  # (B*N) * Q_len * D

        key = (
            key.permute(2, 0, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.div_dimension)
        )  # (B*N) * K_len * D

        value = (
            value.permute(2, 0, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.div_dimension)
        )  # (B*N) * V_len * D

        # Calculating the attention score
        attention_score = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(
            self.div_dimension
        )

        # Mask embedding
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(
                1, self.num_heads, 1, 1
            )  # B * N * Q_len * K_len
            attention_score.masked_fill_(
                mask.view(attention_score.size()), -float("Inf")
            )

        # Calculating the final value
        attention_matrix = F.softmax(attention_score, dim=-1)
        context = torch.bmm(attention_matrix, value)
        context = context.view(self.num_heads, batch_size, -1, self.div_dimension)
        context = (
            context.permute(1, 2, 0, 3)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.div_dimension)
        )  # B * T * (N*D)

        return context, attention_matrix


class AttentionPerceptron(nn.Module):
    """
    This needs to be separated since there is research regarding better alternatives to the usual MLP.

    Use case:

        self.perceptron = AttentionPerceptron(
            in_features=1152,  # Follows the attention hidden dimension
            hidden_size=4608,
            out_features=1152,  # Follows the attention hidden dimension
            use_layer_norm=True,
            use_bias=True,
            dropout_prob=0.1,
        )

    Input: A tensor of [batch size, token size, hidden dimension] from Attention
    Returns: A tensor of [batch size, token size, hidden dimension]
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int],
        out_features: Optional[int],
        use_layer_norm: bool,
        use_bias: bool,
        dropout_prob: float
    ):
        super().__init__()

        out_features = in_features if out_features is None else out_features
        hidden_features = in_features if hidden_features is None else hidden_features

        bias = to_2tuple(use_bias)
        drop_probs = to_2tuple(dropout_prob)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.activation = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            nn.LayerNorm(hidden_features) if use_layer_norm is True else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class DiffusionTransformer(nn.Module):
    """
    A transformer block for diffusion models using adaptive layer norm zero (adaLN-Zero) conditioning.

    Use case:

        self.attn_blocks = nn.ModuleList(
            [
                DiffusionTransformer(
                    attention_hidden_dimension=1152,
                    num_attention_heads=8,
                    perceptron_dropout_rate=0.0,
                    perceptron_use_bias=True,
                    perceptron_use_layer_norm=True,
                    transformer_norm_affine=False,
                    transformer_norm_epsilon=1e-6,
                    transformer_mlp_ratio=4,
                    global_hidden_dimension=1152
                )
                for _ in range(depth)
            ]
        )


    Input: - Three tensors of [batch size, num_patches, hidden dimension] and an optional mask tensor (Q, K, V, M)
           - A tensor of [batch_size, hidden_size] (please refer to LabelEmbedding and TimestepEmbedder returns)
    Returns: A tensor of [batch size, num_patches, hidden dimension]
    """

    def __init__(
        self,
        attention_hidden_dimension: int,
        num_attention_heads: int,
        perceptron_dropout_rate: float,
        perceptron_use_bias: bool,
        perceptron_use_layer_norm: bool,
        transformer_use_norm_affine: bool,
        transformer_norm_epsilon: float,
        transformer_mlp_ratio: float,
        global_hidden_dimension: int
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            global_hidden_dimension,
            elementwise_affine=transformer_use_norm_affine,
            eps=transformer_norm_epsilon
        )
        self.norm2 = nn.LayerNorm(
            global_hidden_dimension,
            elementwise_affine=transformer_use_norm_affine,
            eps=transformer_norm_epsilon
        )
        self.attention = Attention(attention_hidden_dimension, num_attention_heads)
        self.perceptron_hidden_dim = int(
            global_hidden_dimension * transformer_mlp_ratio
        )
        self.perceptron = AttentionPerceptron(
            in_features=global_hidden_dimension,
            hidden_features=self.perceptron_hidden_dim,
            out_features=global_hidden_dimension,
            dropout_prob=perceptron_dropout_rate,
            use_bias=perceptron_use_bias,
            use_layer_norm=perceptron_use_layer_norm
        )
        self.adaptiveLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_hidden_dimension, 6 * global_hidden_dimension, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate = (
            self.adaptiveLN(c).chunk(6, dim=1)
        )
        attn_input = modulate(self.norm1(x), attn_shift, attn_scale)
        attn_out = (
            x
            + attn_gate.unsqueeze(1)
            * self.attention(query=attn_input, key=attn_input, value=attn_input)[0]
        )  # This is a tuple
        perceptron_input = modulate(self.norm2(attn_out), mlp_shift, mlp_scale)
        x = x + mlp_gate.unsqueeze(1) * self.perceptron(perceptron_input)

        return x


class FinalLayer(nn.Module):
    """
    The final DiT layer.

    Use case:

        self.finallayer = FinalLayer(hidden_size=1152,
                                    patch_size=8,
                                    out_channels=2304)


    Input: - A tensor of [batch_size, num_patches, hidden dimension]
           - A tensor of [batch_size, hidden_size] (please refer to LabelEmbedding and TimestepEmbedder returns)
    Returns: A tensor of [batch size, token size, hidden dimension]

    This model is connected to the transformer layer above, thus requires the same Tensor shapes.
    """

    def __init__(
        self,
        global_hidden_dimension: int,
        patch_size: int,
        out_channels: int,
        use_norm_affine: bool,
        norm_epsilon: float,
        use_perceptron_bias: bool,
        use_adaptive_norm_bias: bool
    ):
        super().__init__()
        self.layernorm = nn.LayerNorm(
            global_hidden_dimension,
            elementwise_affine=use_norm_affine,
            eps=norm_epsilon
        )
        self.linear = nn.Linear(
            global_hidden_dimension,
            patch_size * patch_size * out_channels,
            bias=use_perceptron_bias
        )
        self.AdaptiveLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                global_hidden_dimension,
                2 * global_hidden_dimension,
                bias=use_adaptive_norm_bias
            ),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.AdaptiveLN(c).chunk(2, dim=1)
        x = modulate(self.layernorm(x), shift, scale)
        x = self.linear(x)

        return x


class DiTNet(nn.Module):
    """
    A conditioning model based of Vision Transformer architecture.
    """

    def __init__(
        self,
        input_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        global_hidden_dimension: int,
        transformer_depth: int,
        transformer_attn_heads: int,
        transformer_mlp_ratio: int,
        dropout_prob: float,
        learn_sigma: bool
    ):
        super().__init__()

        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.x_embedder = PatchEmbed(
            image_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dimension=global_hidden_dimension,
            use_norm=False,
            use_flatten=True,
            use_bias=True
        )
        self.t_embedder = TimestepEmbedder(
            hidden_dimension=global_hidden_dimension,
            frequency_max_period=10000,
            frequency_embedding_size=256,
            use_perceptron_bias=True
        )
        self.y_embedder = LabelEmbedder(
            num_classes=num_classes,
            hidden_dimension=global_hidden_dimension,
            dropout_prob=dropout_prob
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, global_hidden_dimension),
            requires_grad=False
        )
        self.attn_blocks = nn.ModuleList(
            [
                DiffusionTransformer(
                    attention_hidden_dimension=global_hidden_dimension,
                    num_attention_heads=transformer_attn_heads,
                    transformer_use_norm_affine=False,
                    transformer_norm_epsilon=1e-6,
                    transformer_mlp_ratio=transformer_mlp_ratio,
                    perceptron_dropout_rate=0.0,
                    perceptron_use_bias=True,
                    perceptron_use_layer_norm=True,
                    global_hidden_dimension=global_hidden_dimension
                )
                for _ in range(transformer_depth)
            ]
        )
        self.finallayer = FinalLayer(
            global_hidden_dimension=global_hidden_dimension,
            patch_size=patch_size,
            out_channels=self.out_channels,
            use_norm_affine=False,
            norm_epsilon=1e-6,
            use_perceptron_bias=True,
            use_adaptive_norm_bias=True
        )

    def init_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)

        # Initialize positional embedding (and freeze it)
        positional_embedding = pos_embed_2D(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5), False, 0
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(positional_embedding).float().unsqueeze(0)
        )

        # Initializing patch_embed-like nn.Linear
        projection_weights = self.x_embedder.projection.weight.data
        nn.init.xavier_uniform_(
            projection_weights.view([projection_weights.shape[0], -1])
        )
        nn.init.constant_(self.x_embedder.projection.bias, 0)

        # Initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding perceptron
        nn.init.normal_(self.t_embedder.perceptron[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.perceptron[2].weight, std=0.02)

        # Zeroing out adaLN modulation
        for block in self.attn_blocks:
            nn.init.constant_(block.adaptiveLN[-1].weight, 0)
            nn.init.constant_(block.adaptiveLN[-1].bias, 0)

        # Zeroing out output layers
        nn.init.constant_(self.finallayer.AdaptiveLN[-1].weight, 0)
        nn.init.constant_(self.finallayer.AdaptiveLN[-1].bias, 0)
        nn.init.constant_(self.finallayer.linear.weight, 0)
        nn.init.constant_(self.finallayer.linear.bias, 0)

    def unpatch(self, x):
        """
        x -> [B, T, patch_size ** 2 * C]
        image -> [B, C, H, W]
        """

        channels = self.out_channels
        patch_size = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)

        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum("bhwpqc->bchpwq", x)
        image = torch.reshape(
            x, shape=(x.shape[0], channels, h * patch_size, w * patch_size)
        )

        return image

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        """
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y)
        c = t + y

        for block in self.attn_blocks:
            x = block(x, c)

        x = self.finallayer(x, c)
        x = self.unpatch(x)

        return x


# ===== Testing purposes =====

# model = DiTNet(
#     input_size=32,
#     patch_size=8,
#     in_channels=4,
#     num_classes=1000,
#     global_hidden_dimension=1152,
#     transformer_depth=28,
#     transformer_attn_heads=8,
#     transformer_mlp_ratio=4.0,
#     dropout_prob=0.1,
#     learn_sigma=True,
# )
#
# x = torch.rand((3,))
# y = torch.randint(low=1, high=999, size=(3,))
# value = torch.rand((3, 4, 32, 32))
# result = model(value, x, y)
# print(result.shape)
