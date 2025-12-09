import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embed import PatchEmbed
from .mlp import Mlp
from .attention import Attention
from .rope import RotaryPositionEmbedding2D, PositionGetter
from diffusers import ModelMixin

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, rope=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, rope=rope, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, pos=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos=pos)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(ModelMixin):
    """
    Cascade diffusion model with a transformer backbone.
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=1,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        add_zero_convs=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        rope_freq = 100
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.x_embedder = PatchEmbed(in_chans=in_channels, embed_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, rope=self.rope) for _ in range(depth)]
        )

        self.proj_fusion = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size*4),
                nn.SiLU(),
                nn.Linear(hidden_size*4, hidden_size*4),
                nn.SiLU(),
                nn.Linear(hidden_size*4, hidden_size*4),
            )

        self.final_layer = FinalLayer(hidden_size, 8, self.out_channels)
        self.initialize_weights()

        self.add_zero_convs = add_zero_convs
        if self.add_zero_convs:
            self.initial_zero_conv = nn.Linear(hidden_size, hidden_size)
            self.zero_convs = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size) for _ in range(depth)]
            )

            for zero_conv in [self.initial_zero_conv, *self.zero_convs]:
                nn.init.constant_(zero_conv.weight, 0)
                nn.init.constant_(zero_conv.bias, 0)
        

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, height, width):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = 8
        h = height // p
        w = width // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x=None, semantics=None, timestep=None, dropout=0.1):
        """
        Forward pass of SP-DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """

        if self.add_zero_convs:
            raise NotImplementedError("ControlNet-style DiT cannot be inferred on its own. Only works when used in a ControlNetDiT.")

        N, C, H, W = x.shape
        if len(timestep.shape) == 0:
            timestep = timestep[None]

        pos0 = None
        pos1 = None
        if self.rope is not None:
            pos0 = self.position_getter(N, H // 16, W // 16, device=x.device)
            pos1 = self.position_getter(N, H // 8, W // 8, device=x.device)

        x = self.x_embedder(x)
        N, T, D = x.shape
        t = self.t_embedder(timestep)  # (N, D)

        # for block in self.blocks:
        for i, block in enumerate(self.blocks):
            if i < 12:
                x = block(x, t, pos0)  # (N, T, D)
            else:
                x = block(x, t, pos1)  # (N, T, D)

            if i == 11:

                semantics = F.normalize(semantics, dim=-1)
                x = self.proj_fusion(torch.cat([x, semantics], dim=-1))
                p = 16
                x = x.reshape(shape=(N, H//p, W//p, 2, 2, D))
                x = torch.einsum("nhwpqc->nchpwq", x)
                x = x.reshape(shape=(N, D, (H//p)*2, (W//p)*2))
                x = x.flatten(2).transpose(1, 2)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, height=H, width=W)  # (N, out_channels, H, W)
        return x

class ControlNetDiT(ModelMixin):
    _supports_gradient_checkpointing = True

    def __init__(self, dit, conditioning_dits):
        super().__init__()
        self.dit = dit
        self.conditioning_dits = nn.ModuleList(conditioning_dits)
        self.gradient_checkpointing = False

        assert not self.dit.add_zero_convs, "ControlNetDiT cannot be used with add_zero_convs=True."
        assert all([conditioning_dit.add_zero_convs for conditioning_dit in self.conditioning_dits]), "Controlnet conditioning DiTs must be used with add_zero_convs=True."


    def forward(self, x, conds, semantics, timestep, dropout=0.1):
        """
        Forward pass of SP-DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """

        assert len(conds) == len(self.conditioning_dits), "Number of conditions must match number of conditioning DiTs."
        assert all([cond.shape == x.shape for cond in conds]), "All conditions must have the same shape as the input."

        N, C, H, W = x.shape
        if len(timestep.shape) == 0:
            timestep = timestep[None]

        pos0 = None
        pos1 = None
        if self.dit.rope is not None:
            pos0 = self.dit.position_getter(N, H // 16, W // 16, device=x.device)
            pos1 = self.dit.position_getter(N, H // 8, W // 8, device=x.device)

        x_dit = self.dit.x_embedder(x)
        N, T, D = x_dit.shape
        t = self.dit.t_embedder(timestep)  # (N, D)
        
        x_conditioning = [conditioning_dit.x_embedder(cond) for conditioning_dit, cond in zip(self.conditioning_dits, conds)]
        x_conditioning = [conditioning_dit.initial_zero_conv(x_conditioning_j) for conditioning_dit, x_conditioning_j in zip(self.conditioning_dits, x_conditioning)]

        # for block in self.blocks:
        for i, block in enumerate(self.dit.blocks):
            if self.gradient_checkpointing:
                x_dit, *x_conditioning = self._gradient_checkpointing_func(self.process_block, t, i, x_dit, x_conditioning, semantics, pos0, pos1, N, H, W, D)
            else:
                x_dit, *x_conditioning = self.process_block(t, i, x_dit, x_conditioning, semantics, pos0, pos1, N, H, W, D)

        x_dit = self.dit.final_layer(x_dit, t)  # (N, T, patch_size ** 2 * out_channels)
        x_dit = self.dit.unpatchify(x_dit, height=H, width=W)  # (N, out_channels, H, W)
        return x_dit
    
    def process_block(self, t, i, x_dit, x_conditioning, semantics, pos0, pos1, N, H, W, D):
        block = self.dit.blocks[i]
        if i < 12:
            x_dit = block(x_dit, t, pos0)  # (N, T, D)
            x_conditioning = [conditioning_dit.blocks[i](x_conditioning_j, t, pos0) for conditioning_dit, x_conditioning_j in zip(self.conditioning_dits, x_conditioning)]
        else:
            x_dit = block(x_dit, t, pos1)  # (N, T, D)
            x_conditioning = [conditioning_dit.blocks[i](x_conditioning_j, t, pos1) for conditioning_dit, x_conditioning_j in zip(self.conditioning_dits, x_conditioning)  ]

        if i == 11:

            x_dit = self._merge_semantics(x_dit, semantics, self.dit, N, H, W, D)
            x_conditioning = [self._merge_semantics(x_conditioning_i, semantics, conditioning_dit, N, H, W, D) for x_conditioning_i, conditioning_dit in zip(x_conditioning, self.conditioning_dits)]
        
        for (conditioning_dit, x_conditioning_j) in zip(self.conditioning_dits, x_conditioning):
            x_dit = x_dit + conditioning_dit.zero_convs[i](x_conditioning_j)

        return x_dit, *x_conditioning


    @staticmethod
    def _merge_semantics(x, semantics, dit, N, H, W, D):

        semantics = F.normalize(semantics, dim=-1)
        x = dit.proj_fusion(torch.cat([x, semantics], dim=-1))
        p = 16
        x = x.reshape(shape=(N, H//p, W//p, 2, 2, D))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(N, D, (H//p)*2, (W//p)*2))
        x = x.flatten(2).transpose(1, 2)
        return x
    
    def requires_grad_(self, value):
        for param in self.dit.parameters():
            param.requires_grad = False
        for conditioning_dit in self.conditioning_dits:
            for param in conditioning_dit.parameters():
                param.requires_grad = value
        return self
