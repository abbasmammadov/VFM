"""
Adapter Network for Amortized Bayesian Inverse Problems with DMF

The adapter maps degraded observations y and inverse problem class c 
to a distribution over latent noise: q_φ(x₁|y, c)

Architecture: U-Net style encoder with FiLM (Feature-wise Linear Modulation) 
conditioning on inverse problem class.

Input: y (256×256×3 image) + c (scalar class 0-6)
Output: μ (32×32×4) and log_σ² (32×32×4) parameterizing q_φ(x₁|y,c) = N(μ, σ²I)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Applies affine transformation conditioned on external input:
    output = gamma * input + beta
    
    where gamma and beta are predicted from conditioning vector.
    """
    
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.num_features = num_features
        
        # Predict gamma and beta from conditioning
        self.fc = nn.Linear(cond_dim, num_features * 2)
        
        # Initialize to identity transform (gamma=1, beta=0)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        # Set gamma bias to 1
        self.fc.bias.data[:num_features] = 1.0
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, C, H, W]
            cond: Conditioning vector [B, cond_dim]
            
        Returns:
            Modulated features [B, C, H, W]
        """
        # Predict gamma and beta
        params = self.fc(cond)  # [B, 2*C]
        gamma, beta = params.chunk(2, dim=-1)  # [B, C] each
        
        # Reshape for broadcasting
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        
        return gamma * x + beta


class ResBlock(nn.Module):
    """
    Residual block with FiLM conditioning.
    
    Architecture:
    x -> Conv -> GroupNorm -> FiLM -> SiLU -> Conv -> GroupNorm -> FiLM -> SiLU -> + x
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_dim: int,
        num_groups: int = 32,
        use_conv_shortcut: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv block
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.film1 = FiLMLayer(out_channels, cond_dim)
        
        # Second conv block
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.film2 = FiLMLayer(out_channels, cond_dim)
        
        # Shortcut connection
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = x
        
        # First block
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        h = self.film1(h, cond)
        
        # Second block
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.film2(h, cond)
        
        # Residual
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """Downsampling block with residual blocks."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        num_res_blocks: int = 2,
        downsample: bool = True
    ):
        super().__init__()
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock(in_ch, out_channels, cond_dim))
        
        # Downsampling
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x, cond)
        x = self.downsample(x)
        return x


class InverseProblemAdapter(nn.Module):
    """
    Adapter network for amortized inverse problems.
    
    Maps (y, c) -> q_φ(x₁|y, c) = N(μ, σ²I)
    
    Architecture:
    - U-Net style encoder that progressively downsamples 256 -> 32
    - FiLM conditioning on inverse problem class c throughout
    - Outputs mean μ and log-variance log_σ² for reparameterization
    
    The adapter learns to propose "good" initial noise x₁ such that
    the flow map f_θ(x₁, 1, 0) generates samples matching observation y.
    """
    
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        latent_channels: int = 4,
        latent_size: int = 32,
        num_inverse_problems: int = 7,
        cond_dim: int = 256,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),  # 256 -> 128 -> 64 -> 32
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        init_logvar: float = 0.0,  # Initialize log_σ² to small variance
    ):
        """
        Args:
            image_size: Input image resolution (256)
            in_channels: Input image channels (3 for RGB)
            latent_channels: Output latent channels (4 for SDVAE)
            latent_size: Output latent spatial size (32 for 8x downsampling)
            num_inverse_problems: Number of inverse problem classes (7)
            cond_dim: Conditioning embedding dimension
            base_channels: Base number of channels
            channel_mult: Channel multiplier at each resolution
            num_res_blocks: Number of residual blocks per resolution
            dropout: Dropout probability
            init_logvar: Initial value for log variance output
        """
        super().__init__()
        
        self.image_size = image_size
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.cond_dim = cond_dim
        
        # Inverse problem class embedding
        self.class_embed = nn.Sequential(
            nn.Embedding(num_inverse_problems, cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        current_res = image_size
        
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            # Downsample if not at target resolution
            downsample = current_res > latent_size
            
            self.down_blocks.append(
                DownBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    cond_dim=cond_dim,
                    num_res_blocks=num_res_blocks,
                    downsample=downsample
                )
            )
            
            in_ch = out_ch
            if downsample:
                current_res = current_res // 2
        
        # Middle blocks at lowest resolution
        self.mid_block1 = ResBlock(in_ch, in_ch, cond_dim)
        self.mid_block2 = ResBlock(in_ch, in_ch, cond_dim)
        
        # Output projection to latent space
        # Outputs 2x latent_channels: [mean, logvar]
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.conv_out = nn.Conv2d(in_ch, latent_channels * 2, kernel_size=3, padding=1)
        
        # Initialize output layer
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        # Initialize logvar portion to small variance
        self.conv_out.bias.data[latent_channels:] = init_logvar
        
        self.act = nn.SiLU()
        
        # Store init_logvar for clamping
        self.min_logvar = -10.0
        self.max_logvar = 2.0
    
    def forward(
        self, 
        y: torch.Tensor, 
        c: torch.Tensor,
        return_sample: bool = False,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: (y, c) -> (μ, log_σ²)
        
        Args:
            y: Degraded observation [B, 3, 256, 256]
            c: Inverse problem class [B] (integers 0 to num_classes-1)
            return_sample: If True, also return sampled x₁
            temperature: Sampling temperature (scales std dev)
            
        Returns:
            mu: Mean [B, 4, 32, 32]
            logvar: Log variance [B, 4, 32, 32]
            (optional) x1: Sampled latent [B, 4, 32, 32] if return_sample=True
        """
        # Get class conditioning
        cond = self.class_embed(c)  # [B, cond_dim]
        
        # Initial conv
        h = self.conv_in(y)  # [B, base_ch, 256, 256]
        
        # Encoder path
        for down_block in self.down_blocks:
            h = down_block(h, cond)
        
        # Middle blocks
        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)
        
        # Output projection
        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)  # [B, 8, 32, 32]
        
        # Split into mean and logvar
        mu, logvar = h.chunk(2, dim=1)  # [B, 4, 32, 32] each
        
        # Clamp logvar for stability
        logvar = logvar.clamp(self.min_logvar, self.max_logvar)
        
        if return_sample:
            # Reparameterization trick
            std = torch.exp(0.5 * logvar) * temperature
            eps = torch.randn_like(mu)
            x1 = mu + std * eps
            return mu, logvar, x1
        
        return mu, logvar
    
    def sample(
        self, 
        y: torch.Tensor, 
        c: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Sample x₁ from the approximate posterior q_φ(x₁|y, c).
        
        Args:
            y: Degraded observation [B, 3, 256, 256]
            c: Inverse problem class [B]
            num_samples: Number of samples per observation
            temperature: Sampling temperature
            
        Returns:
            x1: Sampled latents [B*num_samples, 4, 32, 32]
        """
        mu, logvar = self.forward(y, c)
        
        B = mu.shape[0]
        
        if num_samples > 1:
            # Expand for multiple samples
            mu = mu.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)
            mu = mu.reshape(B * num_samples, *mu.shape[2:])
            logvar = logvar.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)
            logvar = logvar.reshape(B * num_samples, *logvar.shape[2:])
        
        # Reparameterization
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(mu)
        x1 = mu + std * eps
        
        return x1
    
    def kl_divergence(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute KL divergence KL(q_φ(x₁|y,c) || N(0, I)).
        
        For Gaussian q = N(μ, σ²I) and prior p = N(0, I):
        KL(q||p) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        
        Args:
            mu: Mean [B, C, H, W]
            logvar: Log variance [B, C, H, W]
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            KL divergence
        """
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        if reduction == "mean":
            return kl.mean()
        elif reduction == "sum":
            return kl.sum()
        elif reduction == "none":
            return kl
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


class AdapterSmall(InverseProblemAdapter):
    """Small adapter variant for faster prototyping."""
    
    def __init__(self, **kwargs):
        defaults = dict(
            base_channels=32,
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=1,
            cond_dim=128,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)


class AdapterBase(InverseProblemAdapter):
    """Base adapter variant (default)."""
    
    def __init__(self, **kwargs):
        defaults = dict(
            base_channels=64,
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            cond_dim=256,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)


class AdapterLarge(InverseProblemAdapter):
    """Large adapter variant for best quality."""
    
    def __init__(self, **kwargs):
        defaults = dict(
            base_channels=128,
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=3,
            cond_dim=512,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)


# Model registry
ADAPTER_MODELS = {
    "small": AdapterSmall,
    "base": AdapterBase,
    "large": AdapterLarge,
}


def create_adapter(
    variant: str = "base",
    num_inverse_problems: int = 4,  # For prototype: only 4 problems
    **kwargs
) -> InverseProblemAdapter:
    """Create adapter model by variant name."""
    if variant not in ADAPTER_MODELS:
        raise ValueError(f"Unknown adapter variant: {variant}. "
                        f"Available: {list(ADAPTER_MODELS.keys())}")
    
    return ADAPTER_MODELS[variant](num_inverse_problems=num_inverse_problems, **kwargs)


if __name__ == "__main__":
    # Test adapter
    print("Testing adapter network...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create adapter
    adapter = create_adapter("base", num_inverse_problems=6).to(device)
    print(f"Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    y = torch.randn(batch_size, 3, 256, 256, device=device)
    c = torch.randint(0, 4, (batch_size,), device=device)
    
    mu, logvar = adapter(y, c)
    print(f"Input: {y.shape}, class: {c.shape}")
    print(f"Output mu: {mu.shape}, logvar: {logvar.shape}")
    
    # Test sampling
    mu, logvar, x1 = adapter(y, c, return_sample=True)
    print(f"Sampled x1: {x1.shape}")
    
    # Test KL
    kl = adapter.kl_divergence(mu, logvar)
    print(f"KL divergence: {kl.item():.4f}")
    
    # Test multi-sample
    x1_multi = adapter.sample(y, c, num_samples=5)
    print(f"Multi-sample x1: {x1_multi.shape}")
    
    print("\nAdapter test passed!")
