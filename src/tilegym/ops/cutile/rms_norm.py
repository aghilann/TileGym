# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
RMSNorm with CuTile kernel - supports forward and backward passes with autotuning.

Based on TileGym's RMSNorm implementation.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import cuda.tile as ct
import cuda.tile_experimental as ct_experimental

from tilegym.backend import register_impl

from .utils import next_power_of_2


# ============================================================================
# CuTile Kernels
# ============================================================================

@ct.kernel(occupancy=2)
def rms_norm_backward_kernel(
    dx,
    dy,
    x,
    weight,
    Rstd,
    temp_buffer,
    TILE_SIZE: ct.Constant[int],
):
    """Compute input gradients for RMSNorm backward pass."""
    row_idx = ct.bid(0)
    M, N = x.shape

    input_row = ct.load(x, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    gradient_row = ct.load(dy, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    inv_std_row = ct.load(Rstd, index=(row_idx,), shape=(1,), padding_mode=ct.PaddingMode.ZERO)
    inv_std_row = ct.reshape(inv_std_row, (1, 1))
    weight_vector = ct.load(weight, index=(0,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    weight_vector = ct.reshape(weight_vector, (1, TILE_SIZE))

    c1 = input_row * gradient_row
    c2 = c1 * inv_std_row
    ct.store(temp_buffer, index=(row_idx, 0), tile=ct.astype(c2, temp_buffer.dtype))

    weighted_gradient_product = c1 * weight_vector
    weighted_gradient_sum = ct.sum(weighted_gradient_product, axis=1, keepdims=True)

    inv_std_cubed = inv_std_row * inv_std_row * inv_std_row
    norm_factor = ct.full((1, 1), N * 1.0, dtype=ct.float32)
    normalization_correction_coeff = input_row * inv_std_cubed / norm_factor
    normalization_correction = normalization_correction_coeff * weighted_gradient_sum

    scaled_gradient = gradient_row * weight_vector * inv_std_row
    input_gradient_row = scaled_gradient - normalization_correction
    input_gradient_row = ct.astype(input_gradient_row, dx.dtype)
    ct.store(dx, index=(row_idx, 0), tile=input_gradient_row)


@ct.kernel
def rms_norm_kernel_gather(
    x,
    w,
    out,
    Rstd,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Standard RMSNorm kernel with gather/scatter."""
    row = ct.bid(0)
    _rms = ct.full((TILE_SIZE,), 0.0, dtype=ct.float32)
    num_tiles = ct.cdiv(N, TILE_SIZE)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    for j in range(0, num_tiles):
        offs = j * TILE_SIZE + offsets
        xj = ct.gather(x, (row, offs), latency=1)
        xj = ct.astype(xj, ct.float32)
        _rms += xj * xj

    rms = ct.rsqrt(ct.sum(_rms, axis=0, keepdims=False) / N + eps)
    ct.scatter(Rstd, row, rms)

    for j in range(0, num_tiles):
        offs = j * TILE_SIZE + offsets
        wj = ct.gather(w, offs, latency=1)
        wj = ct.astype(wj, ct.float32)
        xj = ct.gather(x, (row, offs), latency=1)
        xj = ct.astype(xj, ct.float32)
        yj = xj * rms * wj
        yj = ct.astype(yj, x.dtype)
        ct.scatter(out, (row, offs), yj, latency=1)


# ============================================================================
# Autotune Configurations
# ============================================================================

def _rms_norm_forward_autotune_configs(N):
    """Iterator of autotune configurations for RMSNorm forward kernel."""
    gpu_capability = torch.cuda.get_device_capability()
    
    # Compute base tile size from N
    MAX_FUSED_SIZE = 4096 // 2  # Assume float16/bfloat16
    base_tile = min(MAX_FUSED_SIZE, next_power_of_2(N))
    
    # Try different tile sizes around the base
    tile_sizes = set()
    for mult in [0.5, 1, 2]:
        ts = int(base_tile * mult)
        if 64 <= ts <= 16384 and ts == next_power_of_2(ts):
            tile_sizes.add(ts)
    
    # Add standard power-of-2 tile sizes
    for ts in [256, 512, 1024, 2048, 4096]:
        if ts >= N or ts == next_power_of_2(N):
            tile_sizes.add(ts)
    
    if gpu_capability in [(12, 0), (12, 1)]:  # Blackwell
        for tile_size in sorted(tile_sizes):
            for occupancy in [1, 2, 4]:
                for num_ctas in [1, 2]:
                    yield SimpleNamespace(
                        TILE_SIZE=tile_size,
                        num_ctas=num_ctas,
                        occupancy=occupancy,
                    )
    else:
        for tile_size in sorted(tile_sizes):
            yield SimpleNamespace(
                TILE_SIZE=tile_size,
                num_ctas=1,
                occupancy=1,
            )


def _rms_norm_backward_autotune_configs(N):
    """Iterator of autotune configurations for RMSNorm backward kernel."""
    gpu_capability = torch.cuda.get_device_capability()
    
    tile_size = next_power_of_2(N)
    
    # Keep occupancy=2 fixed (known-good from original implementation)
    # Only tune num_ctas on Blackwell
    if gpu_capability in [(12, 0), (12, 1)]:  # Blackwell
        for num_ctas in [1, 2]:
            yield SimpleNamespace(
                TILE_SIZE=tile_size,
                num_ctas=num_ctas,
                occupancy=2,  # Fixed - known-good value
            )
    else:
        yield SimpleNamespace(
            TILE_SIZE=tile_size,
            num_ctas=1,
            occupancy=2,
        )


# ============================================================================
# Autotuned Launch Functions
# ============================================================================

def cutile_autotune_rms_norm_forward(stream, x, w, out, rstd, N, eps):
    """Autotuned RMSNorm forward pass."""
    M = x.shape[0]
    
    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (M,),
        kernel=rms_norm_kernel_gather,
        args_fn=lambda cfg: (x, w, out, rstd, N, eps, cfg.TILE_SIZE),
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        search_space=lambda: _rms_norm_forward_autotune_configs(N),
    )


def cutile_autotune_rms_norm_backward(stream, dx, dy, x, weight, rstd, temp_buffer, N):
    """Autotuned RMSNorm backward pass."""
    M = x.shape[0]
    
    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (M,),
        kernel=rms_norm_backward_kernel,
        args_fn=lambda cfg: (dx, dy, x, weight, rstd, temp_buffer, cfg.TILE_SIZE),
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        search_space=lambda: _rms_norm_backward_autotune_configs(N),
    )


# ============================================================================
# Main Functions
# ============================================================================

def rms_norm_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTile RMSNorm backward pass with autotuning."""
    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    rstd = rstd.contiguous()

    x_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])

    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    temp_buffer = torch.empty(x.shape, device=x.device, dtype=torch.float32)

    dx = dx.detach()
    dw = dw.detach()

    cutile_autotune_rms_norm_backward(
        torch.cuda.current_stream(),
        dx, dy, x, weight, rstd, temp_buffer, N
    )

    dw = temp_buffer[:, :N].to(torch.float32).sum(dim=0).to(weight.dtype)
    return dx.view(*x_shape), dw


class CuTileRMSNorm(torch.autograd.Function):
    """RMSNorm with CuTile forward and backward, with autotuning."""
    
    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        weight = weight.contiguous()
        x_arg = x.reshape(-1, x.shape[-1])
        
        y = torch.empty_like(x_arg)
        M, N = x_arg.shape
        
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        
        cutile_autotune_rms_norm_forward(
            torch.cuda.current_stream(),
            x_arg.detach(), weight.detach(), y.detach(), rstd, N, eps
        )
        
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return y.view(*x.shape)
    
    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        dx, dw = rms_norm_backward(x, dy, weight, rstd)
        return dx, dw, None


class TileRMSNorm(nn.Module):
    """Drop-in replacement RMSNorm using CuTile with autotuning."""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
    
    def forward(self, hidden_states, static_persistent=None):
        # static_persistent param kept for API compatibility but ignored
        return CuTileRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)
    
    def forward_torch(self, hidden_states):
        """PyTorch reference implementation for comparison"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    @staticmethod
    def compute_rstd_torch(x: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute rstd (reciprocal standard deviation) for RMSNorm using PyTorch."""
        x_2d = x.reshape(-1, x.shape[-1])
        x_fp32 = x_2d.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1)
        rstd = torch.rsqrt(variance + eps)
        return rstd

    @staticmethod
    def rms_norm_backward(
        x: torch.Tensor,
        dy: torch.Tensor,
        weight: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Only for testing purposes."""
        return rms_norm_backward(x, dy, weight, rstd)

    @staticmethod
    def rms_norm_backward_torch(
        x: torch.Tensor,
        dy: torch.Tensor,
        weight: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standalone RMSNorm backward pass using PyTorch reference implementation."""
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        dy = dy.reshape(-1, dy.shape[-1])
        M, N = x.shape

        rstd = rstd.view(M, 1)
        dw = ((x * dy) * rstd).sum(dim=0, dtype=torch.float32)
        x_norm = x * rstd
        dy_weighted = dy * weight
        c1 = (dy_weighted * x_norm).sum(dim=1, keepdim=True, dtype=torch.float32)
        dx = rstd * (dy_weighted - x_norm * c1 / N)

        dx = dx.view(x_shape).to(x.dtype)
        dw = dw.to(weight.dtype)

        return dx, dw
    
    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.variance_epsilon}"


# Alias for compatibility with user's original code
BastileRMSNorm = TileRMSNorm


@register_impl("rms_norm", backend="cutile")
def rms_norm(input, normalized_shape, weight, eps, bias=None, static_persistent=None, **kwargs):
    """
    Root mean square normalization implemented using CUDA Tile with autotuning.

    Args:
        input: Tensor of shape (M, N)
        normalized_shape: Normalization shape (for compatibility, not used)
        weight: Tensor of shape (N,)
        eps: Small constant added to variance calculation
        bias: Bias tensor of shape (N,), default is None (not supported)
        static_persistent: Ignored (kept for API compatibility)
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Normalized tensor with same shape as input
    """
    if bias is not None:
        raise NotImplementedError("Bias is not supported in CuTile RMSNorm")
    return CuTileRMSNorm.apply(input, weight, eps)


@register_impl("get_rms_norm_module", backend="cutile")
def get_rms_norm_module():
    return TileRMSNorm
