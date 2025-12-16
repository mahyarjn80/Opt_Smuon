"""
Distributed versions of advanced optimizers for multi-GPU training.

This module provides distributed implementations that work with PyTorch's DistributedDataParallel (DDP).
Each optimizer distributes parameter update computation across GPUs and synchronizes via all_reduce.

Pattern:
1. Each GPU processes a subset of parameters (based on rank)
2. Updates are collected in a flat buffer
3. all_reduce synchronizes updates across all GPUs
4. All GPUs apply the synchronized updates

Environment variables required:
- WORLD_SIZE: Total number of GPUs
- RANK: Current GPU rank (0 to WORLD_SIZE-1)
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist

from src.coupled_ns import compute_u_sigma_half_vt
from polar import polar


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    This is used by the Muon optimizer for whitening gradient updates.

    Args:
        G: Gradient matrix to orthogonalize
        steps: Number of Newton-Schulz iterations
        eps: Small constant for numerical stability

    Returns:
        Orthogonalized version of G
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonRemez(torch.optim.Optimizer):
    r"""Distributed MuonRemez Optimizer with Coupled Newton-Schulz for Matrix Square Root.

    Similar to Muon but uses coupled Newton-Schulz iteration to compute U·Σ^(1/2)·V^T
    instead of Newton-Schulz orthogonalization (which computes Σ^0).
    Distributed across GPUs for multi-GPU training.

    Update rule:
        v_t = β v_{t-1} + (1-β) g_t              (momentum)
        u_t = compute_u_sigma_half_vt(v_t)       (matrix square root via coupled NS)
        W ← (1 - η*λ) W - η u_t                  (update with weight decay)

    Arguments:
        params: parameters to optimize
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0.9)
        nesterov: whether to use Nesterov momentum (default: False)
        weight_decay: weight decay (L2 penalty) (default: 0.0)
        ns_steps: number of coupled Newton-Schulz iterations (default: 7)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        ns_steps: int = 7,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum value > 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super(MuonRemez, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single distributed optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']

            # Calculate total parameters and allocate flat buffer
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)

            curr_idx = 0
            for i, p in enumerate(group['params']):
                # Distribute parameters across GPUs
                if i % world_size == rank:
                    if p.grad is None:
                        curr_idx += p.numel()
                        continue

                    grad = p.grad.data
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['momentum_buffer'] = torch.zeros_like(grad)

                    buf = state['momentum_buffer']

                    # Update momentum buffer: v_t = β v_{t-1} + (1-β) g_t
                    buf.lerp_(grad, 1 - momentum)

                    # Get update (with Nesterov if enabled)
                    if nesterov:
                        update = grad.lerp(buf, momentum)
                        bias_correction = 1 / (1 - momentum ** (state['step'] + 2))
                    else:
                        update = buf.clone()
                        bias_correction = 1 / (1 - momentum ** (state['step'] + 1))

                    # Apply coupled Newton-Schulz: reshape to 2D, compute U·Σ^(1/2)·V^T, reshape back
                    if len(grad.shape) >= 2:
                        # Reshape to matrix (first dim x product of rest)
                        update_2d = update.view(len(update), -1)

                        # Apply coupled Newton-Schulz to compute U·Σ^(1/2)·V^T
                        update_sqrt_2d = compute_u_sigma_half_vt(
                            update_2d, steps=ns_steps, variant='higham'
                        )

                        # Reshape back
                        update_sqrt = update_sqrt_2d.view(update.shape)

                        # Scale by aspect ratio (from original Muon paper)
                        aspect_ratio = max(1, grad.size(-2) / grad.size(-1))
                        update_sqrt *= aspect_ratio ** 0.5
                    else:
                        # For 1D parameters, just use the update as-is
                        update_sqrt = update

                    # Store update in flat buffer
                    updates_flat[curr_idx:curr_idx+p.numel()] = update_sqrt.flatten().to(torch.bfloat16)
                    state['step'] += 1

                curr_idx += p.numel()

            # Synchronize updates across all GPUs
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # Apply updates to all parameters
            curr_idx = 0
            for p in group['params']:
                if p.grad is not None:
                    update = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)

                    # Apply weight decay (decoupled)
                    p.data.mul_(1 - lr * weight_decay)

                    # Apply update
                    p.data.add_(update, alpha=-lr)

                curr_idx += p.numel()

        return loss


class MuonResidual(torch.optim.Optimizer):
    r"""Distributed Muon optimizer with gradient residual regularization.

    This optimizer combines orthogonalization (Muon) with a residual term that
    captures the difference between the gradient and its square root preconditioner.
    Distributed across GPUs for multi-GPU training.

    Update rule:
        v_t = β v_{t-1} + (1-β) g_t                              (momentum)
        u_orth = NS_zeroth_power(v_t)                            (orthogonalize)
        u_sqrt = NS_sqrt(v_t)                                    (square root)
        u_t = u_orth + γ * (v_t - u_sqrt)                        (residual)
        W ← (1 - η*wd) W - η u_t                                 (update)

    Arguments:
        params: parameters to optimize
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0.9)
        nesterov: whether to use Nesterov momentum (default: True)
        weight_decay: weight decay (L2 penalty) (default: 0.0)
        ns_steps: Newton-Schulz iterations (default: 5)
        gamma: residual scaling factor (default: 0.1)
            - gamma=0.0: Pure Muon (orthogonalization only)
            - gamma>0.0: Adds residual term (gradient - sqrt)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        gamma: float = 0.1,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum value > 0")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            gamma=gamma,
        )
        super(MuonResidual, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single distributed optimization step with residual-regularized update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            gamma = group['gamma']

            # Calculate total parameters and allocate flat buffer
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)

            curr_idx = 0
            for i, p in enumerate(group['params']):
                # Distribute parameters across GPUs
                if i % world_size == rank:
                    if p.grad is None:
                        curr_idx += p.numel()
                        continue

                    grad = p.grad.data
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['momentum_buffer'] = torch.zeros_like(grad)

                    buf = state['momentum_buffer']

                    # Update momentum buffer: v_t = β v_{t-1} + (1-β) g_t
                    buf.lerp_(grad, 1 - momentum)

                    # Get update (with Nesterov if enabled)
                    if nesterov:
                        update = grad.lerp(buf, momentum)
                        bias_correction = 1.0 / (1 - momentum ** (state['step'] + 2))
                    else:
                        update = buf.clone()
                        bias_correction = 1.0 / (1 - momentum ** (state['step'] + 1))

                    # Apply matrix preconditioning with residual
                    if len(grad.shape) >= 2:
                        # Reshape to matrix (first dim x product of rest)
                        update_2d = update.view(len(update), -1)

                        # Compute orthogonalization (Muon)
                        muon_update_2d = zeropower_via_newtonschulz5(update_2d, steps=ns_steps)

                        if gamma > 0.0:
                            # Compute square root (MuonRemez)
                            sqrt_update_2d = compute_u_sigma_half_vt(
                                update_2d, steps=ns_steps, variant='muon_aggressive'
                            )

                            # Residual: u_orth + γ * (v_t - u_sqrt)
                            update_precond_2d = muon_update_2d + gamma * (update_2d - sqrt_update_2d)
                        else:
                            # Pure Muon when gamma=0
                            update_precond_2d = muon_update_2d

                        # Reshape back
                        update_precond = update_precond_2d.view(update.shape)

                        # Scale by aspect ratio (from original Muon paper)
                        aspect_ratio = max(1, grad.size(-2) / grad.size(-1))
                        update_precond *= aspect_ratio ** 0.5
                    else:
                        # For 1D parameters, just use the update as-is
                        update_precond = update

                    # Store update in flat buffer
                    updates_flat[curr_idx:curr_idx+p.numel()] = update_precond.flatten().to(torch.bfloat16)
                    state['step'] += 1

                curr_idx += p.numel()

            # Synchronize updates across all GPUs
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # Apply updates to all parameters
            curr_idx = 0
            for p in group['params']:
                if p.grad is not None:
                    update = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)

                    # Apply weight decay (decoupled)
                    p.data.mul_(1 - lr * weight_decay)

                    # Apply update
                    p.data.add_(update, alpha=-lr)

                curr_idx += p.numel()

        return loss


class MuonSoftCoupled(torch.optim.Optimizer):
    r"""Distributed Muon optimizer with soft-coupled nuclear norm scaling.

    This optimizer extends Muon by scaling the orthogonalized update with a
    factor based on the nuclear norm and effective rank of the momentum buffer.
    Distributed across GPUs for multi-GPU training.

    Update rule:
        v_t = β v_{t-1} + (1-β) g_t                          (momentum)
        u_orth = NS_zeroth_power(v_t)                        (orthogonalize)
        ||v_t||_nuc = trace(H) from polar decomposition     (nuclear norm)
        ||v_t||_F² = trace(v_t^T v_t)                       (Frobenius norm squared)
        r_eff = ||v_t||_nuc² / ||v_t||_F²                   (effective rank)
        scale = sqrt(||v_t||_nuc / r_eff)                   (scaling factor)
        u_t = scale * u_orth                                 (scaled update)
        W ← (1 - η*wd) W - η u_t                            (update)

    Arguments:
        params: parameters to optimize
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0.9)
        nesterov: whether to use Nesterov momentum (default: True)
        weight_decay: weight decay (L2 penalty) (default: 0.0)
        ns_steps: Newton-Schulz iterations (default: 5)
        eps: epsilon for numerical stability in polar decomposition (default: 1e-7)
        nuclear_norm_update_freq: how often to recompute nuclear norm scaling (default: 1)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        eps: float = 1e-7,
        nuclear_norm_update_freq: int = 1,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum value > 0")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if nuclear_norm_update_freq < 1:
            raise ValueError(f"Invalid nuclear_norm_update_freq: {nuclear_norm_update_freq}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
            nuclear_norm_update_freq=nuclear_norm_update_freq,
        )
        super(MuonSoftCoupled, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single distributed optimization step with nuclear norm scaling."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            eps = group['eps']
            update_freq = group['nuclear_norm_update_freq']

            # Calculate total parameters and allocate flat buffer
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)

            curr_idx = 0
            for i, p in enumerate(group['params']):
                # Distribute parameters across GPUs
                if i % world_size == rank:
                    if p.grad is None:
                        curr_idx += p.numel()
                        continue

                    grad = p.grad.data
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['momentum_buffer'] = torch.zeros_like(grad)
                        state['cached_scale'] = None  # Cache for nuclear norm scaling

                    buf = state['momentum_buffer']

                    # Update momentum buffer: v_t = β v_{t-1} + (1-β) g_t
                    buf.lerp_(grad, 1 - momentum)

                    # Get update (with Nesterov if enabled)
                    if nesterov:
                        update = grad.lerp(buf, momentum)
                        bias_correction = 1.0 / (1 - momentum ** (state['step'] + 2))
                    else:
                        update = buf.clone()
                        bias_correction = 1.0 / (1 - momentum ** (state['step'] + 1))

                    # Apply nuclear norm scaling with orthogonalization
                    if len(grad.shape) >= 2:
                        # Reshape to matrix (first dim x product of rest)
                        update_2d = update.view(len(update), -1)

                        # Compute/update scaling factor based on frequency
                        should_update_scale = (state['step'] % update_freq == 0)

                        if should_update_scale or state['cached_scale'] is None:
                            # Compute nuclear norm via polar decomposition
                            _, h = polar(update_2d, method='qdwh', compute_hermitian=True, eps=eps)
                            nuc_norm = torch.trace(h)

                            # Compute Frobenius norm squared: ||G||_F² = trace(G^T G)
                            frob_sq = torch.trace(update_2d.T @ update_2d)

                            # Compute effective rank: r_eff = ||G||_nuc² / ||G||_F²
                            r_eff = (nuc_norm ** 2) / (frob_sq + eps)

                            # Compute scaling factor: sqrt(||G||_nuc / r_eff)
                            scale = torch.sqrt(nuc_norm / (r_eff + eps))

                            # Cache the scaling factor
                            state['cached_scale'] = scale.item()
                        else:
                            # Use cached scaling factor
                            scale = state['cached_scale']

                        # Compute orthogonalization (Muon)
                        update_orth_2d = zeropower_via_newtonschulz5(update_2d, steps=ns_steps)

                        # Scale the orthogonalized update
                        update_scaled_2d = scale * update_orth_2d

                        # Reshape back
                        update_scaled = update_scaled_2d.view(update.shape)

                        # Scale by aspect ratio (from original Muon paper)
                        aspect_ratio = max(1, grad.size(-2) / grad.size(-1))
                        update_scaled *= aspect_ratio ** 0.5
                    else:
                        # For 1D parameters, just use the update as-is
                        update_scaled = update

                    # Store update in flat buffer
                    updates_flat[curr_idx:curr_idx+p.numel()] = update_scaled.flatten().to(torch.bfloat16)
                    state['step'] += 1

                curr_idx += p.numel()

            # Synchronize updates across all GPUs
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # Apply updates to all parameters
            curr_idx = 0
            for p in group['params']:
                if p.grad is not None:
                    update = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)

                    # Apply weight decay (decoupled)
                    p.data.mul_(1 - lr * weight_decay)

                    # Apply update
                    p.data.add_(update, alpha=-lr)

                curr_idx += p.numel()

        return loss


class MuonFixedRank(torch.optim.Optimizer):
    r"""Distributed Muon optimizer with fixed-rank nuclear norm scaling.

    This optimizer extends Muon by scaling the orthogonalized update with a
    factor based on the nuclear norm and the maximum possible rank.
    Distributed across GPUs for multi-GPU training.

    Update rule:
        v_t = β v_{t-1} + (1-β) g_t                          (momentum)
        u_orth = NS_zeroth_power(v_t)                        (orthogonalize)
        ||v_t||_nuc = trace(H) from polar decomposition     (nuclear norm)
        r_max = min(matrix dimensions)                       (maximum rank)
        scale = sqrt(||v_t||_nuc) / r_max                   (scaling factor)
        u_t = scale * u_orth                                 (scaled update)
        W ← (1 - η*wd) W - η u_t                            (update)

    Arguments:
        params: parameters to optimize
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0.9)
        nesterov: whether to use Nesterov momentum (default: True)
        weight_decay: weight decay (L2 penalty) (default: 0.0)
        ns_steps: Newton-Schulz iterations (default: 5)
        eps: epsilon for numerical stability in polar decomposition (default: 1e-7)
        nuclear_norm_update_freq: how often to recompute nuclear norm scaling (default: 1)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        eps: float = 1e-7,
        nuclear_norm_update_freq: int = 1,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum value > 0")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if nuclear_norm_update_freq < 1:
            raise ValueError(f"Invalid nuclear_norm_update_freq: {nuclear_norm_update_freq}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
            nuclear_norm_update_freq=nuclear_norm_update_freq,
        )
        super(MuonFixedRank, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single distributed optimization step with fixed-rank nuclear norm scaling."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            eps = group['eps']
            update_freq = group['nuclear_norm_update_freq']

            # Calculate total parameters and allocate flat buffer
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)

            curr_idx = 0
            for i, p in enumerate(group['params']):
                # Distribute parameters across GPUs
                if i % world_size == rank:
                    if p.grad is None:
                        curr_idx += p.numel()
                        continue

                    grad = p.grad.data
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['momentum_buffer'] = torch.zeros_like(grad)
                        state['cached_scale'] = None  # Cache for nuclear norm scaling

                    buf = state['momentum_buffer']

                    # Update momentum buffer: v_t = β v_{t-1} + (1-β) g_t
                    buf.lerp_(grad, 1 - momentum)

                    # Get update (with Nesterov if enabled)
                    if nesterov:
                        update = grad.lerp(buf, momentum)
                        bias_correction = 1.0 / (1 - momentum ** (state['step'] + 2))
                    else:
                        update = buf.clone()
                        bias_correction = 1.0 / (1 - momentum ** (state['step'] + 1))

                    # Apply fixed-rank nuclear norm scaling with orthogonalization
                    if len(grad.shape) >= 2:
                        # Reshape to matrix (first dim x product of rest)
                        update_2d = update.view(len(update), -1)

                        # Compute/update scaling factor based on frequency
                        should_update_scale = (state['step'] % update_freq == 0)

                        if should_update_scale or state['cached_scale'] is None:
                            # Compute nuclear norm via polar decomposition
                            _, h = polar(update_2d, method='qdwh', compute_hermitian=True, eps=eps)
                            nuc_norm = torch.trace(h)

                            # Compute r_max = min(matrix dimensions)
                            r_max = min(update_2d.shape[0], update_2d.shape[1])

                            # Compute scaling factor: sqrt(||G||_nuc) / r_max
                            scale = torch.sqrt(nuc_norm + eps) / r_max

                            # Cache the scaling factor
                            state['cached_scale'] = scale.item()
                        else:
                            # Use cached scaling factor
                            scale = state['cached_scale']

                        # Compute orthogonalization (Muon)
                        update_orth_2d = zeropower_via_newtonschulz5(update_2d, steps=ns_steps)

                        # Scale the orthogonalized update
                        update_scaled_2d = scale * update_orth_2d

                        # Reshape back
                        update_scaled = update_scaled_2d.view(update.shape)

                        # Scale by aspect ratio (from original Muon paper)
                        aspect_ratio = max(1, grad.size(-2) / grad.size(-1))
                        update_scaled *= aspect_ratio ** 0.5
                    else:
                        # For 1D parameters, just use the update as-is
                        update_scaled = update

                    # Store update in flat buffer
                    updates_flat[curr_idx:curr_idx+p.numel()] = update_scaled.flatten().to(torch.bfloat16)
                    state['step'] += 1

                curr_idx += p.numel()

            # Synchronize updates across all GPUs
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # Apply updates to all parameters
            curr_idx = 0
            for p in group['params']:
                if p.grad is not None:
                    update = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)

                    # Apply weight decay (decoupled)
                    p.data.mul_(1 - lr * weight_decay)

                    # Apply update
                    p.data.add_(update, alpha=-lr)

                curr_idx += p.numel()

        return loss
