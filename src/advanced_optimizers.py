"""
Advanced optimizers (Shampoo, Muon, MuonRemez, MuonSoftCoupled, MuonFixedRank, Adam) implemented as PyTorch optimizers.

This module provides:
- Shampoo: Full-matrix preconditioning optimizer
- ShampooNS: Shampoo with Newton-Schulz orthogonalization
- Muon: Gradient orthogonalization optimizer
- MuonRemez: Muon variant using coupled Newton-Schulz for matrix square root (Σ^(1/2))
- MuonResidual: Muon with gradient residual regularization
- MuonSoftCoupled: Muon with nuclear norm scaling based on effective rank
- MuonFixedRank: Muon with nuclear norm scaling based on maximum rank
- HybridMuonRemez: Blends Muon and MuonRemez with configurable schedule
- Adam: Adaptive moment estimation optimizer (AdamW variant)
- Configuration classes and utilities for creating optimizers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Iterable, Optional, Union, List

import numpy as np
import torch
import torch.optim

from src.coupled_ns import compute_u_sigma_half_vt
from polar import polar


Array = Any



@torch.no_grad()
def _matrix_power(
    matrix: torch.Tensor,
    p: int,
    num_iters: int = 60,
    ridge_epsilon: float = 1e-6,
    error_tolerance: float = 1e-6,
):
    """
    Compute matrix^(-1/p) for symmetric PSD `matrix` using the coupled-Newton iteration.
    Runs on CPU in float32, returns in the input dtype.
    
    Args:
        matrix: Symmetric PSD matrix
        p: Positive integer power
        num_iters: Maximum number of iterations
        ridge_epsilon: Ridge regularization factor
        error_tolerance: Convergence tolerance
        
    Returns:
        matrix^(-1/p)
    """
    assert matrix.ndim == 2 and matrix.size(0) == matrix.size(1), "matrix must be square"
    assert p > 0 and isinstance(p, int), "p must be a positive integer"

    
    device_in, dtype_in = matrix.device, matrix.dtype
    A = matrix.to(torch.float32).cpu()

    n = A.size(0)
    I = torch.eye(n, dtype=torch.float32)

    
    def spectral_norm(m: torch.Tensor) -> float:
        # power iteration for symmetric PSD (largest eigenvalue)
        v = torch.randn(n, 1, dtype=torch.float32)
        v /= v.norm() + 1e-12
        last = 0.0
        for _ in range(100):
            v = m @ v
            nv = v.norm().item()
            if nv < 1e-30:
                return 0.0
            v /= nv
            
            lam = torch.dot(v.flatten(), (m @ v).flatten()).item()
            if abs(lam - last) <= 1e-6 * max(1.0, abs(last)):
                break
            last = lam
        return max(lam, 0.0)

    
    max_ev = spectral_norm(A)
    scaled_ridge = ridge_epsilon * max(max_ev, 1e-16)

    
    if n == 1:
        out = (A + scaled_ridge).pow(-1.0 / p)
        return out.to(device=device_in, dtype=dtype_in)

    
    Ad = A + scaled_ridge * I

    
    z = (1.0 + p) / max(2.0 * spectral_norm(Ad), 1e-16)

    
    alpha = -1.0 / p
    M = Ad * z                  # mat_m
    H = I * (z ** (1.0 / p))    # mat_h

    
    def max_abs(x: torch.Tensor) -> float:
        return x.abs().max().item()

    err = max_abs(M - I)
    i = 0
    run_step = True
    prev_err = err
    prev_H = H.clone()

    
    def mat_power(mat: torch.Tensor, k: int) -> torch.Tensor:
        assert k >= 1
        result = I.clone()
        base = mat
        e = k
        while e > 0:
            if (e & 1) == 1:
                result = result @ base
            e >>= 1
            if e:
                base = base @ base
        return result

    
    while (i < num_iters) and (err > error_tolerance) and run_step:
        M_i = (1.0 - alpha) * I + alpha * M
        M_next = mat_power(M_i, p) @ M
        H_next = H @ M_i

        new_err = max_abs(M_next - I)


        run_step = new_err < (prev_err * 1.2 + 1e-12)

        
        i += 1
        prev_H = H.clone()
        prev_err = err
        M, H, err = M_next, H_next, new_err

    
    H_final = H if run_step else prev_H

    return H_final.to(device=device_in)


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


#############################################
#           Shampoo Optimizer               #
#############################################

class Shampoo(torch.optim.Optimizer):
    r"""Implements Shampoo Optimizer Algorithm.

    Shampoo uses full-matrix preconditioning by maintaining Gram matrices of gradients
    along each dimension of the parameter tensor, then using their matrix roots as
    preconditioners.
    
    For a 2D parameter W, Shampoo maintains:
        L_t = ∑ g_t g_t^T  (left preconditioner)
        R_t = ∑ g_t^T g_t  (right preconditioner)
    
    And applies the preconditioned update:
        W ← W - η L_t^{-1/(order*multiplier)} g_t R_t^{-1/(order*multiplier)}

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-1)
        momentum (float): momentum factor (default: 0.0)
        weight_decay (float): weight decay (L2 penalty) (default: 0.0)
        epsilon (float): epsilon added to each mat_gbar_j for numerical stability
            (default: 1e-4)
        update_freq (int): update frequency to compute inverse (default: 1)
        order_multiplier (int): multiplier for matrix power, e.g., 
            power = -1/(order * order_multiplier) (default: 2)
        nesterov (bool): whether to use Nesterov momentum (default: True)

    Example:
        >>> optimizer = Shampoo(model.parameters(), lr=0.01, order_multiplier=1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        This is a MATRIX optimizer - it works best with 2D+ parameters (weight matrices).
        For 1D parameters (biases), it degenerates to standard gradient descent with momentum.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 1,
        order_multiplier: int = 2,
        nesterov: bool = True,
    ):
        if lr <= 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if epsilon < 0.0:
            raise ValueError(f'Invalid epsilon value: {epsilon}')
        if update_freq < 1:
            raise ValueError(f'Invalid update_freq value: {update_freq}')

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
            nesterov=nesterov,
        )
        self.order_multiplier = order_multiplier
        super(Shampoo, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state['momentum_buffer'] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # Precondition matrices (Gram matrices)
                        state[f'precond_{dim_id}'] = group['epsilon'] * torch.eye(
                            dim, device=grad.device, dtype=torch.float32
                        )
                        state[f'inv_precond_{dim_id}'] = torch.zeros(
                            dim, dim, device=grad.device, dtype=torch.float32
                        )

                # Update momentum buffer
                if momentum > 0:
                    state['momentum_buffer'].lerp_(grad, 1 - momentum)

                # Get update (with Nesterov if enabled)
                if momentum > 0:
                    if group['nesterov']:
                        update = grad.lerp(state['momentum_buffer'], momentum)
                        # Bias correction for Nesterov
                        #update = update * (1 / (1 - momentum ** (state['step'] + 2)))
                    else:
                        update = state['momentum_buffer']
                        # Bias correction
                        #update = update * (1 / (1 - momentum ** (state['step'] + 1)))
                else:
                    update = grad
                
                update32 = update.to(torch.float32)



                # Apply Shampoo preconditioning along each dimension
                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f'precond_{dim_id}']
                    inv_precond = state[f'inv_precond_{dim_id}']

                    # For Gram matrix computation: transpose grad (no in-place mutation)
                    grad_transposed = grad.transpose(0, dim_id).contiguous()
                    grad_flat = grad_transposed.view(dim, -1)
                    g32 = grad_flat.to(torch.float32)
                    g32_t = g32.t()

                    # Transpose update32 to bring dimension dim_id to front (in-place is fine here)
                    update32 = update32.transpose_(0, dim_id).contiguous()
                    transposed_size = update32.size()
                    update32 = update32.view(dim, -1)


                    #precond.lerp_(g32 @ g32_t, 1 - group['momentum'])
                    precond.add_(g32 @ g32_t)
                    #precond_corrected = precond.mul(1/(1-group['momentum']**(state['step']+1)))
                    
                    # Recompute matrix inverse periodically
                    if state['step'] % group['update_freq'] == 0:
                        power = order * self.order_multiplier
                        inv_precond.copy_(_matrix_power(precond, power))

                    # Apply preconditioning
                    if dim_id == order - 1:
                        # Last dimension: apply from left
                        update32 = update32.t() @ inv_precond
                        # Reshape back to original and convert to original dtype only at the end
                        update32 = update32.view(original_size).to(grad.dtype)
                    else:
                        # Intermediate dimensions: apply from right
                        update32 = inv_precond @ update32
                        # Reshape for next iteration but STAY in float32
                        update32 = update32.view(transposed_size)

                state['step'] += 1
                
                # Apply weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                # Apply update
                p.data.add_(update32.view(original_size), alpha=-group['lr'])

        return loss


#############################################
#             Muon Optimizer                #
#############################################

class Muon(torch.optim.Optimizer):
    r"""Implements Muon Optimizer Algorithm.
    
    Muon uses orthogonalization (via Newton-Schulz iteration) to whiten gradient updates.
    It's designed for training neural networks with improved conditioning.
    
    Update rule:
        v_t = β v_{t-1} + (1-β) g_t              (momentum)
        u_t = orthogonalize(v_t)                 (whiten via Newton-Schulz)
        W ← (1 - η*λ) W - η u_t                  (update with weight decay)
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        momentum (float): momentum factor (default: 0.0)
        nesterov (bool): whether to use Nesterov momentum (default: False)
        weight_decay (float): weight decay (L2 penalty) (default: 0.0)

    Example:
        >>> optimizer = Muon(model.parameters(), lr=0.001, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        This is a MATRIX optimizer - it works best with 2D+ parameters.
        The Newton-Schulz orthogonalization is applied to parameters reshaped as matrices.
    """
    
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = True,
        weight_decay: float = 0.0,
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
        )
        super(Muon, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
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
                    # Bias correction for Nesterov
                    bias_correction = 1 / (1 - momentum ** (state['step'] + 2))
                else:
                    update = buf.clone()
                    # Bias correction
                    bias_correction = 1 / (1 - momentum ** (state['step'] + 1))
                
                #update = update * bias_correction
                
                # Orthogonalize: reshape to 2D, apply Newton-Schulz, reshape back
                if len(grad.shape) >= 2:
                    # Reshape to matrix (first dim x product of rest)
                    update_2d = update.view(len(update), -1)
                    
                    # Apply Newton-Schulz orthogonalization
                    update_orth_2d = zeropower_via_newtonschulz5(update_2d)
                    
                    # Reshape back
                    update_orth = update_orth_2d.view(update.shape)
                    
                    # Scale by aspect ratio (from original Muon paper)
                    aspect_ratio = max(1, grad.size(-2) / grad.size(-1))
                    update_orth *= aspect_ratio ** 0.5
                else:
                    # For 1D parameters, just use the update as-is
                    update_orth = update
                
                state['step'] += 1
                
                # Apply weight decay (decoupled)
                p.data.mul_(1 - lr * weight_decay)
                
                # Apply update
                p.data.add_(update_orth, alpha=-lr)

        return loss


#############################################
#           MuonRemez Optimizer             #
#############################################

class MuonRemez(torch.optim.Optimizer):
    r"""Implements Muon Optimizer with Coupled Newton-Schulz for Matrix Square Root.

    Similar to Muon but uses coupled Newton-Schulz iteration to compute U·Σ^(1/2)·V^T
    instead of Newton-Schulz orthogonalization (which computes Σ^0).

    Update rule:
        v_t = β v_{t-1} + (1-β) g_t              (momentum)
        u_t = compute_u_sigma_half_vt(v_t)       (matrix square root via coupled NS)
        W ← (1 - η*λ) W - η u_t                  (update with weight decay)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        momentum (float): momentum factor (default: 0.0)
        nesterov (bool): whether to use Nesterov momentum (default: False)
        weight_decay (float): weight decay (L2 penalty) (default: 0.0)
        ns_steps (int): number of coupled Newton-Schulz iterations (default: 5)

    Example:
        >>> optimizer = MuonRemez(model.parameters(), lr=0.001, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        This is a MATRIX optimizer - it works best with 2D+ parameters.
        The coupled Newton-Schulz method is applied to parameters reshaped as matrices.
        This computes alpha=0.5 (square root) of singular values.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
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
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
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
                    # Bias correction for Nesterov
                    bias_correction = 1 / (1 - momentum ** (state['step'] + 2))
                else:
                    update = buf.clone()
                    # Bias correction
                    bias_correction = 1 / (1 - momentum ** (state['step'] + 1))

                #update = update * bias_correction

                # Apply coupled Newton-Schulz: reshape to 2D, compute U·Σ^(1/2)·V^T, reshape back
                if len(grad.shape) >= 2:
                    # Reshape to matrix (first dim x product of rest)
                    update_2d = update.view(len(update), -1)

                    # Apply coupled Newton-Schulz to compute U·Σ^(1/2)·V^T
                    # update_sqrt_2d = compute_u_sigma_half_vt(
                    #     update_2d, steps=ns_steps, variant='muon_aggressive'
                    # )
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

                state['step'] += 1

                # Apply weight decay (decoupled)
                p.data.mul_(1 - lr * weight_decay)

                # Apply update
                p.data.add_(update_sqrt, alpha=-lr)

        return loss


#############################################
#      Hybrid Muon-MuonRemez Optimizer      #
#############################################

class HybridMuonRemez(torch.optim.Optimizer):
    r"""Hybrid optimizer blending Muon (α=0, orthogonalization) and MuonRemez (α=0.5, sqrt).

    This optimizer blends between two matrix function preconditioners:
    - Muon: Uses Newton-Schulz for orthogonalization (zeroth power, Σ^0)
    - MuonRemez: Uses coupled Newton-Schulz for square root (Σ^0.5)

    Update rule:
        v_t = β v_{t-1} + (1-β) g_t                      (momentum)
        u_muon = NS_zeroth_power(v_t)                    (Muon: orthogonalize)
        u_remez = NS_sqrt(v_t)                           (MuonRemez: sqrt)
        u_t = λ(t) * u_muon + (1-λ(t)) * u_remez         (blend)
        W ← (1 - η*wd) W - η u_t                         (update)

    Arguments:
        params (iterable): parameters to optimize
        lr (float): learning rate (default: 1e-3)
        momentum (float): momentum factor (default: 0.9)
        nesterov (bool): whether to use Nesterov momentum (default: True)
        weight_decay (float): weight decay (L2 penalty) (default: 0.0)
        ns_steps (int): Newton-Schulz iterations (default: 5)
        blend_schedule_fn (callable): function(step) -> lambda in [0,1] where
            lambda=1.0 means 100% Muon (orthogonalization)
            lambda=0.0 means 100% MuonRemez (square root)
            Default: constant 0.5 (50% blend)

    Example:
        >>> # Start with 100% Muon, decay to 100% MuonRemez over 500 steps
        >>> optimizer = HybridMuonRemez(
        ...     model.parameters(),
        ...     lr=0.0005,
        ...     momentum=0.9,
        ...     blend_schedule_fn=lambda s: max(0.0, 1.0 - s/500)
        ... )

    Note:
        - Memory efficient: single momentum buffer, no duplication
        - Both methods share same momentum and apply same scaling
        - Only difference is the matrix preconditioning function
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        blend_schedule_fn=None,
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
        self.blend_schedule_fn = blend_schedule_fn or (lambda step: 0.5)
        super(HybridMuonRemez, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with blended Muon-MuonRemez update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
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
                    # Bias correction for Nesterov
                    bias_correction = 1 / (1 - momentum ** (state['step'] + 2))
                else:
                    update = buf.clone()
                    # Bias correction
                    bias_correction = 1 / (1 - momentum ** (state['step'] + 1))

                # Get blend coefficient for this step
                blend_coeff = self.blend_schedule_fn(state['step'])
                blend_coeff = max(0.0, min(1.0, blend_coeff))  # Clamp to [0, 1]

                # Apply matrix preconditioning (blend Muon and MuonRemez)
                if len(grad.shape) >= 2:
                    # Reshape to matrix (first dim x product of rest)
                    update_2d = update.view(len(update), -1)


                    if blend_coeff == 0.0:
                        update_precond_2d = zeropower_via_newtonschulz5(update_2d, steps=ns_steps)
                        # update_precond_2d = compute_u_sigma_half_vt(
                        #     update_2d, steps=ns_steps, variant='muon_aggressive'
                        # )
                    else:
                        # Blend both
                        muon_update_2d = zeropower_via_newtonschulz5(update_2d, steps=ns_steps)
                        remez_update_2d = compute_u_sigma_half_vt(
                            update_2d, steps=ns_steps, variant='muon_aggressive'
                        )
                        update_precond_2d =  (1)*muon_update_2d + (blend_coeff) * (remez_update_2d)

                    # Reshape back
                    update_precond = update_precond_2d.view(update.shape)

                    # Scale by aspect ratio (from original Muon paper)
                    aspect_ratio = max(1, grad.size(-2) / grad.size(-1))
                    update_precond *= aspect_ratio ** 0.5
                else:
                    # For 1D parameters, just use the update as-is
                    update_precond = update

                state['step'] += 1

                # Apply weight decay (decoupled)
                p.data.mul_(1 - lr * weight_decay)

                # Apply update
                p.data.add_(update_precond, alpha=-lr)

        return loss


#############################################
#        Muon Residual Optimizer            #
#############################################

class MuonResidual(torch.optim.Optimizer):
    r"""Muon optimizer with gradient residual regularization.

    This optimizer combines orthogonalization (Muon) with a residual term that
    captures the difference between the gradient and its square root preconditioner.

    Update rule:
        v_t = β v_{t-1} + (1-β) g_t                              (momentum)
        u_orth = NS_zeroth_power(v_t)                            (orthogonalize)
        u_sqrt = NS_sqrt(v_t)                                    (square root)
        u_t = u_orth + γ * (v_t - u_sqrt)                        (residual)
        W ← (1 - η*wd) W - η u_t                                 (update)

    The key difference from HybridMuonRemez is:
    - Instead of blending: λ*u_orth + (1-λ)*u_sqrt
    - We use residual: u_orth + γ*(v_t - u_sqrt)

    This allows the optimizer to maintain the orthogonalization benefits while
    adding a scaled residual that captures information lost in the square root operation.

    Arguments:
        params (iterable): parameters to optimize
        lr (float): learning rate (default: 1e-3)
        momentum (float): momentum factor (default: 0.9)
        nesterov (bool): whether to use Nesterov momentum (default: True)
        weight_decay (float): weight decay (L2 penalty) (default: 0.0)
        ns_steps (int): Newton-Schulz iterations (default: 5)
        gamma (float): residual scaling factor (default: 0.1)
            - gamma=0.0: Pure Muon (orthogonalization only)
            - gamma>0.0: Adds residual term (gradient - sqrt)
            - Typical range: [0.0, 0.5]

    Example:
        >>> optimizer = MuonResidual(
        ...     model.parameters(),
        ...     lr=0.0005,
        ...     momentum=0.9,
        ...     gamma=0.1
        ... )
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        - Memory efficient: computes both orthogonalization and sqrt but no blending
        - The residual term (v_t - u_sqrt) provides gradient information
        - Works best with 2D+ parameters (weight matrices)
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
        """Performs a single optimization step with residual-regularized update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
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
                    # Bias correction for Nesterov
                    bias_correction = 1.0 / (1 - momentum ** (state['step'] + 2))
                else:
                    update = buf.clone()
                    # Bias correction
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

                state['step'] += 1

                # Apply weight decay (decoupled)
                p.data.mul_(1 - lr * weight_decay)

                # Apply update
                p.data.add_(update_precond, alpha=-lr)

        return loss


#############################################
#        MuonSoftCoupled Optimizer          #
#############################################

class MuonSoftCoupled(torch.optim.Optimizer):
    r"""Muon optimizer with soft-coupled nuclear norm scaling.

    This optimizer extends Muon by scaling the orthogonalized update with a
    factor based on the nuclear norm and effective rank of the momentum buffer.

    Update rule:
        v_t = β v_{t-1} + (1-β) g_t                          (momentum)
        u_orth = NS_zeroth_power(v_t)                        (orthogonalize)
        ||v_t||_nuc = trace(H) from polar decomposition     (nuclear norm)
        ||v_t||_F² = trace(v_t^T v_t)                       (Frobenius norm squared)
        r_eff = ||v_t||_nuc² / ||v_t||_F²                   (effective rank)
        scale = sqrt(||v_t||_nuc / r_eff)                   (scaling factor)
        u_t = scale * u_orth                                 (scaled update)
        W ← (1 - η*wd) W - η u_t                            (update)

    The scaling factor √(||G||_nuc)/√(r_eff) adaptively adjusts based on the
    effective rank of the gradient, providing stronger updates when the gradient
    has lower effective rank.

    Arguments:
        params (iterable): parameters to optimize
        lr (float): learning rate (default: 1e-3)
        momentum (float): momentum factor (default: 0.9)
        nesterov (bool): whether to use Nesterov momentum (default: True)
        weight_decay (float): weight decay (L2 penalty) (default: 0.0)
        ns_steps (int): Newton-Schulz iterations (default: 5)
        eps (float): epsilon for numerical stability in polar decomposition (default: 1e-7)
        nuclear_norm_update_freq (int): how often to recompute nuclear norm scaling (default: 1)
            Set to >1 to cache scaling factors and reduce computational cost

    Example:
        >>> optimizer = MuonSoftCoupled(
        ...     model.parameters(),
        ...     lr=0.0005,
        ...     momentum=0.9,
        ...     nuclear_norm_update_freq=10  # Update scaling every 10 steps
        ... )
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        - Uses QDWH polar decomposition to compute nuclear norm
        - Works best with 2D+ parameters (weight matrices)
        - Automatically handles aspect ratio scaling
        - For large models, set nuclear_norm_update_freq > 1 to improve speed
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
        """Performs a single optimization step with nuclear norm scaling."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            eps = group['eps']
            update_freq = group['nuclear_norm_update_freq']

            for p in group['params']:
                if p.grad is None:
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
                    # Bias correction for Nesterov
                    bias_correction = 1.0 / (1 - momentum ** (state['step'] + 2))
                else:
                    update = buf.clone()
                    # Bias correction
                    bias_correction = 1.0 / (1 - momentum ** (state['step'] + 1))

                # Apply nuclear norm scaling with orthogonalization
                if len(grad.shape) >= 2:
                    # Reshape to matrix (first dim x product of rest)
                    update_2d = update.view(len(update), -1)

                    # Compute/update scaling factor based on frequency
                    should_update_scale = (state['step'] % update_freq == 0)

                    if should_update_scale or state['cached_scale'] is None:
                        # Compute nuclear norm via polar decomposition
                        # polar returns (unitary, hermitian) when compute_hermitian=True
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

                state['step'] += 1

                # Apply weight decay (decoupled)
                p.data.mul_(1 - lr * weight_decay)

                # Apply update
                p.data.add_(update_scaled, alpha=-lr)

        return loss


#############################################
#        MuonFixedRank Optimizer            #
#############################################

class MuonFixedRank(torch.optim.Optimizer):
    r"""Muon optimizer with fixed-rank nuclear norm scaling.

    This optimizer extends Muon by scaling the orthogonalized update with a
    factor based on the nuclear norm and the maximum possible rank.

    Update rule:
        v_t = β v_{t-1} + (1-β) g_t                          (momentum)
        u_orth = NS_zeroth_power(v_t)                        (orthogonalize)
        ||v_t||_nuc = trace(H) from polar decomposition     (nuclear norm)
        r_max = min(matrix dimensions)                       (maximum rank)
        scale = sqrt(||v_t||_nuc) / r_max                   (scaling factor)
        u_t = scale * u_orth                                 (scaled update)
        W ← (1 - η*wd) W - η u_t                            (update)

    The scaling factor √(||G||_nuc)/r_max provides a normalized scaling based
    on the nuclear norm relative to the maximum possible rank.

    Arguments:
        params (iterable): parameters to optimize
        lr (float): learning rate (default: 1e-3)
        momentum (float): momentum factor (default: 0.9)
        nesterov (bool): whether to use Nesterov momentum (default: True)
        weight_decay (float): weight decay (L2 penalty) (default: 0.0)
        ns_steps (int): Newton-Schulz iterations (default: 5)
        eps (float): epsilon for numerical stability in polar decomposition (default: 1e-7)
        nuclear_norm_update_freq (int): how often to recompute nuclear norm scaling (default: 1)
            Set to >1 to cache scaling factors and reduce computational cost

    Example:
        >>> optimizer = MuonFixedRank(
        ...     model.parameters(),
        ...     lr=0.0005,
        ...     momentum=0.9,
        ...     nuclear_norm_update_freq=10  # Update scaling every 10 steps
        ... )
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        - Uses QDWH polar decomposition to compute nuclear norm
        - Works best with 2D+ parameters (weight matrices)
        - Automatically handles aspect ratio scaling
        - For large models, set nuclear_norm_update_freq > 1 to improve speed
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
        """Performs a single optimization step with fixed-rank nuclear norm scaling."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            eps = group['eps']
            update_freq = group['nuclear_norm_update_freq']

            for p in group['params']:
                if p.grad is None:
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
                    # Bias correction for Nesterov
                    bias_correction = 1.0 / (1 - momentum ** (state['step'] + 2))
                else:
                    update = buf.clone()
                    # Bias correction
                    bias_correction = 1.0 / (1 - momentum ** (state['step'] + 1))

                # Apply fixed-rank nuclear norm scaling with orthogonalization
                if len(grad.shape) >= 2:
                    # Reshape to matrix (first dim x product of rest)
                    update_2d = update.view(len(update), -1)

                    # Compute/update scaling factor based on frequency
                    should_update_scale = (state['step'] % update_freq == 0)

                    if should_update_scale or state['cached_scale'] is None:
                        # Compute nuclear norm via polar decomposition
                        # polar returns (unitary, hermitian) when compute_hermitian=True
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

                state['step'] += 1

                # Apply weight decay (decoupled)
                p.data.mul_(1 - lr * weight_decay)

                # Apply update
                p.data.add_(update_scaled, alpha=-lr)

        return loss


#############################################
#          ShampooNS Optimizer              #
#############################################

class ShampooNS(torch.optim.Optimizer):
    r"""Implements Shampoo with Newton-Schulz Orthogonalization.

    This optimizer combines:
    - Shampoo's full-matrix preconditioning along each dimension
    - Newton-Schulz orthogonalization applied after preconditioning

    The algorithm first applies Shampoo's preconditioners to the momentum update,
    then applies Newton-Schulz orthogonalization to the preconditioned result.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-1)
        momentum (float): momentum factor (default: 0.0)
        weight_decay (float): weight decay (L2 penalty) (default: 0.0)
        epsilon (float): epsilon added to preconditioners for numerical stability
            (default: 1e-4)
        update_freq (int): update frequency to compute inverse (default: 1)
        order_multiplier (int): multiplier for matrix power (default: 2)
        nesterov (bool): whether to use Nesterov momentum (default: True)
        ns_steps (int): number of Newton-Schulz iterations (default: 5)

    Example:
        >>> optimizer = ShampooNS(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        This combines Shampoo's preconditioning with Muon's orthogonalization,
        potentially providing benefits of both approaches.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 1,
        order_multiplier: int = 2,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        if lr <= 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if epsilon < 0.0:
            raise ValueError(f'Invalid epsilon value: {epsilon}')
        if update_freq < 1:
            raise ValueError(f'Invalid update_freq value: {update_freq}')
        if ns_steps < 1:
            raise ValueError(f'Invalid ns_steps value: {ns_steps}')

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        self.order_multiplier = order_multiplier
        super(ShampooNS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group['momentum']
                weight_decay = group['weight_decay']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state['momentum_buffer'] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # Precondition matrices (Gram matrices)
                        state[f'precond_{dim_id}'] = group['epsilon'] * torch.eye(
                            dim, device=grad.device, dtype=torch.float32
                        )
                        state[f'inv_precond_{dim_id}'] = torch.zeros(
                            dim, dim, device=grad.device, dtype=torch.float32
                        )

                # Update momentum buffer
                if momentum > 0:
                    state['momentum_buffer'].lerp_(grad, 1 - momentum)

                # Get update (with Nesterov if enabled)
                if momentum > 0:
                    if group['nesterov']:
                        update = grad.lerp(state['momentum_buffer'], momentum)
                    else:
                        update = state['momentum_buffer']
                else:
                    update = grad

                update32 = update.to(torch.float32)

                # Apply Shampoo preconditioning along each dimension
                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f'precond_{dim_id}']
                    inv_precond = state[f'inv_precond_{dim_id}']

                    # For Gram matrix computation: transpose grad (no in-place mutation)
                    grad_transposed = grad.transpose(0, dim_id).contiguous()
                    grad_flat = grad_transposed.view(dim, -1)
                    g32 = grad_flat.to(torch.float32)
                    g32_t = g32.t()

                    # Transpose update32 to bring dimension dim_id to front (in-place is fine here)
                    update32 = update32.transpose_(0, dim_id).contiguous()
                    transposed_size = update32.size()
                    update32 = update32.view(dim, -1)

                    precond.add_(g32 @ g32_t)

                    # Recompute matrix inverse periodically
                    if state['step'] % group['update_freq'] == 0:
                        power = order * self.order_multiplier
                        inv_precond.copy_(_matrix_power(precond, power))

                    # Apply preconditioning
                    if dim_id == order - 1:
                        # Last dimension: apply from left
                        update32 = update32.t() @ inv_precond
                        # Reshape back to original but STAY in float32 for NS
                        update32 = update32.view(original_size)
                    else:
                        # Intermediate dimensions: apply from right
                        update32 = inv_precond @ update32
                        # Reshape for next iteration but STAY in float32
                        update32 = update32.view(transposed_size)

                # Now apply Newton-Schulz orthogonalization on the preconditioned update
                if len(grad.shape) >= 2:
                    # Reshape to 2D matrix
                    update_2d = update32.view(update32.size(0), -1)

                    # Apply Newton-Schulz orthogonalization
                    update_orth_2d = zeropower_via_newtonschulz5(update_2d, steps=group['ns_steps'])

                    # Reshape back
                    update_orth = update_orth_2d.view(original_size)

                    # Scale by aspect ratio (from Muon)
                    aspect_ratio = max(1, grad.size(-2) / grad.size(-1))
                    update_orth *= aspect_ratio ** 0.5

                    # Convert back to original dtype
                    final_update = update_orth.to(grad.dtype)
                else:
                    # For 1D parameters, just use the preconditioned update
                    final_update = update32.to(grad.dtype)

                state['step'] += 1

                # Apply weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                # Apply update
                p.data.add_(final_update, alpha=-group['lr'])

        return loss


#############################################
#             Adam Optimizer                #
#############################################

class Adam(torch.optim.Optimizer):
    r"""Implements Adam Optimizer Algorithm (AdamW variant).
    
    Adam maintains exponential moving averages of both the gradient and its square:
        m_t = β₁ m_{t-1} + (1-β₁) g_t        (first moment)
        v_t = β₂ v_{t-1} + (1-β₂) g_t²       (second moment)
        m̂_t = m_t / (1 - β₁^t)               (bias correction)
        v̂_t = v_t / (1 - β₂^t)               (bias correction)
        w_{t+1} = (1 - η*λ) w_t - η m̂_t / (√v̂_t + ε)
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        betas (Tuple[float, float]): coefficients for computing running averages
            of gradient and its square (default: (0.9, 0.999))
        eps (float): term added to denominator for numerical stability 
            (default: 1e-8)
        weight_decay (float): weight decay coefficient (default: 0.0)

    Example:
        >>> optimizer = Adam(model.parameters(), lr=0.001)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        This is a VECTOR optimizer - it maintains per-parameter adaptive learning rates
        but doesn't use matrix structure like Shampoo or Muon.
    """
    
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(Adam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)       # First moment
                    state['exp_avg_sq'] = torch.zeros_like(p.data)    # Second moment

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Update first moment: m_t = β₁ m_{t-1} + (1-β₁) g_t
                exp_avg.lerp_(grad, 1 - beta1)
                
                # Update second moment: v_t = β₂ v_{t-1} + (1-β₂) g_t²
                exp_avg_sq.lerp_(grad.square(), 1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                # Compute update
                denom = exp_avg_sq_hat.sqrt().add_(eps)
                update = exp_avg_hat / denom
                
                # Apply weight decay (decoupled, AdamW style)
                p.data.mul_(1 - lr * weight_decay)
                
                # Apply update
                p.data.add_(update, alpha=-lr)

        return loss


#############################################
#       Optimizer Configuration Classes     #
#############################################

@dataclass
class MuonConfig:
    """Configuration for Muon optimizer.

    Attributes:
        lr: Learning rate
        momentum: Momentum factor for gradient smoothing
    """
    lr: float = 0.0005
    momentum: float = 0.9

    def __str__(self):
        return f"Muon_lr{self.lr}_mom{self.momentum}"


@dataclass
class MuonRemezConfig:
    """Configuration for MuonRemez optimizer.

    Attributes:
        lr: Learning rate
        momentum: Momentum factor for gradient smoothing
    """
    lr: float = 0.0005
    momentum: float = 0.9

    def __str__(self):
        return f"MuonRemez_lr{self.lr}_mom{self.momentum}"


@dataclass
class MuonResidualConfig:
    """Configuration for MuonResidual optimizer.

    Attributes:
        lr: Learning rate
        momentum: Momentum factor for gradient smoothing
        gamma: Residual scaling factor (controls contribution of gradient residual term)
    """
    lr: float = 0.0005
    momentum: float = 0.9
    gamma: float = 0.1

    def __str__(self):
        return f"MuonResidual_lr{self.lr}_mom{self.momentum}_gamma{self.gamma}"


@dataclass
class ShampooConfig:
    """Configuration for Shampoo optimizer.

    Attributes:
        lr: Learning rate
        momentum: Momentum factor for gradient smoothing
        order_multiplier: Multiplier for matrix power (power = -1/(order*multiplier))
    """
    lr: float = 0.0005
    momentum: float = 0.9
    order_multiplier: int = 2

    def __str__(self):
        return f"Shampoo_lr{self.lr}_mom{self.momentum}_order{self.order_multiplier}"


@dataclass
class ShampooNSConfig:
    """Configuration for ShampooNS optimizer.

    Attributes:
        lr: Learning rate
        momentum: Momentum factor for gradient smoothing
        order_multiplier: Multiplier for matrix power (power = -1/(order*multiplier))
        ns_steps: Number of Newton-Schulz iterations
    """
    lr: float = 0.0005
    momentum: float = 0.9
    order_multiplier: int = 2
    ns_steps: int = 5

    def __str__(self):
        return f"ShampooNS_lr{self.lr}_mom{self.momentum}_order{self.order_multiplier}_ns{self.ns_steps}"


@dataclass
class AdamConfig:
    """Configuration for Adam optimizer.

    Attributes:
        lr: Learning rate
        beta1: Exponential decay rate for first moment estimates
        beta2: Exponential decay rate for second moment estimates
    """
    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95

    def __str__(self):
        return f"Adam_lr{self.lr}_b1{self.beta1}_b2{self.beta2}"


@dataclass
class MuonSoftCoupledConfig:
    """Configuration for MuonSoftCoupled optimizer.

    Attributes:
        lr: Learning rate
        momentum: Momentum factor for gradient smoothing
    """
    lr: float = 0.0005
    momentum: float = 0.9

    def __str__(self):
        return f"MuonSoftCoupled_lr{self.lr}_mom{self.momentum}"


@dataclass
class MuonFixedRankConfig:
    """Configuration for MuonFixedRank optimizer.

    Attributes:
        lr: Learning rate
        momentum: Momentum factor for gradient smoothing
    """
    lr: float = 0.0005
    momentum: float = 0.9

    def __str__(self):
        return f"MuonFixedRank_lr{self.lr}_mom{self.momentum}"


@dataclass
class HybridMuonRemezConfig:
    """Configuration for HybridMuonRemez optimizer.

    Attributes:
        lr: Learning rate
        momentum: Momentum factor for gradient smoothing
        blend_start: Initial blend coefficient (1.0 = 100% Muon, 0.0 = 100% MuonRemez)
        blend_end: Final blend coefficient after decay
        blend_decay_fraction: Fraction of total training steps over which to decay
            (e.g., 0.5 means decay over first 50% of training)
        blend_schedule: Schedule type: 'linear', 'cosine', 'polynomial', 'exponential', 'constant'
        blend_power: Power for polynomial schedule (only used if blend_schedule='polynomial')
        blend_exp_rate: Exponential decay rate (only used if blend_schedule='exponential')
    """
    lr: float = 0.0005
    momentum: float = 0.9
    blend_start: float = 1.0
    blend_end: float = 0.0
    blend_decay_fraction: float = 0.5
    blend_schedule: str = 'linear'
    blend_power: float = 2.0
    blend_exp_rate: float = 5.0

    def __str__(self):
        if self.blend_decay_fraction == 0.0 or self.blend_start == self.blend_end:
            return f"HybridMR_lr{self.lr}_mom{self.momentum}_blend{self.blend_start}"
        else:
            return f"HybridMR_lr{self.lr}_mom{self.momentum}_{self.blend_schedule}_b{self.blend_start}to{self.blend_end}_decay{self.blend_decay_fraction}"

    def create_blend_schedule(self, total_train_steps: int):
        """Create blend schedule function based on config and total training steps.

        Args:
            total_train_steps: Total number of training steps

        Returns:
            Callable that takes current step and returns blend coefficient in [0, 1]
        """
        import math

        decay_steps = math.ceil(self.blend_decay_fraction * total_train_steps)

        if decay_steps == 0 or self.blend_start == self.blend_end:
            # Constant blend
            return lambda step: self.blend_start

        if self.blend_schedule == 'linear':
            def linear_blend(step):
                if step >= decay_steps:
                    return self.blend_end
                progress = step / decay_steps
                return self.blend_start + progress * (self.blend_end - self.blend_start)
            return linear_blend

        elif self.blend_schedule == 'cosine':
            def cosine_blend(step):
                if step >= decay_steps:
                    return self.blend_end
                progress = step / decay_steps
                # Cosine annealing: smooth transition from start to end
                cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                return self.blend_end + cosine_factor * (self.blend_start - self.blend_end)
            return cosine_blend

        elif self.blend_schedule == 'polynomial':
            def polynomial_blend(step):
                if step >= decay_steps:
                    return self.blend_end
                progress = step / decay_steps
                poly_factor = (1 - progress) ** self.blend_power
                return self.blend_end + poly_factor * (self.blend_start - self.blend_end)
            return polynomial_blend

        elif self.blend_schedule == 'exponential':
            def exponential_blend(step):
                if step >= decay_steps:
                    return self.blend_end
                progress = step / decay_steps
                # Exponential decay: exp(-rate * progress)
                exp_factor = math.exp(-self.blend_exp_rate * progress)
                return self.blend_end + exp_factor * (self.blend_start - self.blend_end)
            return exponential_blend

        else:  # constant or unknown
            return lambda step: self.blend_start


# Type alias for optimizer configurations
OptimizerConfig = Union[
    MuonConfig,
    MuonRemezConfig,
    MuonResidualConfig,
    MuonSoftCoupledConfig,
    MuonFixedRankConfig,
    ShampooConfig,
    ShampooNSConfig,
    AdamConfig,
    HybridMuonRemezConfig
]


def parse_optimizer_config(config_str: str) -> OptimizerConfig:
    """
    Parse optimizer config from CLI string format.

    Format: "OptimizerType:param1=value1,param2=value2,..."

    Examples:
        >>> parse_optimizer_config("Muon:lr=0.0005,momentum=0.9")
        MuonConfig(lr=0.0005, momentum=0.9)

        >>> parse_optimizer_config("MuonRemez:lr=0.0005,momentum=0.9")
        MuonRemezConfig(lr=0.0005, momentum=0.9)

        >>> parse_optimizer_config("MuonResidual:lr=0.0005,momentum=0.9,gamma=0.1")
        MuonResidualConfig(lr=0.0005, momentum=0.9, gamma=0.1)

        >>> parse_optimizer_config("MuonSoftCoupled:lr=0.0005,momentum=0.9")
        MuonSoftCoupledConfig(lr=0.0005, momentum=0.9)

        >>> parse_optimizer_config("MuonFixedRank:lr=0.0005,momentum=0.9")
        MuonFixedRankConfig(lr=0.0005, momentum=0.9)

        >>> parse_optimizer_config("Shampoo:lr=0.0005,momentum=0.9,order_multiplier=2")
        ShampooConfig(lr=0.0005, momentum=0.9, order_multiplier=2)

        >>> parse_optimizer_config("Adam:lr=0.01,beta1=0.9,beta2=0.95")
        AdamConfig(lr=0.01, beta1=0.9, beta2=0.95)

        >>> parse_optimizer_config("HybridMuonRemez:lr=0.0005,blend_start=1.0,blend_end=0.0,blend_schedule=linear")
        HybridMuonRemezConfig(lr=0.0005, blend_start=1.0, blend_end=0.0, blend_schedule='linear')

    Args:
        config_str: String specification of optimizer config

    Returns:
        OptimizerConfig instance

    Raises:
        ValueError: If config string format is invalid or optimizer type is unknown
    """
    if ':' not in config_str:
        raise ValueError(f"Config string must contain ':' separator. Got: {config_str}")

    parts = config_str.split(':', 1)
    opt_type = parts[0].strip().lower()

    # Parse parameters
    params = {}
    if len(parts) > 1 and parts[1].strip():
        for param in parts[1].split(','):
            param = param.strip()
            if '=' not in param:
                raise ValueError(f"Parameter must be in format 'key=value'. Got: {param}")

            key, val = param.split('=', 1)
            key = key.strip()
            val = val.strip()

            # Try to convert to appropriate type
            try:
                # Check if it's an integer
                if val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                    params[key] = int(val)
                # Check if it's a float (has decimal point)
                elif '.' in val:
                    params[key] = float(val)
                else:
                    # Keep as string (for blend_schedule, etc.)
                    params[key] = val
            except ValueError:
                # Keep as string if conversion fails
                params[key] = val

    # Create appropriate config
    if opt_type == 'muon':
        return MuonConfig(**params)
    elif opt_type == 'muonremez':
        return MuonRemezConfig(**params)
    elif opt_type == 'muonresidual':
        return MuonResidualConfig(**params)
    elif opt_type == 'muonsoftcoupled':
        return MuonSoftCoupledConfig(**params)
    elif opt_type == 'muonfixedrank':
        return MuonFixedRankConfig(**params)
    elif opt_type == 'shampoo':
        return ShampooConfig(**params)
    elif opt_type == 'shampoons':
        return ShampooNSConfig(**params)
    elif opt_type == 'adam':
        return AdamConfig(**params)
    elif opt_type == 'hybridmuonremez' or opt_type == 'hybrid':
        return HybridMuonRemezConfig(**params)
    else:
        raise ValueError(
            f"Unknown optimizer type: {opt_type}. "
            f"Valid types: 'muon', 'muonremez', 'muonresidual', 'muonsoftcoupled', 'muonfixedrank', "
            f"'shampoo', 'shampoons', 'adam', 'hybridmuonremez'"
        )


def create_optimizer(
    config: OptimizerConfig,
    filter_params: List,
    head_params: List,
    bias_params: List,
    weight_decay: float,
    weight_decay_misc: float,
    lr_head: float,
    lr_bias: float,
    total_train_steps: int = None
) -> List[torch.optim.Optimizer]:
    """
    Create optimizer instances based on configuration.

    This function creates separate optimizers for different parameter groups:
    - Filter/weight parameters use the specified optimizer (Muon, MuonRemez, Shampoo, ShampooNS, or HybridMuonRemez)
    - Biases and head/embedding parameters ALWAYS use Adam (with default settings)

    For CifarNet, the Adam optimizer has 3 param groups:
        param_groups[0]: whitening bias (first element of bias_params)
        param_groups[1]: other biases (remaining elements of bias_params)
        param_groups[2]: head params

    Args:
        config: Optimizer configuration
        filter_params: List of filter/weight parameters (2D+ parameters)
        head_params: List of head/embedding parameters
        bias_params: List of bias parameters (1D parameters, first is whitening bias for CifarNet)
        weight_decay: Weight decay for main optimizer
        weight_decay_misc: Weight decay for biases and heads
        lr_head: Learning rate for head parameters
        lr_bias: Learning rate for bias parameters
        total_train_steps: Total training steps (required for HybridMuonRemezConfig)

    Returns:
        List of optimizer instances for this model

    Example:
        >>> config = HybridMuonRemezConfig(lr=0.0005, blend_start=1.0, blend_end=0.0)
        >>> opts = create_optimizer(
        ...     config, filter_params, head_params, bias_params,
        ...     weight_decay=1.0, weight_decay_misc=1e-4,
        ...     lr_head=0.01, lr_bias=0.01,
        ...     total_train_steps=1000
        ... )
        >>> for opt in opts:
        ...     opt.step()
    """
    optimizers = []

    # Always create Adam optimizer for biases and heads (regardless of main optimizer)
    param_configs_adam = []

    if len(bias_params) > 0:
        # For CifarNet: first bias param is whitening bias (needs separate param group for different LR schedule)
        # Split into 3 param groups: [whitening_bias], [other_biases], [head]
        if len(bias_params) > 1:
            # CifarNet case: separate whitening bias from other biases
            param_configs_adam.append(dict(
                params=bias_params[:1],  # Whitening bias
                lr=lr_bias,
                weight_decay=weight_decay_misc
            ))
            param_configs_adam.append(dict(
                params=bias_params[1:],  # Other biases
                lr=lr_bias,
                weight_decay=weight_decay_misc
            ))
        else:
            # Other architectures: single bias param group
            param_configs_adam.append(dict(
                params=bias_params,
                lr=lr_bias,
                weight_decay=weight_decay_misc
            ))

    if len(head_params) > 0:
        param_configs_adam.append(dict(
            params=head_params,
            lr=lr_head,
            weight_decay=weight_decay_misc
        ))

    if param_configs_adam:
        # Always use Adam for bias/head params with default betas
        adam_opt = Adam(
            param_configs_adam,
            lr=1.0,  # Not used - each param group has its own lr
            betas=(0.9, 0.95),
            eps=1e-10,
            weight_decay=0.0  # Not used - each param group has its own weight_decay
        )
        optimizers.append(adam_opt)

    # Create main optimizer for filter params (Shampoo, ShampooNS, Muon, MuonRemez, MuonResidual, MuonSoftCoupled, MuonFixedRank, or HybridMuonRemez)
    if len(filter_params) > 0:
        if isinstance(config, MuonConfig):
            main_opt = Muon(
                filter_params,
                lr=config.lr,
                momentum=config.momentum,
                nesterov=True,
                weight_decay=weight_decay
            )
            optimizers.append(main_opt)
        elif isinstance(config, MuonRemezConfig):
            main_opt = MuonRemez(
                filter_params,
                lr=config.lr,
                momentum=config.momentum,
                nesterov=True,
                weight_decay=weight_decay
            )
            optimizers.append(main_opt)
        elif isinstance(config, MuonResidualConfig):
            main_opt = MuonResidual(
                filter_params,
                lr=config.lr,
                momentum=config.momentum,
                nesterov=True,
                weight_decay=weight_decay,
                gamma=config.gamma
            )
            optimizers.append(main_opt)
        elif isinstance(config, MuonSoftCoupledConfig):
            main_opt = MuonSoftCoupled(
                filter_params,
                lr=config.lr,
                momentum=config.momentum,
                nesterov=True,
                weight_decay=weight_decay
            )
            optimizers.append(main_opt)
        elif isinstance(config, MuonFixedRankConfig):
            main_opt = MuonFixedRank(
                filter_params,
                lr=config.lr,
                momentum=config.momentum,
                nesterov=True,
                weight_decay=weight_decay
            )
            optimizers.append(main_opt)
        elif isinstance(config, ShampooConfig):
            main_opt = Shampoo(
                filter_params,
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=weight_decay,
                order_multiplier=config.order_multiplier
            )
            optimizers.append(main_opt)
        elif isinstance(config, ShampooNSConfig):
            main_opt = ShampooNS(
                filter_params,
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=weight_decay,
                order_multiplier=config.order_multiplier,
                ns_steps=config.ns_steps
            )
            optimizers.append(main_opt)
        elif isinstance(config, HybridMuonRemezConfig):
            if total_train_steps is None:
                raise ValueError("total_train_steps is required for HybridMuonRemezConfig")

            # Create blend schedule function
            blend_schedule_fn = config.create_blend_schedule(total_train_steps)

            main_opt = HybridMuonRemez(
                filter_params,
                lr=config.lr,
                momentum=config.momentum,
                nesterov=True,
                weight_decay=weight_decay,
                blend_schedule_fn=blend_schedule_fn
            )
            optimizers.append(main_opt)
        else:
            raise ValueError(
                f"Unknown optimizer config: {type(config)}. "
                f"Expected MuonConfig, MuonRemezConfig, MuonResidualConfig, MuonSoftCoupledConfig, "
                f"MuonFixedRankConfig, ShampooConfig, ShampooNSConfig, or HybridMuonRemezConfig."
            )

    return optimizers

