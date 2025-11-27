import torch

def get_quintic_coeffs(variant='muon_aggressive'):
    """
    Returns polynomial coefficients (alpha, beta, gamma).
    Higham (Standard): Order 5 convergence, very stable.
    Muon Aggressive: Optimized for speed (steep slope at 0), theoretically derived for 5 steps.
    """
    if variant == 'higham':
        return 1.875, -1.25, 0.375
    elif variant == 'muon_aggressive':
        # These are approximate values often used to explain the "Greedy" approach
        # Real Muon implementations tune these based on specific constraints.
        return 3.4445, -4.7750, 2.0315
    return 1.875, -1.25, 0.375
    """
    Standard Muon Algorithm.
    Target: Polar Factor (U * V.T).
    Formula: X_new = X * (alpha*I + beta*X^T*X + gamma*(X^T*X)^2)
    """
    alpha, beta, gamma = get_quintic_coeffs(variant)
    
    # 1. Spectral Normalization
    norm_M = torch.norm(M, p='fro')
    if norm_M < 1e-8: return torch.zeros_like(M)
    X = M / norm_M
    
    I = torch.eye(M.shape[1], device=M.device, dtype=M.dtype)

    for k in range(steps):
        # Gram Matrix A = X^T X
        A = X.T @ X
        A2 = A @ A
        
        # Update Factor S
        S = alpha * I + beta * A + gamma * A2
        
        # Update X
        X = X @ S
        
    # Result is U V^T (Unitary), no denormalization needed for Polar factor.
    return X

def coupled_newton_schulz_quintic(A, steps=5, variant='muon_aggressive'):
    """
    Coupled Iteration for Square Root.
    Target: Z -> A^(-1/2), Y -> A^(1/2)
    """
    alpha, beta, gamma = get_quintic_coeffs(variant)
    
    # Normalization (Crucial)
    norm_A = torch.norm(A, p='fro')
    if norm_A < 1e-8: return torch.zeros_like(A), torch.zeros_like(A)
    A_scaled = A / norm_A
    
    Y = A_scaled
    Z = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)

    for k in range(steps):
        P = Z @ Y
        P2 = P @ P
        S = alpha * I + beta * P + gamma * P2
        Y = Y @ S
        Z = S @ Z
        
    sqrt_norm = torch.sqrt(norm_A)
    return Y * sqrt_norm, Z * (1.0 / sqrt_norm)

def compute_u_sigma_half_vt(M, steps=5, variant='muon_aggressive'):
    """
    Chained Coupled Iteration.
    Target: U * Sigma^0.5 * V.T
    """
    # Pre-normalize M
    norm_M = torch.norm(M, p='fro')
    if norm_M < 1e-8: return torch.zeros_like(M)
    M_scaled = M / norm_M

    if M_scaled.shape[0] >= M_scaled.shape[1]:
        A = M_scaled.T @ M_scaled
        _, Z1 = coupled_newton_schulz_quintic(A, steps, variant)
        Y2, _ = coupled_newton_schulz_quintic(Z1, steps, variant)
        Result_scaled = M_scaled @ Y2
    else:
        A = M_scaled @ M_scaled.T
        _, Z1 = coupled_newton_schulz_quintic(A, steps, variant)
        Y2, _ = coupled_newton_schulz_quintic(Z1, steps, variant)
        Result_scaled = Y2 @ M_scaled

    return Result_scaled * torch.sqrt(norm_M)
