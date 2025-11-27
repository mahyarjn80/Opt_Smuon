import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import torch

class RemezSolver:
    """
    Implements the Remez Exchange Algorithm to find the minimax optimal
    polynomial approximation for a given function f(x) on interval [a, b].
    """
    def __init__(self, func, degree, interval, tol=1e-10, max_iter=100):
        self.func = func
        self.degree = degree
        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter
        self.coeffs = None
        self.error_bound = None

    def _eval_poly(self, coeffs, x):
        """Evaluates polynomial with coefficients c at x."""
        return np.polyval(coeffs[::-1], x) # NumPy expects highest degree first

    def _solve_system(self, nodes):
        """
        Solves the linear system: P(x_i) + (-1)^i * E = f(x_i)
        for polynomial coefficients and Error E.
        """
        n = self.degree
        # System size is (n+2) x (n+2): n+1 coeffs + 1 Error term
        A = np.zeros((n + 2, n + 2))
        b = np.zeros(n + 2)

        # Build Vandermonde-like matrix
        for i, x in enumerate(nodes):
            # Polynomial terms: c0 + c1*x + ... + cn*x^n
            for j in range(n + 1):
                A[i, j] = x**j
            # Alternating error term
            A[i, n + 1] = (-1)**i
            b[i] = self.func(x)

        try:
            solution = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback for stability if standard solve fails (rare in low degree)
            return None, None

        coeffs = solution[:n + 1]
        E = solution[n + 1]
        return coeffs, E

    def _find_extrema(self, coeffs, nodes):
        """
        Finds the new set of extrema for the error function e(x) = f(x) - P(x).
        Uses a dense grid search + refinement.
        """
        # Create a dense grid to find approximate extrema
        grid_size = 1000 * self.degree
        grid = np.linspace(self.interval[0], self.interval[1], grid_size)

        # Calculate error on grid
        poly_vals = self._eval_poly(coeffs, grid)
        func_vals = self.func(grid)
        error = func_vals - poly_vals
        abs_error = np.abs(error)

        # Find local maxima of absolute error
        # In a real Remez implementation, we carefully select n+2 alternating signs.
        # For simplicity here, we pick the global max error points effectively.

        # Simple Exchange Strategy:
        # 1. Find the point of maximum absolute error
        max_err_idx = np.argmax(abs_error)
        max_x = grid[max_err_idx]
        max_err = error[max_err_idx]

        # 2. Replace the closest node in the current set with this new max,
        #    preserving alternating signs of the error if possible.
        #    (Standard Remez requires maintaining sign alternation).

        # Recalculate errors at current nodes to check signs
        current_node_errs = self.func(nodes) - self._eval_poly(coeffs, nodes)

        new_nodes = nodes.copy()

        # Find which interval the max_x falls into or which node it is closest to
        # Heuristic: Replace node with same sign of error as max_err

        # This is a simplified exchange step.
        # For full robustness, we'd ensure the alternating property exactly.
        # Here we swap the node that has the same error sign and maximizes the gap.

        sign_new = np.sign(max_err)

        best_idx = -1

        # Try to find a node with the same error sign to replace
        for i in range(len(nodes)):
            if np.sign(current_node_errs[i]) == sign_new:
                # Potential candidate
                new_nodes[i] = max_x
                # Sort to keep order
                new_nodes.sort()
                return new_nodes, np.max(abs_error)

        # If no same sign found (edge case), replace based on distance
        dists = np.abs(nodes - max_x)
        idx = np.argmin(dists)
        new_nodes[idx] = max_x
        new_nodes.sort()

        return new_nodes, np.max(abs_error)

    def solve(self):
        # 1. Initialize nodes (Chebyshev nodes are a good start)
        a, b = self.interval
        n = self.degree
        k = np.arange(n + 2)
        # Chebyshev nodes mapped to [a, b]
        nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2*k + 1) * np.pi / (2*(n + 2)))
        nodes.sort()

        for it in range(self.max_iter):
            coeffs, E = self._solve_system(nodes)
            if coeffs is None:
                print("Linear solve failed.")
                break

            new_nodes, max_abs_err = self._find_extrema(coeffs, nodes)

            # Check convergence
            # Remez converges when the max error on the interval is close to the max error on the nodes (|E|)
            if np.abs(max_abs_err - np.abs(E)) < self.tol:
                self.coeffs = coeffs
                self.error_bound = np.abs(E)
                return coeffs

            nodes = new_nodes

        self.coeffs = coeffs
        return coeffs

# Global cache for coefficients: Key=(alpha, degree), Value=coeffs
_COEFF_CACHE = {}

def get_cached_coefficients(alpha, degree=15, fixed_range=(1e-6, 1.0)):
    """
    Offline Stage: Computes optimal coefficients for a FIXED range [0, 1] (or close to it).
    These coefficients are reused for any matrix by normalizing the matrix first.
    """
    cache_key = (alpha, degree, fixed_range)

    if cache_key in _COEFF_CACHE:
        return _COEFF_CACHE[cache_key]

    # Range for y = sigma^2 (since we iterate on M^T M)
    # If we normalize M such that ||M|| <= 1, then sigma^2 is in [0, 1]
    # We use a small lower bound epsilon to avoid singularity at 0 if alpha < 1
    l, u = fixed_range
    y_min, y_max = l**2, u**2

    # Target function P(y) ~ y^((alpha - 1)/2)
    exponent = (alpha - 1.0) / 2.0
    target_func = lambda y: y ** exponent

    # Suppress output for cleaner training logs
    # print(f"[Offline] Computing coeffs for Alpha={alpha}, Degree={degree} on range [{y_min:.1e}, {y_max:.1e}]...")
    solver = RemezSolver(target_func, degree, [y_min, y_max])
    coeffs = solver.solve()

    _COEFF_CACHE[cache_key] = coeffs
    return coeffs

def compute_matrix_fractional_power_remez(M, alpha, degree=15):
    """
    Computes U * Sigma^alpha * V^T efficiently.
    Works with both NumPy arrays and PyTorch tensors.

    1. Normalizes M to ensure spectral norm <= 1.
    2. Fetches cached coefficients (Offline Stage).
    3. Applies polynomial (Online Stage).
    4. Rescales result.

    Args:
        M: Input matrix (NumPy array or PyTorch tensor)
        alpha: Fractional power exponent
        degree: Polynomial degree for Remez approximation

    Returns:
        M^alpha in the same format as input (NumPy or PyTorch)
    """
    # Detect if input is PyTorch tensor
    is_torch = isinstance(M, torch.Tensor)

    if is_torch:
        return _compute_matrix_fractional_power_remez_torch(M, alpha, degree)
    else:
        return _compute_matrix_fractional_power_remez_numpy(M, alpha, degree)


def _compute_matrix_fractional_power_remez_torch(M, alpha, degree=15):
    """
    PyTorch version of compute_matrix_fractional_power_remez.
    Handles tensors on any device (CPU/CUDA).
    """
    device = M.device
    dtype = M.dtype
    m, n = M.shape

    # 1. Normalize Matrix
    scale = torch.linalg.matrix_norm(M, ord='fro') + 1e-8
    M_norm = M / scale

    # 2. Get Coefficients (computed in NumPy, then convert to PyTorch)
    coeffs_np = get_cached_coefficients(alpha, degree, fixed_range=(1e-3, 1.0))
    coeffs = torch.tensor(coeffs_np, device=device, dtype=dtype)

    # 3. Evaluate Polynomial on G = M_norm^T @ M_norm
    if m >= n:
        G = M_norm.T @ M_norm

        # Horner's method: P(G)
        P_G = coeffs[-1] * torch.eye(n, device=device, dtype=dtype)
        for i in range(degree - 1, -1, -1):
            P_G = coeffs[i] * torch.eye(n, device=device, dtype=dtype) + G @ P_G

        # X_norm = M_norm @ P(G)
        X_norm = M_norm @ P_G
    else:
        G = M_norm @ M_norm.T
        P_G = coeffs[-1] * torch.eye(m, device=device, dtype=dtype)
        for i in range(degree - 1, -1, -1):
            P_G = coeffs[i] * torch.eye(m, device=device, dtype=dtype) + G @ P_G

        X_norm = P_G @ M_norm

    # 4. Rescale Result
    X = X_norm * (scale ** alpha)

    return X


def _compute_matrix_fractional_power_remez_numpy(M, alpha, degree=15):
    """
    NumPy version of compute_matrix_fractional_power_remez.
    Original implementation for NumPy arrays.
    """
    m, n = M.shape

    # 1. Normalize Matrix
    scale = np.linalg.norm(M, 'fro') + 1e-8
    M_norm = M / scale

    # 2. Get Coefficients
    coeffs = get_cached_coefficients(alpha, degree, fixed_range=(1e-3, 1.0))

    # 3. Evaluate Polynomial on G = M_norm^T M_norm
    if m >= n:
        G = M_norm.T @ M_norm

        # Horner-like evaluation: P(G)
        P_G = coeffs[-1] * np.eye(n)
        for i in range(degree - 1, -1, -1):
            P_G = coeffs[i] * np.eye(n) + G @ P_G

        # X_norm = M_norm @ P(G)
        X_norm = M_norm @ P_G
    else:
        G = M_norm @ M_norm.T
        P_G = coeffs[-1] * np.eye(m)
        for i in range(degree - 1, -1, -1):
            P_G = coeffs[i] * np.eye(m) + G @ P_G

        X_norm = P_G @ M_norm

    # 4. Rescale Result
    X = X_norm * (scale ** alpha)

    return X

def verification_checks():
    """
    Generates a random matrix and verifies the numerical approximation
    against the exact SVD solution.
    """
    print("--- Starting Verification ---")

    # Setup
    np.random.seed(42)
    rows, cols = 50, 50 # Small size for verification

    # Generate random matrix with controlled condition number for fair testing
    # (High condition numbers require higher degrees or multiple iterations)
    U_true, _ = np.linalg.qr(np.random.randn(rows, rows))
    V_true, _ = np.linalg.qr(np.random.randn(cols, cols))
    singular_values = np.linspace(0.1, 2.0, min(rows, cols)) # Range [0.1, 2.0]
    S_true = np.diag(singular_values)

    if rows > cols:
        S_mat = np.zeros((rows, cols))
        S_mat[:cols, :cols] = S_true
    elif cols > rows:
        S_mat = np.zeros((rows, cols))
        S_mat[:rows, :rows] = S_true
    else:
        S_mat = S_true

    M = U_true @ S_mat @ V_true.T

    # Test cases: Different Alphas
    alphas = [0.0, 0.5, 0.9, -0.5] # 0.0 is Polar, 0.9 is near Identity

    for alpha in alphas:
        print(f"\n--- Testing Alpha = {alpha} ---")

        # 1. Exact Solution via SVD
        # U * S^alpha * V.T
        S_alpha = singular_values ** alpha

        if rows == cols:
            X_exact = U_true @ np.diag(S_alpha) @ V_true.T

        # 2. Numerical Approximation
        # Note: We no longer pass ranges manually; the function handles normalization.
        X_approx = compute_matrix_fractional_power_remez(M, alpha, degree=15)

        # 3. Compare
        diff = X_approx - X_exact
        spectral_error = np.linalg.norm(diff, 2)
        relative_error = spectral_error / np.linalg.norm(X_exact, 2)

        print(f"Exact vs Approx Spectral Norm Error: {spectral_error:.2e}")
        print(f"Relative Error: {relative_error:.2%}")

        if relative_error < 1e-3:
            print(">> CHECK PASSED")
        else:
            print(">> CHECK FAILED (Try increasing degree or checking condition number)")

if __name__ == "__main__":
    verification_checks()
