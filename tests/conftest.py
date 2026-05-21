import pytest
import numpy as np
from numpy.typing import NDArray


@pytest.fixture
def rng():
    """Random number generator with fixed seed for reproducibility.

    Returns:
        numpy.random.Generator: RNG seeded with 42

    Example:
        def test_random(rng):
            data = rng.standard_normal((32, 2))
            indices = rng.integers(0, 100, size=10)
    """
    return np.random.default_rng(42)


@pytest.fixture
def make_X():
    """Factory fixture: creates deterministic test data arrays.

    Args:
        n: Number of data points (rows)
        d: Dimensionality of each point (columns/features)

    Returns:
        NumPy array of shape (n, d) with deterministic random values

    Note:
        Each (n, d) pair always produces the same array regardless of call order.
        Different (n, d) pairs produce independent random data.

    Example:
        def test_something(make_X):
            X = make_X(n=16, d=2)    # (16, 2) - 16 points in 2D
            X = make_X(n=32, d=3)    # (32, 3) - 32 points in 3D
            X = make_X(n=100, d=5)   # (100, 5) - 100 points in 5D
    """

    def _make(n: int, d: int) -> NDArray[np.float64]:
        return np.random.default_rng((42, n, d)).standard_normal((n, d))

    return _make


@pytest.fixture
def gaussian_kernel_fn():
    """Gaussian (RBF) kernel function: k(y, x) = exp(-||y - x||²).

    Returns:
        Function k(y, X) where:
            - y: Query point, shape (1, d)
            - X: Data matrix, shape (n, d)
            - Returns: Array of shape (n,) with k(y, x_i) for each row x_i

    Properties:
        - Normalized: k(x, x) = 1 for all x
        - Symmetric: k(x, y) = k(y, x)
        - Decreases with distance: farther points have smaller kernel values

    Example:
        def test_kernel(make_X, gaussian_kernel_fn):
            X = make_X(n=16, d=2)
            y = X[0:1]  # Shape (1, 2)

            k_values = gaussian_kernel_fn(y, X)
            assert k_values.shape == (16,)
            assert k_values[0] == 1.0  # k(x, x) = 1
    """

    def k(y: NDArray[np.float64], X: NDArray[np.float64]) -> NDArray[np.float64]:
        diff = X - y
        return np.exp(-np.sum(diff**2, axis=1))

    return k


@pytest.fixture
def make_K_gauss(make_X, gaussian_kernel_fn):
    """Factory for Gaussian kernel matrices: K[i,j] = exp(-||x_i - x_j||²).

    Returns:
        Function make_K_gauss(n) where:
            - n: Number of data points
            - Returns: Kernel matrix K of shape (n, n)

    Matrix properties:
        - K[i,j] represents similarity between points i and j
        - Symmetric: K[i,j] = K[j,i]
        - Diagonal is 1.0: K[i,i] = 1 (point is identical to itself)
        - Positive semi-definite (all eigenvalues ≥ 0)

    Note:
        Uses make_X(n) internally, so same n always produces same matrix.

    Example:
        def test_kernel_matrix(make_K_gauss):
            K = make_K_gauss(n=16)  # (16, 16) kernel matrix

            assert K.shape == (16, 16)
            assert np.allclose(K, K.T)           # Symmetric
            assert np.allclose(np.diag(K), 1.0)  # Diagonal = 1
    """

    def _make(n: int) -> NDArray[np.float64]:
        from goodpoints.kt import kernel_matrix

        return kernel_matrix(make_X(n, 2), gaussian_kernel_fn)

    return _make


@pytest.fixture
def laplacian_kernel_fn():
    """Laplacian (exponential) kernel: k(y, x) = exp(-||y - x|| / λ).

    Returns:
        Function k(y, X, lam=1.0) where:
            - y: Query point, shape (1, d)
            - X: Data matrix, shape (n, d)
            - lam: Bandwidth parameter (default 1.0)
            - Returns: Array of shape (n,) with k(y, x_i) for each row x_i

    Properties:
        - Normalized: k(x, x) = 1 for all x
        - Symmetric: k(x, y) = k(y, x)
        - Uses L1 distance (||·||₁), not L2 like Gaussian

    Example:
        def test_laplacian(make_X, laplacian_kernel_fn):
            X = make_X(n=16, d=2)
            y = X[0:1]

            k_values = laplacian_kernel_fn(y, X, lam=1.0)
            assert k_values.shape == (16,)
            assert k_values[0] == 1.0  # k(x, x) = 1
    """

    def k(
        y: NDArray[np.float64], X: NDArray[np.float64], lam: float = 1.0
    ) -> NDArray[np.float64]:
        diff = X - y
        dist = np.sum(np.abs(diff), axis=1)
        return np.exp(-dist / lam)

    return k


@pytest.fixture
def imq_kernel_fn():
    """Inverse Multiquadratic (IMQ) kernel: k(y, x) = 1 / sqrt(1 + ||y - x||² / λ²).

    Returns:
        Function k(y, X, lam=1.0) where:
            - y: Query point, shape (1, d)
            - X: Data matrix, shape (n, d)
            - lam: Bandwidth parameter (default 1.0)
            - Returns: Array of shape (n,) with k(y, x_i) for each row x_i

    Properties:
        - Normalized: k(x, x) = 1 for all x
        - Symmetric: k(x, y) = k(y, x)
        - Heavier tails than Gaussian (decays slower with distance)
        - Popular in Stein variational inference

    Example:
        def test_imq(make_X, imq_kernel_fn):
            X = make_X(n=16, d=2)
            y = X[0:1]

            k_values = imq_kernel_fn(y, X, lam=1.0)
            assert k_values.shape == (16,)
            assert k_values[0] == 1.0  # k(x, x) = 1
            assert np.all(k_values > 0)  # Always positive
    """

    def k(
        y: NDArray[np.float64], X: NDArray[np.float64], lam: float = 1.0
    ) -> NDArray[np.float64]:
        diff = X - y
        dist_sq = np.sum(diff**2, axis=1)
        return 1.0 / np.sqrt(1.0 + dist_sq / (lam**2))

    return k
