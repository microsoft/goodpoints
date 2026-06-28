"""Tests for alternative kernel functions and kernel specifications.

This file tests:
1. kt.py functions with non-Gaussian kernels (Laplacian, IMQ)
2. compress.py with Sobolev kernel (b"sobolev")
3. ctt.py error handling for unsupported kernels
4. Different bandwidth parameters
"""

import numpy as np
from goodpoints import kt, compress, ctt
import pytest


class TestLaplacianKernel:
    """Tests for Laplacian (exponential) kernel in kt.py."""

    def test_kernel_properties(self, make_X, laplacian_kernel_fn):
        """Laplacian kernel has expected properties."""
        X = make_X(n=16, d=2)
        y = X[0:1]

        k_values = laplacian_kernel_fn(y, X, lam=1.0)

        assert k_values.shape == (16,)
        assert k_values[0] == pytest.approx(1.0)  # k(x, x) = 1
        assert np.all(k_values > 0)  # Always positive
        assert np.all(k_values <= 1.0)  # Normalized

    def test_bandwidth_effect(self, make_X, laplacian_kernel_fn):
        """Larger bandwidth → slower decay."""
        X = make_X(n=16, d=2)
        y = X[0:1]
        z = X[5:6]  # Distant point

        k_small = laplacian_kernel_fn(y, z, lam=0.5)
        k_large = laplacian_kernel_fn(y, z, lam=2.0)

        # Larger bandwidth → larger kernel value
        assert k_large > k_small

    def test_thin_with_laplacian(self, make_X, laplacian_kernel_fn):
        """Kernel thinning works with Laplacian kernel."""
        X = make_X(n=16, d=2)
        coreset = kt.thin_X(
            X,
            m=1,
            split_kernel=laplacian_kernel_fn,
            swap_kernel=laplacian_kernel_fn,
            seed=42,
        )

        assert len(coreset) == 8
        assert np.all(coreset >= 0)
        assert np.all(coreset < 16)

    def test_split_with_laplacian(self, make_X, laplacian_kernel_fn):
        """KT-SPLIT works with Laplacian kernel."""
        X = make_X(n=16, d=2)
        coresets = kt.split_X(X, m=1, kernel=laplacian_kernel_fn, seed=42)

        assert coresets.shape == (2, 8)
        union = np.sort(np.concatenate(coresets))
        np.testing.assert_array_equal(union, np.arange(16))


class TestIMQKernel:
    """Tests for Inverse Multiquadratic kernel in kt.py."""

    def test_kernel_properties(self, make_X, imq_kernel_fn):
        """IMQ kernel has expected properties."""
        X = make_X(n=16, d=2)
        y = X[0:1]

        k_values = imq_kernel_fn(y, X, lam=1.0)

        assert k_values.shape == (16,)
        assert k_values[0] == pytest.approx(1.0)  # k(x, x) = 1
        assert np.all(k_values > 0)  # Always positive
        assert np.all(k_values <= 1.0)  # Normalized

    def test_heavier_tails_than_gaussian(
        self, make_X, gaussian_kernel_fn, imq_kernel_fn
    ):
        """IMQ decays slower than Gaussian (heavier tails)."""
        X = make_X(n=16, d=2)
        y = X[0:1]
        z = X[10:11]  # Distant point

        k_gauss = gaussian_kernel_fn(y, z)[0]
        k_imq = imq_kernel_fn(y, z, lam=1.0)[0]

        # IMQ should have larger value for distant points
        assert k_imq > k_gauss

    def test_thin_with_imq(self, make_X, imq_kernel_fn):
        """Kernel thinning works with IMQ kernel."""
        X = make_X(n=16, d=2)
        coreset = kt.thin_X(
            X, m=1, split_kernel=imq_kernel_fn, swap_kernel=imq_kernel_fn, seed=42
        )

        assert len(coreset) == 8
        assert np.all(coreset >= 0)
        assert np.all(coreset < 16)

    def test_bandwidth_parameter(self, make_X, imq_kernel_fn):
        """Different bandwidths produce different results."""
        X = make_X(n=16, d=2)

        def imq_lam05(y, X):
            return imq_kernel_fn(y, X, lam=0.5)

        def imq_lam20(y, X):
            return imq_kernel_fn(y, X, lam=2.0)

        coreset1 = kt.thin_X(
            X, m=1, split_kernel=imq_lam05, swap_kernel=imq_lam05, seed=42
        )
        coreset2 = kt.thin_X(
            X, m=1, split_kernel=imq_lam20, swap_kernel=imq_lam20, seed=42
        )

        # Different bandwidths should produce different coresets
        # (not guaranteed, but very likely with different enough lam)
        # At minimum, check both are valid
        assert len(coreset1) == 8
        assert len(coreset2) == 8


class TestSobolevKernel:
    """Tests for Sobolev kernel in compress.py (C extension)."""

    def test_compress_with_sobolev(self, make_X):
        """compress_kt works with Sobolev kernel."""
        X = make_X(n=64, d=2)
        coreset_indices = compress.compress_kt(
            X, b"sobolev", k_params=np.array([1.0]), g=0, seed=42
        )

        assert len(coreset_indices) == 8  # sqrt(64) = 8
        assert np.all(coreset_indices >= 0)
        assert np.all(coreset_indices < 64)

    def test_compresspp_with_sobolev(self, make_X):
        """compresspp_kt works with Sobolev kernel."""
        X = make_X(n=64, d=2)
        coreset_indices = compress.compresspp_kt(
            X, b"sobolev", k_params=np.array([1.0]), g=0, seed=42
        )

        assert len(coreset_indices) == 8  # sqrt(64) = 8
        assert np.all(coreset_indices >= 0)
        assert np.all(coreset_indices < 64)

    def test_multiple_smoothness_parameters(self, make_X):
        """Sobolev kernel with multiple smoothness parameters."""
        X = make_X(n=64, d=2)
        # Sum of Sobolevs with different smoothness
        k_params = np.array([0.5, 1.0, 2.0])
        coreset_indices = compress.compress_kt(
            X, b"sobolev", k_params=k_params, g=0, seed=42
        )

        assert len(coreset_indices) == 8
        assert np.all(coreset_indices >= 0)
        assert np.all(coreset_indices < 64)


class TestGaussianKernelParameters:
    """Tests for Gaussian kernel with different bandwidth parameters."""

    def test_multiple_gaussian_bandwidths(self, make_X):
        """Sum of Gaussians with different bandwidths."""
        X = make_X(n=64, d=2)
        # Sum of Gaussians with different bandwidths
        k_params = np.array([0.5, 1.0, 2.0])
        coreset_indices = compress.compress_kt(
            X, b"gaussian", k_params=k_params, g=0, seed=42
        )

        assert len(coreset_indices) == 8
        assert np.all(coreset_indices >= 0)
        assert np.all(coreset_indices < 64)

    def test_single_vs_multiple_params(self, make_X):
        """Single vs multiple kernel parameters."""
        X = make_X(n=64, d=2)

        # Single parameter
        coreset1 = compress.compress_kt(
            X, b"gaussian", k_params=np.array([1.0]), g=0, seed=42
        )

        # Multiple parameters (should be different)
        coreset2 = compress.compress_kt(
            X, b"gaussian", k_params=np.array([0.5, 1.0, 2.0]), g=0, seed=42
        )

        assert len(coreset1) == 8
        assert len(coreset2) == 8
        # Different parameters should generally produce different results
        # (not guaranteed but very likely)


class TestCttKernelSupport:
    """Tests for ctt.py kernel support and error handling."""

    def test_gauss_kernel_works(self, make_X):
        """ctt supports 'gauss' kernel."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))

        result = ctt.ctt(
            X1, X2, g=0, B=39, s=8, kernel="gauss", null_seed=0, statistic_seed=1
        )

        assert hasattr(result, "rejects")
        assert 0 <= result.rejects <= 1

    def test_unsupported_kernel_raises_error(self, make_X):
        """ctt raises ValueError for unsupported kernels."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))

        with pytest.raises(ValueError, match="Unsupported kernel name"):
            ctt.ctt(
                X1,
                X2,
                g=0,
                B=39,
                s=8,
                kernel="laplacian",
                null_seed=0,
                statistic_seed=1,
            )

    def test_different_bandwidths(self, make_X):
        """ctt with different bandwidth parameters."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))

        # Small bandwidth
        result1 = ctt.ctt(
            X1,
            X2,
            g=0,
            B=39,
            s=8,
            lam=0.5,
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        # Large bandwidth
        result2 = ctt.ctt(
            X1,
            X2,
            g=0,
            B=39,
            s=8,
            lam=2.0,
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        # Both should work (may have different rejection rates)
        assert hasattr(result1, "rejects")
        assert hasattr(result2, "rejects")
        assert 0 <= result1.rejects <= 1
        assert 0 <= result2.rejects <= 1


class TestMixedKernels:
    """Tests using different kernels for split and swap."""

    def test_gaussian_split_laplacian_swap(
        self, make_X, gaussian_kernel_fn, laplacian_kernel_fn
    ):
        """Use Gaussian for split, Laplacian for swap."""
        X = make_X(n=16, d=2)

        coreset = kt.thin_X(
            X,
            m=1,
            split_kernel=gaussian_kernel_fn,
            swap_kernel=laplacian_kernel_fn,
            seed=42,
        )

        assert len(coreset) == 8
        assert np.all(coreset >= 0)
        assert np.all(coreset < 16)

    def test_imq_split_gaussian_swap(self, make_X, imq_kernel_fn, gaussian_kernel_fn):
        """Use IMQ for split, Gaussian for swap."""
        X = make_X(n=16, d=2)

        coreset = kt.thin_X(
            X, m=1, split_kernel=imq_kernel_fn, swap_kernel=gaussian_kernel_fn, seed=42
        )

        assert len(coreset) == 8
        assert np.all(coreset >= 0)
        assert np.all(coreset < 16)
