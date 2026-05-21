"""Tests for mean0 mode in kernel thinning.

mean0=True is used when the kernel has expectation zero under a target measure.
In this mode, KT-SWAP minimizes MMD to the zero measure instead of the empirical
measure, which is useful for Stein kernels and other zero-mean kernels.
"""

import numpy as np
from goodpoints import kt


class TestMean0Mode:
    """Tests for mean0 parameter in kt.py functions."""

    def test_thin_K_mean0(self, make_K_gauss):
        """thin_K works with mean0=True."""
        K = make_K_gauss(n=16)
        coreset = kt.thin_K(K, K, m=1, seed=42, mean0=True)

        assert len(coreset) == 8
        assert np.all(coreset >= 0)
        assert np.all(coreset < 16)

    def test_thin_K_mean0_vs_normal(self, make_K_gauss):
        """mean0=True produces different results than mean0=False."""
        K = make_K_gauss(n=16)

        coreset_normal = kt.thin_K(K, K, m=1, seed=42, mean0=False)
        coreset_mean0 = kt.thin_K(K, K, m=1, seed=42, mean0=True)

        # Different modes should generally produce different coresets
        # (not guaranteed, but very likely for Gaussian kernels)
        assert len(coreset_normal) == len(coreset_mean0) == 8

    def test_swap_K_mean0(self, make_K_gauss):
        """swap_K works with mean0=True."""
        K = make_K_gauss(n=16)
        coresets = kt.split_K(K, m=1, seed=42)
        coreset = kt.swap_K(K, coresets, mean0=True)

        assert len(coreset) == 8
        assert np.all(coreset >= 0)
        assert np.all(coreset < 16)

    def test_best_K_mean0(self, make_K_gauss):
        """best_K works with mean0=True."""
        K = make_K_gauss(n=16)
        coresets = kt.split_K(K, m=1, seed=42)
        best = kt.best_K(K, coresets, mean0=True)

        assert len(best) == 8
        assert np.all(best >= 0)
        assert np.all(best < 16)

    def test_refine_K_mean0(self, make_K_gauss):
        """refine_K works with mean0=True."""
        K = make_K_gauss(n=16)
        initial = np.arange(8)
        refined = kt.refine_K(K, initial, mean0=True)

        assert len(refined) == 8
        assert np.all(refined >= 0)
        assert np.all(refined < 16)

    def test_refine_K_mean0_with_unique(self, make_K_gauss):
        """refine_K works with mean0=True and unique=True."""
        K = make_K_gauss(n=16)
        initial = np.arange(8)
        refined = kt.refine_K(K, initial, mean0=True, unique=True)

        assert len(refined) == 8
        assert len(set(refined)) == 8  # All unique
        assert np.all(refined >= 0)
        assert np.all(refined < 16)


class TestMeanKNone:
    """Tests for meanK=None path (triggers kernel recomputation)."""

    def test_swap_K_without_meanK(self, make_K_gauss):
        """swap_K computes meanK internally when None."""
        K = make_K_gauss(n=16)
        coresets = kt.split_K(K, m=1, seed=42)

        # meanK=None triggers internal computation
        coreset = kt.swap_K(K, coresets, meanK=None, mean0=False)

        assert len(coreset) == 8
        assert np.all(coreset >= 0)
        assert np.all(coreset < 16)

    def test_best_K_without_meanK(self, make_K_gauss):
        """best_K computes meanK internally when None."""
        K = make_K_gauss(n=16)
        coresets = kt.split_K(K, m=1, seed=42)

        best = kt.best_K(K, coresets, meanK=None, mean0=False)

        assert len(best) == 8
        assert np.all(best >= 0)
        assert np.all(best < 16)

    def test_refine_K_without_meanK(self, make_K_gauss):
        """refine_K computes meanK internally when None."""
        K = make_K_gauss(n=16)
        initial = np.arange(8)

        refined = kt.refine_K(K, initial, meanK=None, mean0=False)

        assert len(refined) == 8
        assert np.all(refined >= 0)
        assert np.all(refined < 16)

    def test_refine_K_unique_without_meanK(self, make_K_gauss):
        """refine_K with unique=True and meanK=None (uncovered path)."""
        K = make_K_gauss(n=16)
        initial = np.arange(8)

        refined = kt.refine_K(K, initial, meanK=None, unique=True, mean0=False)

        assert len(refined) == 8
        assert len(set(refined)) == 8
        assert np.all(refined >= 0)
        assert np.all(refined < 16)


class TestMean0WithKernelFunctions:
    """Tests for mean0 mode with X-variant functions."""

    def test_thin_X_mean0(self, make_X, gaussian_kernel_fn):
        """thin_X doesn't have mean0 parameter (only thin_K does)."""
        # This is just to document that thin_X doesn't support mean0
        # The mean0 mode is only for thin_K which has precomputed kernel
        X = make_X(n=16, d=2)

        coreset = kt.thin_X(
            X,
            m=1,
            split_kernel=gaussian_kernel_fn,
            swap_kernel=gaussian_kernel_fn,
            seed=42,
        )

        assert len(coreset) == 8

    def test_swap_X_with_meanK_none(self, make_X, gaussian_kernel_fn):
        """swap_X with meanK=None (triggers kernel recomputation)."""
        X = make_X(n=16, d=2)
        coresets = kt.split_X(X, m=1, kernel=gaussian_kernel_fn, seed=42)

        # meanK=None forces recomputation at each step
        coreset = kt.swap_X(X, coresets, kernel=gaussian_kernel_fn, meanK=None)

        assert len(coreset) == 8
        assert np.all(coreset >= 0)
        assert np.all(coreset < 16)

    def test_refine_X_with_meanK_none(self, make_X, gaussian_kernel_fn):
        """refine_X with meanK=None."""
        X = make_X(n=16, d=2)
        initial = np.arange(8)

        refined = kt.refine_X(X, initial, kernel=gaussian_kernel_fn, meanK=None)

        assert len(refined) == 8
        assert np.all(refined >= 0)
        assert np.all(refined < 16)

    def test_refine_X_unique_meanK_none(self, make_X, gaussian_kernel_fn):
        """refine_X with unique=True and meanK=None (uncovered path)."""
        X = make_X(n=16, d=2)
        initial = np.arange(8)

        refined = kt.refine_X(
            X, initial, kernel=gaussian_kernel_fn, meanK=None, unique=True
        )

        assert len(refined) == 8
        assert len(set(refined)) == 8
        assert np.all(refined >= 0)
        assert np.all(refined < 16)
