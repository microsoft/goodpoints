"""Integration tests for cross-module workflows.

This file tests workflows that span multiple modules:
1. compress.py + kt.py: Compression followed by kernel thinning
2. kt.py + ctt.py: Thinning with hypothesis testing
3. End-to-end thinning pipelines
4. Different kernel backends working together
"""

import numpy as np
from goodpoints import kt, compress, ctt, herding


class TestCompressThenThin:
    """Test workflows combining compression and kernel thinning."""

    def test_compress_then_thin_pipeline(self, make_X):
        """Compress large dataset then apply kernel thinning."""
        X = make_X(n=256, d=2)

        # Step 1: Compress with compress_kt
        compressed_indices = compress.compress_kt(
            X, b"gaussian", k_params=np.array([1.0]), g=1, seed=42
        )

        # Step 2: Apply thin_X on full dataset
        X_compressed = X[compressed_indices]
        thinned_indices_local = kt.thin_X(
            X_compressed,
            m=1,
            split_kernel=lambda y, X: np.exp(-np.sum((X - y) ** 2, axis=1)),
            swap_kernel=lambda y, X: np.exp(-np.sum((X - y) ** 2, axis=1)),
            seed=42,
        )

        # Map back to original indices
        thinned_indices = compressed_indices[thinned_indices_local]

        # Should get small coreset
        # compress_kt with g=1 on n=256: output = sqrt(sqrt(256)) * 4 = 4 * 4 = 16 per bin
        # But actual compress output varies; just check it's reasonable
        assert len(X_compressed) > 0
        # thin_X with m=1 halves the size
        assert len(thinned_indices) == len(X_compressed) // 2
        assert np.all(thinned_indices >= 0)
        assert np.all(thinned_indices < 256)

    def test_compresspp_then_swap(self, make_X, gaussian_kernel_fn):
        """compresspp_kt followed by swap refinement."""
        X = make_X(n=256, d=2)

        # Compress with compresspp
        coreset_indices = compress.compresspp_kt(
            X, b"gaussian", k_params=np.array([1.0]), g=1, seed=42
        )

        # Build kernel matrix for coreset
        X_coreset = X[coreset_indices]
        from goodpoints.kt import kernel_matrix

        K_coreset = kernel_matrix(X_coreset, gaussian_kernel_fn)

        # Split and swap
        coresets = kt.split_K(K_coreset, m=1, seed=42)
        refined = kt.swap_K(K_coreset, coresets)

        assert len(refined) == 8
        assert np.all(refined >= 0)
        assert np.all(refined < len(X_coreset))


class TestThinningWithHypothesisTesting:
    """Test combining kernel thinning with statistical tests."""

    def test_thin_then_ctt(self, make_X):
        """Thin datasets then run CTT hypothesis test."""
        rng = np.random.default_rng(42)

        # Two large datasets from same distribution
        X1 = rng.standard_normal((512, 2))
        X2 = rng.standard_normal((512, 2))

        # Thin both datasets
        coreset1 = kt.thin_X(
            X1,
            m=2,
            split_kernel=lambda y, X: np.exp(-np.sum((X - y) ** 2, axis=1)),
            swap_kernel=lambda y, X: np.exp(-np.sum((X - y) ** 2, axis=1)),
            seed=42,
        )
        coreset2 = kt.thin_X(
            X2,
            m=2,
            split_kernel=lambda y, X: np.exp(-np.sum((X - y) ** 2, axis=1)),
            swap_kernel=lambda y, X: np.exp(-np.sum((X - y) ** 2, axis=1)),
            seed=43,
        )

        # Run CTT on coresets
        X1_thin = X1[coreset1]
        X2_thin = X2[coreset2]

        result = ctt.ctt(
            X1_thin,
            X2_thin,
            g=0,
            B=39,
            s=16,
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        # Same distribution should not reject
        assert result is not None
        assert result.rejects < 1  # Type I error control

    def test_compress_then_lrctt(self, make_X):
        """Compress datasets then run large-scale CTT."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((512, 2))
        X2 = rng.standard_normal((512, 2))

        # Compress both datasets
        indices1 = compress.compress_kt(
            X1, b"gaussian", k_params=np.array([1.0]), g=1, seed=42
        )
        indices2 = compress.compress_kt(
            X2, b"gaussian", k_params=np.array([1.0]), g=1, seed=43
        )

        X1_comp = X1[indices1]
        X2_comp = X2[indices2]

        # Run lrctt on compressed data
        result = ctt.lrctt(
            X1_comp,
            X2_comp,
            r=32,
            g=0,
            a=0,
            B=39,
            s=8,
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        assert result is not None
        assert hasattr(result, "rejects")


class TestEndToEndPipelines:
    """Test complete end-to-end thinning pipelines."""

    def test_full_kt_pipeline_with_K_matrices(self, make_X, gaussian_kernel_fn):
        """Complete KT pipeline: split → swap → refine."""
        X = make_X(n=64, d=2)
        from goodpoints.kt import kernel_matrix

        K = kernel_matrix(X, gaussian_kernel_fn)
        meanK = K.mean(axis=1)

        # Split
        coresets = kt.split_K(K, m=2, seed=42)
        assert coresets.shape == (4, 16)

        # Swap
        swapped = kt.swap_K(K, coresets, meanK=meanK)
        assert len(swapped) == 16

        # Refine
        refined = kt.refine_K(K, swapped, meanK=meanK)
        assert len(refined) == 16

        # All indices should be valid
        assert np.all(refined >= 0)
        assert np.all(refined < 64)

    def test_full_kt_pipeline_with_X_arrays(self, make_X, gaussian_kernel_fn):
        """Complete KT pipeline using X-variant functions."""
        X = make_X(n=64, d=2)

        # Split
        coresets = kt.split_X(X, m=2, kernel=gaussian_kernel_fn, seed=42)
        assert coresets.shape == (4, 16)

        # Swap
        swapped = kt.swap_X(X, coresets, kernel=gaussian_kernel_fn)
        assert len(swapped) == 16

        # Refine
        refined = kt.refine_X(X, swapped, kernel=gaussian_kernel_fn)
        assert len(refined) == 16

        # All unique
        refined_unique = kt.refine_X(X, swapped, kernel=gaussian_kernel_fn, unique=True)
        assert len(set(refined_unique)) == 16

    def test_thin_with_store_K_modes(self, make_X, gaussian_kernel_fn):
        """Test thin with different store_K modes."""
        X = make_X(n=64, d=2)

        # store_K=True: stores kernel for efficiency (may affect swap behavior)
        result_with_K = kt.thin(
            X,
            m=2,
            split_kernel=gaussian_kernel_fn,
            swap_kernel=gaussian_kernel_fn,
            store_K=True,
            seed=42,
        )

        # store_K=False: doesn't store kernel (lower memory usage)
        result_no_K = kt.thin(
            X,
            m=2,
            split_kernel=gaussian_kernel_fn,
            swap_kernel=gaussian_kernel_fn,
            store_K=False,
            seed=42,
        )

        # Both should produce valid coresets of same size
        # (Note: results may differ because store_K affects swap behavior)
        assert len(result_with_K) == 16
        assert len(result_no_K) == 16
        assert np.all(result_with_K >= 0) and np.all(result_with_K < 64)
        assert np.all(result_no_K >= 0) and np.all(result_no_K < 64)


class TestCrossModuleKernels:
    """Test that kernel functions work consistently across modules."""

    def test_same_kernel_compress_and_kt(self, make_X, gaussian_kernel_fn):
        """Use Gaussian kernel in both compress and kt."""
        X = make_X(n=256, d=2)

        # Compress with Gaussian
        compressed = compress.compress_kt(
            X, b"gaussian", k_params=np.array([1.0]), g=1, seed=42
        )

        # Thin with same kernel
        X_comp = X[compressed]
        thinned_local = kt.thin_X(
            X_comp,
            m=1,
            split_kernel=gaussian_kernel_fn,
            swap_kernel=gaussian_kernel_fn,
            seed=42,
        )

        # Pipeline should complete successfully
        # thin_X with m=1 halves the compressed size
        assert len(thinned_local) == len(X_comp) // 2

    def test_different_kernels_split_vs_swap(self, make_X):
        """Use different kernels for split and swap."""
        X = make_X(n=64, d=2)

        def gaussian(y, X):
            return np.exp(-np.sum((X - y) ** 2, axis=1))

        def laplacian(y, X, lam=1.0):
            return np.exp(-np.sum(np.abs(X - y), axis=1) / lam)

        # Split with Gaussian, swap with Laplacian
        coreset = kt.thin_X(
            X, m=2, split_kernel=gaussian, swap_kernel=laplacian, seed=42
        )

        assert len(coreset) == 16
        assert np.all(coreset >= 0)
        assert np.all(coreset < 64)


class TestHerdingIntegration:
    """Test herding with other modules."""

    def test_herding_then_compress(self, make_X):
        """Run herding then compress the result."""
        X = make_X(n=256, d=2)

        def gaussian(y, X):
            return np.exp(-np.sum((X - y) ** 2, axis=1))

        # Herding
        herded = herding.herding(X, m=2, kernel=gaussian)
        assert len(herded) == 64

        # Compress herded subset
        X_herded = X[herded]
        compressed = compress.compress_kt(
            X_herded, b"gaussian", k_params=np.array([1.0]), g=0, seed=42
        )

        assert len(compressed) == 8  # sqrt(64) = 8

    def test_herding_vs_kt_comparison(self, make_X, gaussian_kernel_fn):
        """Compare herding and KT on same dataset."""
        X = make_X(n=64, d=2)
        from goodpoints.kt import kernel_matrix

        # Herding
        herded = herding.herding(X, m=1, kernel=gaussian_kernel_fn)

        # KT
        K = kernel_matrix(X, gaussian_kernel_fn)
        kt_coreset = kt.thin_K(K, K, m=1, seed=42)

        # Both should produce same-sized output
        assert len(herded) == len(kt_coreset) == 32

        # But potentially different indices (different algorithms)
        # Just verify both are valid
        assert np.all(herded >= 0) and np.all(herded < 64)
        assert np.all(kt_coreset >= 0) and np.all(kt_coreset < 64)
