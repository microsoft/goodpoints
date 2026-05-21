"""Property-based tests using Hypothesis library.

These tests use Hypothesis to generate random test cases and verify
that key properties hold across a wide range of inputs.

To run these tests, install hypothesis:
    uv pip install hypothesis

Properties tested:
1. Output size correctness (deterministic halving)
2. Valid index ranges (0 <= indices < n)
3. Uniqueness when requested
4. Deterministic behavior (same seed → same output)
5. Monotonicity properties (refinement doesn't worsen MMD)
"""

import numpy as np
import pytest

# Try importing hypothesis; skip all tests if not available
hypothesis = pytest.importorskip("hypothesis", reason="Hypothesis not installed")
from hypothesis import given, strategies as st, settings  # noqa: E402
from hypothesis.extra.numpy import arrays  # noqa: E402

from goodpoints import kt, compress, herding  # noqa: E402


# Custom strategies for goodpoints
@st.composite
def valid_data_arrays(draw, min_size=16, max_size=256):
    """Strategy for generating valid data arrays."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    # Ensure n is power of 2 for most algorithms
    n = 2 ** int(np.log2(n))
    d = draw(st.integers(min_value=1, max_value=5))

    return draw(
        arrays(
            dtype=np.float64,
            shape=(n, d),
            elements=st.floats(
                min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
            ),
        )
    )


@st.composite
def valid_halving_rounds(draw, n):
    """Strategy for generating valid m values given n."""
    max_m = int(np.log2(n))
    return draw(st.integers(min_value=0, max_value=max_m))


class TestOutputSizeProperties:
    """Property tests for output size correctness."""

    @given(st.integers(min_value=16, max_value=256))
    @settings(max_examples=20, deadline=1000)
    def test_thin_K_output_size_property(self, n):
        """thin_K output size = floor(n / 2^m) for any valid n and m."""
        # Ensure n is power of 2
        n = 2 ** int(np.log2(n))
        max_m = int(np.log2(n))

        for m in range(0, max_m + 1):
            # Create random kernel matrix
            rng = np.random.default_rng(42)
            X = rng.standard_normal((n, 2))
            from goodpoints.kt import kernel_matrix

            def gaussian(y, X):

                return np.exp(-np.sum((X - y) ** 2, axis=1))

            K = kernel_matrix(X, gaussian)

            # Run thin_K
            coreset = kt.thin_K(K, K, m=m, seed=42)

            # Check size property
            expected_size = n // (2**m)
            assert len(coreset) == expected_size, (
                f"n={n}, m={m}: expected {expected_size}, got {len(coreset)}"
            )

    @given(st.integers(min_value=64, max_value=256))
    @settings(max_examples=10, deadline=2000)
    def test_compress_output_size_property(self, n):
        """compress_kt output size = sqrt(n) * 2^g for power-of-4 inputs."""
        # Ensure n is power of 4
        n = 4 ** int(np.log2(n) / 2)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))

        for g in [0, 1]:
            compressed = compress.compress_kt(
                X, b"gaussian", k_params=np.array([1.0]), g=g, seed=42
            )

            # Expected size for compress_kt: sqrt(n) * 2^g
            expected_size = int(np.sqrt(n) * (2**g))
            assert len(compressed) == expected_size, (
                f"n={n}, g={g}: expected {expected_size}, got {len(compressed)}"
            )


class TestValidRangeProperties:
    """Property tests for index validity."""

    @given(valid_data_arrays(min_size=32, max_size=128))
    @settings(max_examples=20, deadline=1000)
    def test_indices_in_valid_range(self, X):
        """All output indices must be in [0, n)."""
        n = X.shape[0]

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        # Determine valid m
        max_m = int(np.log2(n))
        m = min(2, max_m)

        # Run kernel thinning
        coreset = kt.thin_X(
            X, m=m, split_kernel=gaussian, swap_kernel=gaussian, seed=42
        )

        # All indices must be in valid range
        assert np.all(coreset >= 0), "Some indices are negative"
        assert np.all(coreset < n), f"Some indices >= n={n}"

    @given(st.integers(min_value=64, max_value=256))
    @settings(max_examples=10, deadline=2000)
    def test_compress_indices_valid(self, n):
        """compress_kt indices are in valid range."""
        # Ensure n is power of 4
        n = 4 ** int(np.log2(n) / 2)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))

        compressed = compress.compress_kt(
            X, b"gaussian", k_params=np.array([1.0]), g=0, seed=42
        )

        assert np.all(compressed >= 0)
        assert np.all(compressed < n)


class TestUniquenessProperties:
    """Property tests for uniqueness constraints."""

    @given(st.integers(min_value=32, max_value=128))
    @settings(max_examples=15, deadline=1500)
    def test_unique_parameter_enforced(self, n):
        """When unique=True, output contains no duplicates."""
        n = 2 ** int(np.log2(n))

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))
        from goodpoints.kt import kernel_matrix

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        K = kernel_matrix(X, gaussian)

        # Create initial coreset (may have duplicates)
        initial = np.arange(n // 4)

        # Refine with unique=True
        refined = kt.refine_K(K, initial, unique=True)

        # Should have no duplicates
        assert len(set(refined)) == len(refined), (
            f"Found duplicates: {len(refined)} indices but {len(set(refined))} unique"
        )


class TestDeterminismProperties:
    """Property tests for deterministic behavior."""

    @given(
        st.integers(min_value=32, max_value=128),
        st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=15, deadline=1500)
    def test_same_seed_same_output(self, n, seed):
        """Same seed produces same output."""
        n = 2 ** int(np.log2(n))

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        max_m = int(np.log2(n))
        m = min(2, max_m)

        # Run twice with same seed
        coreset1 = kt.thin_X(
            X, m=m, split_kernel=gaussian, swap_kernel=gaussian, seed=seed
        )
        coreset2 = kt.thin_X(
            X, m=m, split_kernel=gaussian, swap_kernel=gaussian, seed=seed
        )

        # Should be identical
        np.testing.assert_array_equal(coreset1, coreset2)

    @given(st.integers(min_value=64, max_value=256))
    @settings(max_examples=10, deadline=2000)
    def test_compress_deterministic(self, n):
        """compress_kt is deterministic with same seed."""
        n = 4 ** int(np.log2(n) / 2)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))

        result1 = compress.compress_kt(
            X, b"gaussian", k_params=np.array([1.0]), g=0, seed=42
        )
        result2 = compress.compress_kt(
            X, b"gaussian", k_params=np.array([1.0]), g=0, seed=42
        )

        np.testing.assert_array_equal(result1, result2)


class TestMonotonicityProperties:
    """Property tests for optimization monotonicity."""

    @given(st.integers(min_value=64, max_value=128))
    @settings(max_examples=10, deadline=2000)
    def test_refine_doesnt_worsen_mmd(self, n):
        """refine_K should not increase MMD (modulo numerical error)."""
        n = 2 ** int(np.log2(n))

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))
        from goodpoints.kt import kernel_matrix

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        K = kernel_matrix(X, gaussian)
        meanK = K.mean(axis=1)

        # Create initial coreset
        initial = np.arange(0, n, 4)

        # Refine
        refined = kt.refine_K(K, initial, meanK=meanK)

        # Compute relative MMD for both
        def rel_mmd(coreset):
            return np.mean(K[np.ix_(coreset, coreset)]) - 2 * np.mean(meanK[coreset])

        mmd_initial = rel_mmd(initial)
        mmd_refined = rel_mmd(refined)

        # Refinement should not worsen MMD (with numerical tolerance)
        assert mmd_refined <= mmd_initial + 1e-10, (
            f"Refinement increased MMD: {mmd_initial:.6e} → {mmd_refined:.6e}"
        )


class TestKernelInvarianceProperties:
    """Property tests for kernel function behavior."""

    @given(valid_data_arrays(min_size=32, max_size=64))
    @settings(max_examples=10, deadline=1500)
    def test_kernel_scaling_invariance(self, X):
        """Scaling kernel uniformly doesn't change coreset selection."""
        n = X.shape[0]

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        def scaled_gaussian(y, X):

            return 2.0 * np.exp(-np.sum((X - y) ** 2, axis=1))

        max_m = int(np.log2(n))
        m = min(1, max_m)

        # Run with both kernels
        coreset1 = kt.thin_X(
            X, m=m, split_kernel=gaussian, swap_kernel=gaussian, seed=42
        )
        coreset2 = kt.thin_X(
            X, m=m, split_kernel=scaled_gaussian, swap_kernel=scaled_gaussian, seed=42
        )

        # Should produce same coreset (scaling doesn't change optimization)
        np.testing.assert_array_equal(coreset1, coreset2)


class TestHerdingProperties:
    """Property tests for kernel herding."""

    @given(st.integers(min_value=32, max_value=128))
    @settings(max_examples=15, deadline=1500)
    def test_herding_output_size(self, n):
        """herding produces correct output size."""
        n = 2 ** int(np.log2(n))

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        max_m = int(np.log2(n))
        m = min(2, max_m)

        herded = herding.herding(X, m=m, kernel=gaussian)

        expected_size = n // (2**m)
        assert len(herded) == expected_size

    @given(st.integers(min_value=32, max_value=128))
    @settings(max_examples=15, deadline=1500)
    def test_herding_unique_indices(self, n):
        """herding with unique=True produces no duplicates."""
        n = 2 ** int(np.log2(n))

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        max_m = int(np.log2(n))
        m = min(2, max_m)

        herded = herding.herding(X, m=m, kernel=gaussian, unique=True)

        # Should have no duplicates
        assert len(set(herded)) == len(herded)


class TestCompressionProperties:
    """Property tests for compression algorithms."""

    @given(st.integers(min_value=16, max_value=256))
    @settings(max_examples=15, deadline=2000)
    def test_compress_indices_subset(self, n):
        """Compression returns valid subset of input indices."""
        n = 4 ** int(np.log2(n) / 2)  # Nearest power of 4

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))

        compressed = compress.compress_kt(
            X, b"gaussian", k_params=np.array([1.0]), g=0, seed=42
        )

        # All indices should be unique
        assert len(set(compressed)) == len(compressed)

        # All indices in valid range
        assert np.all(compressed >= 0)
        assert np.all(compressed < n)

    @given(st.integers(min_value=16, max_value=128))
    @settings(max_examples=10, deadline=3000)
    def test_compresspp_deterministic(self, n):
        """compresspp_kt is deterministic with same seed."""
        n = 4 ** int(np.log2(n) / 2)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))

        result1 = compress.compresspp_kt(
            X, b"gaussian", k_params=np.array([1.0]), g=0, num_bins=4, seed=42
        )
        result2 = compress.compresspp_kt(
            X, b"gaussian", k_params=np.array([1.0]), g=0, num_bins=4, seed=42
        )

        np.testing.assert_array_equal(result1, result2)


class TestSplitSwapProperties:
    """Property tests for split and swap operations."""

    @given(st.integers(min_value=32, max_value=128))
    @settings(max_examples=15, deadline=2000)
    def test_split_partition_property(self, n):
        """split_K partitions indices into disjoint sets."""
        n = 2 ** int(np.log2(n))

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))
        from goodpoints.kt import kernel_matrix

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        K = kernel_matrix(X, gaussian)

        max_m = int(np.log2(n))
        m = min(2, max_m)

        coresets = kt.split_K(K, m=m, seed=42)

        # Coresets should partition the indices
        all_indices = np.sort(np.concatenate(coresets))
        np.testing.assert_array_equal(all_indices, np.arange(n))

        # Coresets should be disjoint
        for i in range(len(coresets)):
            for j in range(i + 1, len(coresets)):
                intersection = set(coresets[i]) & set(coresets[j])
                assert len(intersection) == 0, (
                    f"Coresets {i} and {j} overlap: {intersection}"
                )

    @given(st.integers(min_value=32, max_value=128))
    @settings(max_examples=10, deadline=2000)
    def test_swap_produces_valid_output(self, n):
        """swap_K produces valid coreset indices."""
        n = 2 ** int(np.log2(n))

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))
        from goodpoints.kt import kernel_matrix

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        K = kernel_matrix(X, gaussian)
        meanK = K.mean(axis=1)

        # Create initial coresets via split
        max_m = int(np.log2(n))
        m = min(2, max_m)
        coresets = kt.split_K(K, m=m, seed=42)

        # Apply swap
        swapped = kt.swap_K(K, coresets, meanK=meanK)

        # Swap should preserve size
        expected_size = n // (2**m)
        assert len(swapped) == expected_size

        # All indices should be valid
        assert np.all(swapped >= 0)
        assert np.all(swapped < n)


class TestEdgeCaseProperties:
    """Property tests for edge cases and boundary conditions."""

    @given(
        st.integers(min_value=8, max_value=64), st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=20, deadline=1500)
    def test_thin_with_varying_dimensions(self, n, d):
        """thin_X works with any dimension."""
        n = 2 ** int(np.log2(n))

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, d))

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        max_m = int(np.log2(n))
        m = min(2, max_m)

        result = kt.thin_X(X, m=m, split_kernel=gaussian, swap_kernel=gaussian, seed=42)

        expected_size = n // (2**m)
        assert len(result) == expected_size
        assert np.all(result >= 0)
        assert np.all(result < n)

    @given(st.integers(min_value=16, max_value=128))
    @settings(max_examples=15, deadline=1500)
    def test_refine_preserves_coreset_size(self, n):
        """refine_K maintains coreset size."""
        n = 2 ** int(np.log2(n))

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))
        from goodpoints.kt import kernel_matrix

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        K = kernel_matrix(X, gaussian)

        # Create initial coreset
        initial_size = n // 4
        initial = np.arange(0, n, 4)[:initial_size]

        # Refine
        refined = kt.refine_K(K, initial)

        # Size should be preserved
        assert len(refined) == len(initial)

        # All indices should be valid
        assert np.all(refined >= 0)
        assert np.all(refined < n)

    @given(st.integers(min_value=16, max_value=128))
    @settings(max_examples=10, deadline=2000)
    def test_mean0_mode_produces_valid_output(self, n):
        """Functions with mean0=True produce valid output."""
        n = 2 ** int(np.log2(n))

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 2))
        from goodpoints.kt import kernel_matrix

        def gaussian(y, X):

            return np.exp(-np.sum((X - y) ** 2, axis=1))

        K = kernel_matrix(X, gaussian)

        # Test thin_K with mean0=True
        result = kt.thin_K(K, K, m=1, seed=42, mean0=True)

        expected_size = n // 2
        assert len(result) == expected_size
        assert np.all(result >= 0)
        assert np.all(result < n)
