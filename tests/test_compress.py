"""Tests for compression algorithms (compress.py) using factory fixtures."""

import numpy as np
from goodpoints.compress import (
    largest_power_of_four,
    compress_kt,
    compresspp_kt,
    compress,
    symmetrize,
)


class TestLargestPowerOfFour:
    def test_known_values(self):
        assert largest_power_of_four(1) == 1
        assert largest_power_of_four(4) == 4
        assert largest_power_of_four(16) == 16
        assert largest_power_of_four(64) == 64
        assert largest_power_of_four(256) == 256

    def test_non_power_of_four(self):
        assert largest_power_of_four(3) == 1
        assert largest_power_of_four(5) == 4
        assert largest_power_of_four(15) == 4
        assert largest_power_of_four(63) == 16
        assert largest_power_of_four(100) == 64


class TestCompressKt:
    def test_output_size_pow4(self, make_X):
        X = make_X(64, 2)
        indices = compress_kt(X, b"gaussian", g=0, seed=42)
        assert len(indices) == int(np.sqrt(64))

    def test_valid_indices(self, make_X):
        X = make_X(64, 2)
        indices = compress_kt(X, b"gaussian", g=0, seed=42)
        assert np.all(indices >= 0)
        assert np.all(indices < 64)

    def test_deterministic(self, make_X):
        X = make_X(64, 2)
        i1 = compress_kt(X, b"gaussian", seed=42)
        i2 = compress_kt(X, b"gaussian", seed=42)
        np.testing.assert_array_equal(i1, i2)

    def test_oversampling_g1(self, make_X):
        X = make_X(64, 2)
        indices = compress_kt(X, b"gaussian", g=1, seed=42)
        expected_size = min(64, int(np.sqrt(64) * 2))
        assert len(indices) == expected_size

    def test_non_power_of_four_n(self, rng):
        X = rng.standard_normal((50, 2))
        indices = compress_kt(X, b"gaussian", g=0, seed=42)
        n_prime = largest_power_of_four(50)
        assert len(indices) == int(np.sqrt(n_prime))

    def test_multi_bin(self, make_X):
        X = make_X(64, 2)
        indices = compress_kt(X, b"gaussian", g=0, num_bins=4, seed=42)
        expected_size = min(64, int(np.sqrt(64 * 4)))
        assert len(indices) == expected_size

    def test_skip_swap(self, make_X):
        X = make_X(64, 2)
        indices = compress_kt(X, b"gaussian", g=0, skip_swap=True, seed=42)
        assert len(indices) == int(np.sqrt(64))
        assert np.all(indices >= 0)
        assert np.all(indices < 64)

    def test_scalar_k_params(self, make_X):
        X = make_X(64, 2)
        indices = compress_kt(X, b"gaussian", k_params=2.0, g=0, seed=42)
        assert len(indices) == int(np.sqrt(64))


class TestCompresspKt:
    def test_output_size(self, make_X):
        X = make_X(64, 2)
        indices = compresspp_kt(X, b"gaussian", g=0, seed=42)
        n_prime = largest_power_of_four(64)
        assert len(indices) == int(np.sqrt(n_prime))

    def test_valid_indices(self, make_X):
        X = make_X(64, 2)
        indices = compresspp_kt(X, b"gaussian", g=0, seed=42)
        assert np.all(indices >= 0)
        assert np.all(indices < 64)

    def test_deterministic(self, make_X):
        X = make_X(64, 2)
        i1 = compresspp_kt(X, b"gaussian", g=0, seed=42)
        i2 = compresspp_kt(X, b"gaussian", g=0, seed=42)
        np.testing.assert_array_equal(i1, i2)

    def test_mean0(self, make_X):
        X = make_X(64, 2)
        indices = compresspp_kt(X, b"gaussian", g=0, seed=42, mean0=True)
        assert len(indices) == int(np.sqrt(64))

    def test_n_not_power_of_four(self, rng):
        X = rng.standard_normal((17, 2))
        coreset = compresspp_kt(X, b"gaussian", seed=42)
        assert len(coreset) > 0
        assert np.all(coreset >= 0)
        assert np.all(coreset < 17)

    def test_n_not_power_of_four_with_g1(self, rng):
        X = rng.standard_normal((17, 2))
        coreset = compresspp_kt(X, b"gaussian", g=1, seed=42)
        assert len(coreset) > 0
        assert np.all(coreset >= 0)
        assert np.all(coreset < 17)

    def test_skip_swap_true(self, make_X):
        X = make_X(64, 2)
        coreset = compresspp_kt(X, b"gaussian", skip_swap=True, seed=42)
        assert len(coreset) > 0
        assert np.all(coreset >= 0)
        assert np.all(coreset < 64)

    def test_skip_swap_with_g1(self, make_X):
        X = make_X(64, 2)
        coreset = compresspp_kt(X, b"gaussian", g=1, skip_swap=True, seed=42)
        assert len(coreset) > 0

    def test_skip_swap_direct_thin_path(self, rng):
        # Use small n where 2^m >= sqrt(n) to trigger direct thin path
        X = rng.standard_normal((4, 2))
        coreset = compresspp_kt(X, b"gaussian", skip_swap=True, seed=42)
        assert len(coreset) > 0
        assert np.all(coreset >= 0)
        assert np.all(coreset < 4)

    def test_mean0_with_skip_swap(self, make_X):
        X = make_X(64, 2)
        coreset = compresspp_kt(X, b"gaussian", mean0=True, skip_swap=True, seed=42)
        assert len(coreset) > 0


class TestCompress:
    def test_with_custom_halve(self, rng):
        X = rng.standard_normal((16, 2))

        def simple_halve(Y):
            return np.arange(Y.shape[0] // 2)

        indices = compress(X, simple_halve, g=0)
        assert len(indices) == int(np.sqrt(16))
        assert np.all(indices >= 0)
        assert np.all(indices < 16)

    def test_g1_oversampling(self, rng):
        X = rng.standard_normal((64, 2))

        def simple_halve(Y):
            return np.arange(Y.shape[0] // 2)

        indices = compress(X, simple_halve, g=1)
        assert len(indices) == int(np.sqrt(64) * 2)


class TestSymmetrize:
    def test_returns_half(self, rng):
        X = rng.standard_normal((16, 2))

        def halve(Y):
            return np.arange(Y.shape[0] // 2)

        sym_halve = symmetrize(halve, seed=42)
        result = sym_halve(X)
        assert len(result) == 8

    def test_complement_coverage(self, rng):
        X = rng.standard_normal((8, 2))

        def halve(Y):
            return np.arange(Y.shape[0] // 2)

        outputs = set()
        for seed in range(100):
            sym_halve = symmetrize(halve, seed=seed)
            result = sym_halve(X)
            outputs.add(tuple(result))
        assert len(outputs) == 2
