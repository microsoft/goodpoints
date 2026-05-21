"""Tests for kernel thinning (kt.py) using factory fixtures."""

import numpy as np
from goodpoints import kt
import pytest


class TestLargestPowerOfTwo:
    def test_known_values(self):
        assert kt.largest_power_of_two(1) == 0
        assert kt.largest_power_of_two(2) == 1
        assert kt.largest_power_of_two(4) == 2
        assert kt.largest_power_of_two(8) == 3
        assert kt.largest_power_of_two(16) == 4

    def test_odd_numbers(self):
        assert kt.largest_power_of_two(3) == 0
        assert kt.largest_power_of_two(5) == 0
        assert kt.largest_power_of_two(7) == 0

    def test_mixed(self):
        assert kt.largest_power_of_two(6) == 1
        assert kt.largest_power_of_two(12) == 2
        assert kt.largest_power_of_two(24) == 3


class TestKernelMatrix:
    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_shape(self, make_X, gaussian_kernel_fn, n):
        X = make_X(n, 2)
        K = kt.kernel_matrix(X, gaussian_kernel_fn)
        assert K.shape == (n, n)

    def test_symmetry(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        K = kt.kernel_matrix(X, gaussian_kernel_fn)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_diagonal(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        K = kt.kernel_matrix(X, gaussian_kernel_fn)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)


class TestKernelMatrixRowMean:
    def test_shape(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        meanK = kt.kernel_matrix_row_mean(X, gaussian_kernel_fn)
        assert meanK.shape == (16,)

    def test_matches_kernel_matrix(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        meanK = kt.kernel_matrix_row_mean(X, gaussian_kernel_fn)
        K = kt.kernel_matrix(X, gaussian_kernel_fn)
        np.testing.assert_allclose(meanK, K.mean(axis=1), atol=1e-12)


class TestOptHalve4K:
    def test_returns_two_indices(self, make_K_gauss):
        K4 = make_K_gauss(16)[:4, :4].copy()
        coreset = kt.opt_halve4_K(K4, seed=42)
        assert len(coreset) == 2

    def test_valid_indices(self, make_K_gauss):
        K4 = make_K_gauss(16)[:4, :4].copy()
        coreset = kt.opt_halve4_K(K4, seed=42)
        assert np.all(coreset >= 0)
        assert np.all(coreset < 4)

    def test_distinct_indices(self, make_K_gauss):
        K4 = make_K_gauss(16)[:4, :4].copy()
        coreset = kt.opt_halve4_K(K4, seed=42)
        assert len(set(coreset)) == 2

    def test_deterministic(self, make_K_gauss):
        K4 = make_K_gauss(16)[:4, :4].copy()
        c1 = kt.opt_halve4_K(K4, seed=99)
        c2 = kt.opt_halve4_K(K4, seed=99)
        np.testing.assert_array_equal(c1, c2)


class TestHalveK:
    def test_shape(self, make_K_gauss):
        K = make_K_gauss(16)
        coresets = kt.halve_K(K, seed=42)
        assert coresets.shape == (2, 8)

    def test_union_is_permutation(self, make_K_gauss):
        K = make_K_gauss(16)
        coresets = kt.halve_K(K, seed=42)
        union = np.sort(np.concatenate(coresets))
        np.testing.assert_array_equal(union, np.arange(16))

    def test_deterministic(self, make_K_gauss):
        K = make_K_gauss(16)
        c1 = kt.halve_K(K, seed=42)
        c2 = kt.halve_K(K, seed=42)
        np.testing.assert_array_equal(c1, c2)


class TestSplitK:
    def test_m0_returns_all(self, make_K_gauss):
        K = make_K_gauss(16)
        coresets = kt.split_K(K, m=0)
        assert coresets.shape == (1, 16)
        np.testing.assert_array_equal(coresets[0], np.arange(16))

    def test_m1_consistent_with_halve(self, make_K_gauss):
        K = make_K_gauss(16)
        coresets_split = kt.split_K(K, m=1, seed=42)
        coresets_halve = kt.halve_K(K, seed=42)
        np.testing.assert_array_equal(coresets_split, coresets_halve)

    @pytest.mark.parametrize("m,expected_coresets", [(1, 2), (2, 4), (3, 8)])
    def test_coreset_count(self, make_K_gauss, m, expected_coresets):
        K = make_K_gauss(16)
        coresets = kt.split_K(K, m=m, seed=42)
        assert coresets.shape[0] == expected_coresets

    def test_valid_indices(self, make_K_gauss):
        K = make_K_gauss(16)
        coresets = kt.split_K(K, m=2, seed=42)
        assert np.all(coresets >= 0)
        assert np.all(coresets < 16)

    def test_deterministic(self, make_K_gauss):
        K = make_K_gauss(16)
        c1 = kt.split_K(K, m=2, seed=42)
        c2 = kt.split_K(K, m=2, seed=42)
        np.testing.assert_array_equal(c1, c2)


class TestThinK:
    def test_m0_returns_all(self, make_K_gauss):
        K = make_K_gauss(16)
        coreset = kt.thin_K(K, K, m=0)
        np.testing.assert_array_equal(coreset, np.arange(16))

    @pytest.mark.parametrize(
        "n,m,expected_size",
        [
            (16, 1, 8),
            (16, 2, 4),
            (32, 1, 16),
            (64, 2, 16),
        ],
    )
    def test_output_size(self, make_K_gauss, n, m, expected_size):
        K = make_K_gauss(n)
        coreset = kt.thin_K(K, K, m=m, seed=42)
        assert len(coreset) == expected_size

    def test_valid_indices(self, make_K_gauss):
        K = make_K_gauss(16)
        coreset = kt.thin_K(K, K, m=1, seed=42)
        assert np.all(coreset >= 0)
        assert np.all(coreset < 16)

    def test_deterministic(self, make_K_gauss):
        K = make_K_gauss(16)
        c1 = kt.thin_K(K, K, m=1, seed=42)
        c2 = kt.thin_K(K, K, m=1, seed=42)
        np.testing.assert_array_equal(c1, c2)

    def test_unique(self, make_K_gauss):
        K = make_K_gauss(16)
        coreset = kt.thin_K(K, K, m=1, seed=42, unique=True)
        assert len(set(coreset)) == len(coreset)

    def test_mean0(self, make_K_gauss):
        K = make_K_gauss(16)
        coreset = kt.thin_K(K, K, m=1, seed=42, mean0=True)
        assert len(coreset) == 8


class TestSplitX:
    def test_m0_returns_all(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        coresets = kt.split_X(X, m=0, kernel=gaussian_kernel_fn)
        assert coresets.shape == (1, 16)

    def test_m1_shape(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        coresets = kt.split_X(X, m=1, kernel=gaussian_kernel_fn, seed=42)
        assert coresets.shape == (2, 8)


class TestSwapK:
    def test_correct_size(self, make_K_gauss):
        K = make_K_gauss(16)
        coresets = kt.split_K(K, m=1, seed=42)
        coreset = kt.swap_K(K, coresets)
        assert len(coreset) == 8

    def test_valid_indices(self, make_K_gauss):
        K = make_K_gauss(16)
        coresets = kt.split_K(K, m=1, seed=42)
        coreset = kt.swap_K(K, coresets)
        assert np.all(coreset >= 0)
        assert np.all(coreset < 16)


class TestBestK:
    def test_returns_valid_coreset(self, make_K_gauss):
        K = make_K_gauss(16)
        coresets = kt.split_K(K, m=1, seed=42)
        best = kt.best_K(K, coresets)
        assert len(best) == 8
        assert np.all(best >= 0)
        assert np.all(best < 16)

    def test_mean0_mode(self, make_K_gauss):
        K = make_K_gauss(16)
        coresets = kt.split_K(K, m=1, seed=42)
        best = kt.best_K(K, coresets, mean0=True)
        assert len(best) == 8


class TestRefineK:
    def test_preserves_size(self, make_K_gauss):
        K = make_K_gauss(16)
        initial = np.arange(8)
        refined = kt.refine_K(K, initial)
        assert len(refined) == 8

    def test_unique_no_duplicates(self, make_K_gauss):
        K = make_K_gauss(16)
        initial = np.arange(8)
        refined = kt.refine_K(K, initial, unique=True)
        assert len(set(refined)) == len(refined)

    def test_does_not_worsen_mmd(self, make_K_gauss):
        K = make_K_gauss(64)
        initial = np.arange(0, 64, 2)
        meanK = K.mean(axis=1)
        refined = kt.refine_K(K, initial, meanK=meanK)

        def rel_mmd(coreset):
            return np.mean(K[np.ix_(coreset, coreset)]) - 2 * np.mean(meanK[coreset])

        assert rel_mmd(refined) <= rel_mmd(initial) + 1e-10


class TestSplitWithStoreK:
    def test_store_k_true(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        coresets = kt.split(X, m=1, kernel=gaussian_kernel_fn, store_K=True, seed=42)
        assert coresets.shape == (2, 8)

    def test_store_k_false(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        coresets = kt.split(X, m=1, kernel=gaussian_kernel_fn, store_K=False, seed=42)
        assert coresets.shape == (2, 8)


class TestSwapWithStoreK:
    def test_store_k_true(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        coresets = kt.split(X, m=1, kernel=gaussian_kernel_fn, seed=42)
        coreset = kt.swap(X, coresets, kernel=gaussian_kernel_fn, store_K=True)
        assert len(coreset) == 8

    def test_store_k_false(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        coresets = kt.split(X, m=1, kernel=gaussian_kernel_fn, seed=42)
        coreset = kt.swap(X, coresets, kernel=gaussian_kernel_fn, store_K=False)
        assert len(coreset) == 8


class TestSquaredEmpRelMmdX:
    def test_with_meanK_none(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        coreset = np.arange(8)
        mmd = kt.squared_emp_rel_mmd_X(X, coreset, gaussian_kernel_fn, meanK=None)
        assert isinstance(mmd, (float, np.floating))

    def test_with_meanK_provided(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        coreset = np.arange(8)
        meanK = kt.kernel_matrix_row_mean(X, gaussian_kernel_fn)
        mmd = kt.squared_emp_rel_mmd_X(X, coreset, gaussian_kernel_fn, meanK=meanK)
        assert isinstance(mmd, (float, np.floating))
