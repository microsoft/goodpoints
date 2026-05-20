"""Tests for kernel herding (herding.py) using factory fixtures."""

import numpy as np
from goodpoints.herding import herding
import pytest


class TestHerding:
    @pytest.mark.parametrize(
        "n,m,expected_size",
        [
            (16, 1, 8),
            (32, 1, 16),
            (64, 2, 16),
        ],
    )
    def test_output_size(self, make_X, gaussian_kernel_fn, n, m, expected_size):
        X = make_X(n, 2)
        coreset = herding(X, m=m, kernel=gaussian_kernel_fn)
        assert len(coreset) == expected_size

    def test_valid_indices(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        coreset = herding(X, m=1, kernel=gaussian_kernel_fn)
        assert np.all(coreset >= 0)
        assert np.all(coreset < 16)

    def test_unique_no_duplicates(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        coreset = herding(X, m=1, kernel=gaussian_kernel_fn)
        assert len(set(coreset)) == len(coreset)

    def test_deterministic(self, make_X, gaussian_kernel_fn):
        X = make_X(16, 2)
        c1 = herding(X, m=1, kernel=gaussian_kernel_fn)
        c2 = herding(X, m=1, kernel=gaussian_kernel_fn)
        np.testing.assert_array_equal(c1, c2)
