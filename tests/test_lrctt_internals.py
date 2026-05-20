"""Tests for lrctt() internal logic and edge cases.

This file tests uncovered paths in lrctt():
1. Small r values (r <= 4*four_to_g) - lines 338-339
2. User-specified compression factor (a != 0) - lines 348-352
3. Warning: compression bins < permutation bins - lines 357-362
4. Warning: compression output > input size - lines 369-375
"""

import numpy as np
from goodpoints import ctt
import warnings


class TestLrcttSmallR:
    """Tests for lrctt() with small r values (minimal compression path)."""

    def test_small_r_no_compression_path(self):
        """lrctt with small r uses RFF without compression."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((128, 2))
        X2 = rng.standard_normal((128, 2))

        # Small r (r <= 4*four_to_g) triggers comp_bins = n//four_to_g path
        # With g=0, four_to_g=1, so r <= 4 triggers this
        result = ctt.lrctt(
            X1,
            X2,
            r=4,  # Small r
            g=0,  # four_to_g = 1
            a=0,  # Auto-select compression (triggers if statement)
            B=39,
            s=8,
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        # Should complete without returning None
        assert result is not None
        assert hasattr(result, "rejects")
        assert 0 <= result.rejects <= 1

    def test_very_small_r_g1(self):
        """lrctt with g=1 and small r."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))

        # With g=1, four_to_g=4, so r <= 16 triggers small-r path
        result = ctt.lrctt(
            X1,
            X2,
            r=16,  # r <= 4 * 4^1 = 16
            g=1,
            a=0,
            B=39,
            s=8,
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        assert result is not None
        assert hasattr(result, "rejects")


class TestLrcttUserSpecifiedA:
    """Tests for lrctt() with user-specified compression factor a."""

    def test_explicit_compression_factor(self):
        """lrctt with a != 0 uses user-specified compression."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))

        # a != 0 triggers the else branch (lines 348-352)
        # comp_bins = four_to_g * n * a^2 / r^2
        # With n=256, g=0, a=8, r=16: comp_bins = 1 * 256 * 64 / 256 = 64
        # perm_bins = n // s = 256 // 8 = 32
        # Note: Using integer a to avoid float comp_bins (potential bug in ctt.py)
        result = ctt.lrctt(
            X1,
            X2,
            r=16,  # Smaller r for larger comp_bins
            g=0,
            a=8,  # Integer a to ensure comp_bins is int
            B=39,
            s=8,
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        assert result is not None
        assert hasattr(result, "rejects")

    def test_different_compression_factors(self):
        """Different a values produce valid results."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))

        # Use smaller r=8 so comp_bins is larger
        # perm_bins = 256 // 8 = 32
        # comp_bins = 1 * 256 * a^2 / 64
        # Need a^2 >= 8 → a >= 2.83 for comp_bins >= 32
        # Note: Using integer a values to avoid float comp_bins
        for a_val in [3, 4, 5, 6]:
            result = ctt.lrctt(
                X1,
                X2,
                r=8,  # Small r
                g=0,
                a=a_val,
                B=39,
                s=8,
                kernel="gauss",
                null_seed=0,
                statistic_seed=1,
            )

            # All should produce valid results
            assert result is not None
            assert hasattr(result, "rejects")


class TestLrcttWarnings:
    """Tests for lrctt() warning paths that return None."""

    def test_comp_bins_less_than_perm_bins_returns_none(self):
        """lrctt returns None when compression bins < permutation bins."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))

        # Force comp_bins < perm_bins by using large r and small a
        # comp_bins = four_to_g * n * a^2 / r^2
        # perm_bins = n // s
        # With n=64, s=8: perm_bins = 8
        # With g=0, a=1, r=100: comp_bins = 1 * 64 * 1 / 10000 ≈ 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ctt.lrctt(
                X1,
                X2,
                r=100,  # Large r
                g=0,
                a=1.0,  # Small a
                B=39,
                s=8,
                kernel="gauss",
                null_seed=0,
                statistic_seed=1,
            )

            # Should return None and issue warning
            assert result is None
            assert len(w) > 0
            assert "comp_bins" in str(w[0].message)
            assert "perm_bins" in str(w[0].message)

    def test_compression_output_exceeds_input_returns_none(self):
        """lrctt returns None when compression output > input size."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((32, 2))
        X2 = rng.standard_normal((32, 2))

        # Force compression output > input
        # out_thresh = comp_bins * 4^g
        # comp_bins = four_to_g * n * a^2 / r^2
        # With n=32, g=2 (four_to_g=16), a=10, r=8:
        # comp_bins = 16 * 32 * 100 / 64 = 800
        # out_thresh = 800 * 16 = 12800 >> 32

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ctt.lrctt(
                X1,
                X2,
                r=8,  # Small r
                g=2,  # Large g
                a=10.0,  # Large a
                B=39,
                s=4,
                kernel="gauss",
                null_seed=0,
                statistic_seed=1,
            )

            # Should return None and issue warning
            assert result is None
            assert len(w) > 0
            assert "compression output size" in str(w[0].message)
            assert "exceeds input size" in str(w[0].message)

    def test_borderline_valid_parameters(self):
        """lrctt works with parameters just within valid range."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))

        # Parameters that should just barely work
        # perm_bins = 256 // 16 = 16
        # comp_bins = 1 * 256 * 16 / 256 = 16 (exactly equal)
        # This should just barely work
        # Note: Using integer a to avoid float comp_bins
        result = ctt.lrctt(
            X1,
            X2,
            r=16,  # r^2 = 256
            g=0,
            a=4,  # Integer a, a^2 = 16
            B=39,
            s=16,  # Large s to make perm_bins small
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        # Should not return None
        assert result is not None
        assert hasattr(result, "rejects")
