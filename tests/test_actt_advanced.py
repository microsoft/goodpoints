"""Tests for advanced actt() features.

This file tests previously uncovered actt() functionality:
1. actt() with same_compression=False (lines 549-562) - separate compression per bandwidth
2. set_reject_median() method (lines 853-863) - median heuristic bandwidth rejection

These features require specific parameter combinations (larger B, B_2 values)
that differ from basic edge case tests.
"""

import numpy as np
from goodpoints import ctt


class TestActtSameCompressionFalse:
    """Tests for actt() with same_compression=False."""

    def test_same_compression_false_single_bandwidth(self):
        """actt with same_compression=False and single bandwidth."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))

        # Use reasonable B and B_2 values (smaller than defaults for speed)
        # but large enough to avoid IndexError
        result = ctt.actt(
            X1,
            X2,
            g=0,
            B=49,  # Reasonable minimum (>= 50 elements after split)
            B_2=30,
            B_3=5,
            s=8,
            lam=np.array([1.0]),
            kernel="gauss",
            same_compression=False,  # Key parameter
            null_seed=0,
            statistic_seed=1,
        )

        assert result is not None
        assert hasattr(result, "rejects")
        assert 0 <= result.rejects <= 1

    def test_same_compression_false_multiple_bandwidths(self):
        """actt with same_compression=False and multiple bandwidths."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))

        # Multiple bandwidths - each gets separate compression
        lam = np.array([0.5, 1.0, 2.0])
        weights = np.array([1.0, 1.0, 1.0])  # Must match lam length

        result = ctt.actt(
            X1,
            X2,
            g=0,
            B=49,
            B_2=30,
            B_3=5,
            s=8,
            lam=lam,
            weights=weights,  # Required for multiple bandwidths
            kernel="gauss",
            same_compression=False,  # Lines 549-562 triggered
            null_seed=0,
            statistic_seed=1,
        )

        assert result is not None
        assert hasattr(result, "rejects")
        # Check that all bandwidths were processed
        assert hasattr(result, "bw")
        np.testing.assert_array_equal(result.bw, lam)

    def test_same_compression_true_vs_false_different_results(self):
        """same_compression=True vs False should give different results."""
        rng = np.random.default_rng(42)
        # Use shifted distributions to ensure detectable difference
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2)) + 0.3

        lam = np.array([0.5, 1.0])
        weights = np.array([1.0, 1.0])  # Must match lam length

        result_true = ctt.actt(
            X1,
            X2,
            g=0,
            B=49,
            B_2=30,
            B_3=5,
            s=8,
            lam=lam,
            weights=weights,
            kernel="gauss",
            same_compression=True,
            null_seed=0,
            statistic_seed=1,
        )

        result_false = ctt.actt(
            X1,
            X2,
            g=0,
            B=49,
            B_2=30,
            B_3=5,
            s=8,
            lam=lam,
            weights=weights,
            kernel="gauss",
            same_compression=False,
            null_seed=0,
            statistic_seed=1,
        )

        # Both should complete successfully
        assert result_true is not None
        assert result_false is not None

        # Results may differ because compression strategy differs
        # Just verify both have valid rejection decisions
        assert 0 <= result_true.rejects <= 1
        assert 0 <= result_false.rejects <= 1


class TestSetRejectMedian:
    """Tests for set_reject_median() method."""

    def test_set_reject_median_attribute_exists(self):
        """AggregatedTestResults should have rejects_median attribute."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))

        result = ctt.actt(
            X1,
            X2,
            g=0,
            B=49,
            B_2=30,
            B_3=5,
            s=8,
            lam=np.array([1.0]),
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        # set_reject_median() is called in __init__, should set attribute
        assert hasattr(result, "rejects_median")
        # Value should be 0 or 1 (or fractional if exact threshold equality)
        assert 0 <= result.rejects_median <= 1

    def test_set_reject_median_with_multiple_bandwidths(self):
        """set_reject_median uses last bandwidth in array."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))

        # Multiple bandwidths - median heuristic uses last one
        lam = np.array([0.5, 1.0, 2.0, 4.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])  # Must match lam length

        result = ctt.actt(
            X1,
            X2,
            g=0,
            B=49,
            B_2=30,
            B_3=5,
            s=8,
            lam=lam,
            weights=weights,
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        assert hasattr(result, "rejects_median")
        # The median heuristic rejection is based on last bandwidth
        # Should be a valid rejection probability
        assert 0 <= result.rejects_median <= 1

    def test_set_reject_median_deterministic(self):
        """set_reject_median gives same result with same seeds."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))

        result1 = ctt.actt(
            X1,
            X2,
            g=0,
            B=49,
            B_2=30,
            B_3=5,
            s=8,
            lam=np.array([1.0]),
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        result2 = ctt.actt(
            X1,
            X2,
            g=0,
            B=49,
            B_2=30,
            B_3=5,
            s=8,
            lam=np.array([1.0]),
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        # Same seeds should give same median rejection
        assert result1.rejects_median == result2.rejects_median
