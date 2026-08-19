"""Regression tests for the unbalanced-sample-size heap-corruption bug.

gaussianc.sum_gaussian_kernel_by_bin / sum_gaussian_kernel_by_bin_aggregated
used to re-derive num_bins1/num_bins2 internally from the combined size of
the (already-compressed) X1/X2 coresets, instead of accepting the values
ctt()/actt() already computed correctly. Whenever the two datasets'
KT-Compress coresets ended up with different per-bin sizes -- generic
whenever n1 and n2 differ substantially -- this produced bin counts that
overran the caller-allocated K_sum matrix, corrupting the heap (observed as
SIGSEGV / "free(): invalid size" / "corrupted size vs. prev_size").
"""

import numpy as np
import pytest
from goodpoints import gaussianc
from goodpoints.ctt import ctt, actt, CttResult, AggregatedCttResult


class TestUnbalancedCttDoesNotCrash:
    def test_reference_2000_current_100_matches_trustyai_crash_shape(self):
        """Reproduces the exact TrustyAI drift-detection crash scenario:
        a reference batch much larger than the current batch."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((2000, 4))  # "reference"
        X2 = rng.standard_normal((100, 4))  # "current"
        result = ctt(X1, X2, g=0, B=39, s=16, null_seed=42, statistic_seed=42)
        assert isinstance(result, CttResult)
        assert np.isfinite(result.statistic_values)
        assert 0 <= result.rejects <= 1

    @pytest.mark.parametrize(
        ("n1", "n2"),
        [(2000, 100), (1000, 37), (100, 2000)],
    )
    def test_various_unbalanced_shapes_do_not_crash(self, n1, n2):
        """Test edge cases including extreme imbalance where num_bins
        would have been 0 without the max(1, ...) fix."""
        rng = np.random.default_rng(7)
        X1 = rng.standard_normal((n1, 2))
        X2 = rng.standard_normal((n2, 2))
        result = ctt(X1, X2, g=0, B=9, s=16, null_seed=0, statistic_seed=1)
        assert np.isfinite(result.statistic_values)

    @pytest.mark.parametrize("same_compression", [True, False])
    def test_actt_unbalanced_does_not_crash(self, same_compression):
        """Exercises both the aggregated (same_compression=True) and
        per-bandwidth (same_compression=False) code paths.

        Uses a single bandwidth: multiple bandwidths with these B/B_2/B_3
        trigger an unrelated, pre-existing IndexError in
        AggregatedCttResult.compute_hat_u_alpha that reproduces even for
        balanced inputs and is out of scope for this fix.
        """
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((2000, 2))
        X2 = rng.standard_normal((100, 2))
        result = actt(
            X1,
            X2,
            g=0,
            B=49,
            B_2=30,
            B_3=5,
            s=16,
            lam=np.array([1.0]),
            kernel="gauss",
            same_compression=same_compression,
            null_seed=0,
            statistic_seed=1,
        )
        assert isinstance(result, AggregatedCttResult)
        for value in result.statistic_values.values():
            assert np.isfinite(value)


class TestSumGaussianKernelByBinNumericalEquivalence:
    """Proves the new (num_bins1, num_bins2)-explicit signature produces
    identical output to the pre-fix implementation for balanced inputs."""

    def test_balanced_matches_golden_values(self):
        # Golden matrix captured from the ORIGINAL (pre-fix) implementation:
        #   rng = np.random.default_rng(7); X1,X2 = rng.standard_normal((16,2)) x2
        #   K_sum = np.empty((4,4)); sum_gaussian_kernel_by_bin(X1, X2, 1.0, K_sum)
        golden = np.array(
            [
                [28.73529978, 11.81695961, 17.85693934, 20.0338135],
                [11.81695961, 25.55758466, 14.18951336, 7.7712617],
                [17.85693934, 14.18951336, 19.33207834, 15.86730704],
                [20.0338135, 7.7712617, 15.86730704, 21.54884222],
            ]
        )
        rng = np.random.default_rng(7)
        X1 = rng.standard_normal((16, 2))
        X2 = rng.standard_normal((16, 2))
        K_sum = np.empty((4, 4))
        gaussianc.sum_gaussian_kernel_by_bin(X1, X2, 1.0, K_sum)
        np.testing.assert_allclose(K_sum, golden, rtol=1e-8)

    def test_ctt_end_to_end_deterministic_value_unchanged(self):
        # Golden value captured by running the ORIGINAL (pre-fix) ctt.py:
        #   rng = np.random.default_rng(42); X1,X2 = rng.standard_normal((64,2)) x2 (shared rng)
        #   ctt(X1, X2, g=0, B=9, s=4, null_seed=0, statistic_seed=1).statistic_values
        #   == 0.005336021275383447; .rejects == 0
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))
        result = ctt(X1, X2, g=0, B=9, s=4, null_seed=0, statistic_seed=1)
        assert result.statistic_values == pytest.approx(0.005336021275383447)
        assert result.rejects == 0

    def test_actt_end_to_end_deterministic_value_unchanged(self):
        # Golden values captured by running the ORIGINAL (pre-fix) ctt.py:
        #   rng = np.random.default_rng(42); X1,X2 = rng.standard_normal((256,2)) x2 (shared rng)
        #   actt(X1, X2, g=0, B=9, B_2=5, B_3=3, s=8, lam=np.array([1.0]),
        #        null_seed=0, statistic_seed=1)
        #   .statistic_values[1.0] == 0.0006098185889738752
        #   .threshold_values[1.0] == 0.0006489148612469335
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))
        result = actt(
            X1,
            X2,
            g=0,
            B=9,
            B_2=5,
            B_3=3,
            s=8,
            lam=np.array([1.0]),
            null_seed=0,
            statistic_seed=1,
        )
        assert float(result.statistic_values[1.0]) == pytest.approx(
            0.0006098185889738752
        )
        assert float(result.threshold_values[1.0]) == pytest.approx(
            0.0006489148612469335
        )
