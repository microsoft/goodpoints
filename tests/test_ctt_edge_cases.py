"""Tests for ctt.py edge cases and parameter variations.

This file targets uncovered lines in ctt.py (excluding GroupResults class):
1. Error handling for unsupported kernels/features (lines 309, 311, 501)
2. Edge cases in rejection logic
3. Results object attribute access

Note: Exact threshold equality edge cases (lines 656-662, 853-863) require
contrived numerical setup to force floating point equality and remain
intentionally untested.
"""

import numpy as np
from goodpoints import ctt
import pytest
import tempfile
import os


class TestUnsupportedParametersErrors:
    """Tests for error handling with unsupported parameters."""

    def test_lrctt_unsupported_kernel(self):
        """lrctt raises ValueError for non-Gaussian kernels."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((128, 2))
        X2 = rng.standard_normal((128, 2))

        # Line 309: kernel != "gauss" in lrctt
        with pytest.raises(ValueError, match="Unsupported kernel name"):
            ctt.lrctt(
                X1,
                X2,
                r=32,
                g=0,
                a=0,
                B=39,
                s=8,
                kernel="laplacian",  # Not supported
                null_seed=0,
                statistic_seed=1,
            )

    def test_lrctt_unsupported_feature_type(self):
        """lrctt raises ValueError for non-RFF feature types."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((128, 2))
        X2 = rng.standard_normal((128, 2))

        # Line 311: feat_type != "rff" in lrctt
        with pytest.raises(ValueError, match="Unsupported feature type"):
            ctt.lrctt(
                X1,
                X2,
                r=32,
                g=0,
                a=0,
                B=39,
                s=8,
                kernel="gauss",
                feat_type="nyström",  # Not supported
                null_seed=0,
                statistic_seed=1,
            )

    def test_actt_unsupported_kernel(self):
        """actt raises ValueError for non-Gaussian kernels."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))

        # Line 501: kernel != "gauss" in actt
        with pytest.raises(ValueError, match="Unsupported kernel name"):
            ctt.actt(
                X1,
                X2,
                g=0,
                B=9,
                B_2=0,
                B_3=0,
                s=8,
                lam=np.array([1.0]),
                kernel="imq",  # Not supported
                null_seed=0,
                statistic_seed=1,
            )


class TestSaveWithCustomFilename:
    """Tests for save() method with custom filename."""

    def test_save_updates_fname_attribute(self):
        """save(fname=...) updates the fname attribute."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))

        result = ctt.actt(
            X1,
            X2,
            g=0,
            B=9,
            B_2=0,
            B_3=0,
            s=8,  # B_2=0 to skip aggregation
            lam=np.array([1.0]),
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        original_fname = result.fname

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            new_fname = f.name

        try:
            # Save with new filename - lines 835-837
            result.save(fname=new_fname)

            # fname should be updated
            assert result.fname == new_fname
            assert result.fname != original_fname

        finally:
            if os.path.exists(new_fname):
                os.remove(new_fname)


class TestRejectionEdgeCases:
    """Tests for edge cases in rejection logic."""

    def test_ctt_with_very_small_sample(self):
        """Test CTT behavior with minimal sample size."""
        rng = np.random.default_rng(42)
        # Small sample to potentially trigger edge cases
        X1 = rng.standard_normal((32, 2))
        X2 = rng.standard_normal((32, 2))

        result = ctt.ctt(
            X1, X2, g=0, B=39, s=4, kernel="gauss", null_seed=0, statistic_seed=1
        )

        assert result is not None
        assert hasattr(result, "rejects")
        assert 0 <= result.rejects <= 1

    def test_actt_with_single_bandwidth(self):
        """Test actt with L=1 (single bandwidth) behaves correctly."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((128, 2))
        X2 = rng.standard_normal((128, 2))

        # Single bandwidth (edge case for aggregated test)
        # Use B_2=0 to skip aggregation for single bandwidth
        result = ctt.actt(
            X1,
            X2,
            g=0,
            B=9,
            B_2=0,
            B_3=0,
            s=8,
            lam=np.array([1.0]),  # L=1
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        assert result is not None
        assert hasattr(result, "rejects")

    def test_lrctt_with_small_feature_count(self):
        """Test lrctt with small r (few features)."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((128, 2))
        X2 = rng.standard_normal((128, 2))

        # Small r to test edge behavior
        result = ctt.lrctt(
            X1,
            X2,
            r=8,  # Very small feature count
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


class TestCttResultObjectEdgeCases:
    """Tests for edge cases in CttResult and AggregatedCttResult."""

    def test_results_alpha_attribute(self):
        """Test that results objects have alpha attribute."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))

        alpha = 0.05
        result = ctt.ctt(
            X1,
            X2,
            g=0,
            B=39,
            s=8,
            alpha=alpha,
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        assert hasattr(result, "alpha")
        assert result.alpha == alpha

    def test_aggregated_results_has_bandwidth_array(self):
        """Test that AggregatedCttResult has bw attribute."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((128, 2))
        X2 = rng.standard_normal((128, 2))

        lam = np.array([1.0])
        result = ctt.actt(
            X1,
            X2,
            g=0,
            B=9,
            B_2=0,
            B_3=0,
            s=8,
            lam=lam,
            kernel="gauss",
            null_seed=0,
            statistic_seed=1,
        )

        assert hasattr(result, "bw")
        np.testing.assert_array_equal(result.bw, lam)
