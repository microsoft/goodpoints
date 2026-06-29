import numpy as np
import pickle
import tempfile
import pytest
from goodpoints.ctt import ctt, rff, lrctt, actt, kernel_features, CttResult


class TestCtt:
    def test_returns_test_results(self, rng):
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))
        result = ctt(X1, X2, g=0, B=9, s=4, null_seed=0, statistic_seed=1)
        assert isinstance(result, CttResult)

    def test_rejects_in_range(self, rng):
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))
        result = ctt(X1, X2, g=0, B=9, s=4, null_seed=0, statistic_seed=1)
        assert 0 <= result.rejects <= 1

    def test_deterministic(self, rng):
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))
        r1 = ctt(X1, X2, g=0, B=9, s=4, null_seed=0, statistic_seed=1)
        r2 = ctt(X1, X2, g=0, B=9, s=4, null_seed=0, statistic_seed=1)
        assert r1.rejects == r2.rejects
        assert r1.statistic_values == r2.statistic_values

    def test_does_not_reject_same_distribution(self):
        rng = np.random.default_rng(123)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))
        result = ctt(X1, X2, g=0, B=39, s=8, null_seed=0, statistic_seed=1)
        assert result.rejects < 1

    def test_rejects_shifted_distribution(self):
        rng = np.random.default_rng(456)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2)) + 5.0
        result = ctt(X1, X2, g=0, B=39, s=8, null_seed=0, statistic_seed=1)
        assert result.rejects == 1

    def test_invalid_kernel_raises(self, rng):
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))
        with pytest.raises(ValueError, match="Unsupported kernel"):
            ctt(X1, X2, g=0, kernel="invalid")


class TestRff:
    def test_returns_test_results(self, rng):
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))
        result = rff(X1, X2, r=50, B=9, null_seed=0, statistic_seed=1)
        assert isinstance(result, CttResult)

    def test_rejects_in_range(self, rng):
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))
        result = rff(X1, X2, r=50, B=9, null_seed=0, statistic_seed=1)
        assert 0 <= result.rejects <= 1

    def test_deterministic(self, rng):
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))
        r1 = rff(X1, X2, r=50, B=9, null_seed=0, statistic_seed=1)
        r2 = rff(X1, X2, r=50, B=9, null_seed=0, statistic_seed=1)
        assert r1.rejects == r2.rejects
        assert r1.statistic_values == r2.statistic_values

    def test_does_not_reject_same_distribution(self):
        rng = np.random.default_rng(789)
        X1 = rng.standard_normal((128, 2))
        X2 = rng.standard_normal((128, 2))
        result = rff(X1, X2, r=100, B=39, null_seed=0, statistic_seed=1)
        assert result.rejects < 1

    def test_rejects_shifted_distribution(self):
        rng = np.random.default_rng(101)
        X1 = rng.standard_normal((128, 2))
        X2 = rng.standard_normal((128, 2)) + 5.0
        result = rff(X1, X2, r=100, B=39, null_seed=0, statistic_seed=1)
        assert result.rejects == 1


class TestLrctt:
    def test_returns_test_results_or_none(self, rng):
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))
        result = lrctt(X1, X2, g=0, r=50, B=9, s=4, null_seed=0, statistic_seed=1)
        assert result is None or isinstance(result, CttResult)

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((4096, 2))
        X2 = rng.standard_normal((4096, 2))
        r1 = lrctt(X1, X2, g=0, r=50, B=9, s=4, null_seed=0, statistic_seed=1)
        r2 = lrctt(X1, X2, g=0, r=50, B=9, s=4, null_seed=0, statistic_seed=1)
        if r1 is not None and r2 is not None:
            assert r1.rejects == r2.rejects

    def test_does_not_reject_same_distribution(self):
        rng = np.random.default_rng(55)
        X1 = rng.standard_normal((4096, 2))
        X2 = rng.standard_normal((4096, 2))
        result = lrctt(X1, X2, g=0, r=50, B=39, s=4, null_seed=0, statistic_seed=1)
        if result is not None:
            assert result.rejects < 1


class TestActt:
    def test_returns_aggregated_results(self):
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
        from goodpoints.ctt import AggregatedCttResult

        assert isinstance(result, AggregatedCttResult)

    def test_rejects_is_binary(self):
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
        assert result.rejects in (0, 1)

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))
        kwargs = dict(
            g=0,
            B=9,
            B_2=5,
            B_3=3,
            s=8,
            lam=np.array([1.0]),
            null_seed=0,
            statistic_seed=1,
        )
        r1 = actt(X1, X2, **kwargs)
        r2 = actt(X1, X2, **kwargs)
        assert r1.rejects == r2.rejects

    def test_does_not_reject_same_distribution(self):
        rng = np.random.default_rng(77)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2))
        result = actt(
            X1,
            X2,
            g=0,
            B=39,
            B_2=10,
            B_3=5,
            s=8,
            lam=np.array([1.0]),
            null_seed=0,
            statistic_seed=1,
        )
        assert result.rejects == 0

    def test_rejects_shifted_distribution(self):
        rng = np.random.default_rng(88)
        X1 = rng.standard_normal((256, 2))
        X2 = rng.standard_normal((256, 2)) + 5.0
        result = actt(
            X1,
            X2,
            g=0,
            B=39,
            B_2=10,
            B_3=5,
            s=8,
            lam=np.array([1.0]),
            null_seed=0,
            statistic_seed=1,
        )
        assert result.rejects == 1


class TestKernelFeatures:
    def test_shape(self, rng):
        X1 = rng.standard_normal((16, 2))
        X2 = rng.standard_normal((16, 2))
        features = kernel_features(X1, X2, r=20, seed=42)
        assert features.shape == (32, 20)

    def test_deterministic(self, rng):
        X1 = rng.standard_normal((16, 2))
        X2 = rng.standard_normal((16, 2))
        f1 = kernel_features(X1, X2, r=20, seed=42)
        f2 = kernel_features(X1, X2, r=20, seed=42)
        np.testing.assert_array_equal(f1, f2)

    def test_binned_shape(self, rng):
        X1 = rng.standard_normal((16, 2))
        X2 = rng.standard_normal((16, 2))
        features = kernel_features(X1, X2, r=20, seed=42, bin_size=4)
        assert features.shape == (8, 20)


class TestCttResultSaveLoad:
    def test_roundtrip(self, rng):
        X1 = rng.standard_normal((64, 2))
        X2 = rng.standard_normal((64, 2))
        result = ctt(X1, X2, g=0, B=9, s=4, null_seed=0, statistic_seed=1)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            result.save(f.name)
            loaded = pickle.load(open(f.name, "rb"))
        assert loaded.rejects == result.rejects
        assert loaded.statistic_values == result.statistic_values
        assert loaded.name == result.name


class TestKernelFeaturesErrorHandling:
    def test_invalid_feat_type_raises(self, rng):
        X1 = rng.standard_normal((16, 2))
        X2 = rng.standard_normal((16, 2))
        with pytest.raises(ValueError, match="Unsupported feature type"):
            kernel_features(X1, X2, r=10, feat_type="invalid")

    def test_invalid_kernel_raises(self, rng):
        X1 = rng.standard_normal((16, 2))
        X2 = rng.standard_normal((16, 2))
        with pytest.raises(ValueError, match="Unsupported kernel name"):
            kernel_features(X1, X2, r=10, kernel="invalid")
