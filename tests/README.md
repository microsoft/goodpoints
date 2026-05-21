# GoodPoints Test Suite

## Quick Start

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=goodpoints --cov-report=html

# Run specific file or pattern
pytest tests/test_kt.py -v
pytest tests/ -k "kernel"
```

---

## Fixtures

All fixtures use deterministic seeding for reproducibility.

**make_X(n, d)** - Standard normal test data (mean=0, std=1)
```python
X = make_X(n=64, d=2)  # (64, 2) array, values typically in [-3, 3]
X = make_X(n=32, d=3)  # Each (n, d) pair produces same data every time
```

**make_K_gauss(n)** - Gaussian kernel matrices from standard normal data
```python
K = make_K_gauss(n=64)  # (64, 64) symmetric PSD matrix
# K[i,j] = exp(-‖x_i - x_j‖²), diagonal is 1.0
```

**gaussian_kernel_fn** - Gaussian RBF kernel: k(y, x) = exp(-‖y - x‖²)
```python
# Returns similarity scores in (0, 1], where 1 = identical points
result = kt.thin_X(X, m=1, split_kernel=gaussian_kernel_fn,
                   swap_kernel=gaussian_kernel_fn)
```

---

## Examples

For testing patterns, see:
- `test_properties.py` - Property-based tests with Hypothesis
- `test_kernels.py` - Custom kernels (Laplacian, IMQ, Sobolev)
- `test_integration.py` - Cross-module workflows

---

## Contributing

Before submitting:
1. Run: `pytest tests/`
2. Check coverage: `pytest tests/ --cov=goodpoints`
3. Use factory fixtures (`make_X`, `make_K_gauss`)
4. Add docstrings to test functions
