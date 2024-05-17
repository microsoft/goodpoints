'''
A simple batched matrix-vector multiplication implementation.
'''


def mat_vec_dot(M, x):
    '''
    Args:
        M: (..., m, n)
        x: (..., n)

    Returns:
        (..., m)
    '''

    # Using fancy indexing to work with both numpy and jax.
    return (M * x[..., None, :]).sum(-1)
