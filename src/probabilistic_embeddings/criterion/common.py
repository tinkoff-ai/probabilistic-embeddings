def non_diag(a):
    """Get non-diagonal elements of matrices.

    Args:
        a: Matrices tensor with shape (..., N, N).

    Returns:
        Non-diagonal elements with shape (..., N, N - 1).
    """
    n = a.shape[-1]
    prefix = list(a.shape)[:-2]
    return a.reshape(*(prefix + [n * n]))[..., :-1].reshape(*(prefix + [n - 1, n + 1]))[..., 1:].reshape(*(prefix + [n, n - 1]))
