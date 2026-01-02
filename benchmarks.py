import numpy as np

def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock general:
    f(x) = sum_{i=1}^{D-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    Global minimum at x = (1, ..., 1) with f(x) = 0
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def beale(x: np.ndarray) -> float:
    """
    Beale function (D = 2 only)
    Global minimum at (3, 0.5) with f(x) = 0
    """
    x = np.asarray(x, dtype=float)
    if x.size != 2:
        raise ValueError("Beale function is defined only for D = 2")

    x1, x2 = x
    return float(
        (1.5 - x1 + x1 * x2) ** 2
        + (2.25 - x1 + x1 * (x2 ** 2)) ** 2
        + (2.625 - x1 + x1 * (x2 ** 3)) ** 2
    )


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function (D general)
    Global minimum at x = (0, ..., 0) with f(x) = 0
    """
    x = np.asarray(x, dtype=float)
    A = 10.0
    return float(A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
