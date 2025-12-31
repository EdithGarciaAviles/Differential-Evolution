import numpy as np

def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock general: sum_{i=1}^{D-1} [100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2]
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))
