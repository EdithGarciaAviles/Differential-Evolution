from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any
import numpy as np
from utils import clip_bounds, rng_from_seed

@dataclass
class DEResult:
    best_x: np.ndarray
    best_f: float
    best_history: np.ndarray   # best fitness per generation (incl. gen0)

def de_rand_1_bin(
    fobj: Callable[[np.ndarray], float],
    bounds: np.ndarray,     # shape (D, 2) -> [[lower, upper], ...]
    NP: int,
    G: int,
    F: float,
    CR: float,
    seed: int | None = 42,
) -> DEResult:
    """
    Implementación DE/rand/1/bin:
      mutación: v = Xr1 + F * (Xr2 - Xr3)
      cruza binomial con j_rand
      selección voraz (greedy)
      límites: clipping
    """
    rng = rng_from_seed(seed)

    bounds = np.asarray(bounds, dtype=float)
    D = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    # Inicialización población uniforme en límites
    X = rng.uniform(lower, upper, size=(NP, D))
    fitness = np.array([fobj(X[i]) for i in range(NP)], dtype=float)

    best_idx = int(np.argmin(fitness))
    best_x = X[best_idx].copy()
    best_f = float(fitness[best_idx])

    best_history = [best_f]

    for _gen in range(1, G + 1):
        for i in range(NP):
            # Elegir r1,r2,r3 distintos entre sí y distintos de i
            idxs = np.arange(NP)
            idxs = idxs[idxs != i]
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)

            xr1, xr2, xr3 = X[r1], X[r2], X[r3]

            # Mutación DE/rand/1
            v = xr1 + F * (xr2 - xr3)

            # Manejo de límites (clipping)
            v = clip_bounds(v, lower, upper)

            # Cruza binomial con j_rand
            u = X[i].copy()
            j_rand = int(rng.integers(0, D))
            for j in range(D):
                if (rng.random() < CR) or (j == j_rand):
                    u[j] = v[j]

            # Selección voraz
            fu = fobj(u)
            if fu <= fitness[i]:
                X[i] = u
                fitness[i] = fu

                # actualizar best global
                if fu < best_f:
                    best_f = float(fu)
                    best_x = u.copy()

        best_history.append(best_f)

    return DEResult(best_x=best_x, best_f=best_f, best_history=np.array(best_history, dtype=float))
