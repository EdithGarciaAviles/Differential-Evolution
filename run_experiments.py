import numpy as np
import matplotlib.pyplot as plt

from benchmarks import rosenbrock
from de import de_rand_1_bin
from utils import ensure_dir

def main():
    # --- Parámetros del checkpoint ---
    NP = 50
    G = 500
    F = 0.8
    CR = 0.9
    seed = 42

    # Rosenbrock D=2 típico: límites [-5, 10]
    D = 2
    bounds = np.array([[-5.0, 10.0]] * D)

    res = de_rand_1_bin(
        fobj=rosenbrock,
        bounds=bounds,
        NP=NP,
        G=G,
        F=F,
        CR=CR,
        seed=seed,
    )

    ensure_dir("results")
    out_path = "results/rosenbrock_D2_conv.png"


    plt.figure()
    plt.plot(res.best_history)
    plt.xlabel("Generación")
    plt.ylabel("Mejor f(x)")
    plt.yscale("log")
    plt.title(f"DE/rand/1/bin en Rosenbrock 2D | NP={NP}, G={G}, F={F}, CR={CR}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Listo ✅")
    print(f"Mejor fitness final: {res.best_f:.6e}")
    print(f"Mejor x final: {res.best_x}")
    print(f"Gráfica guardada en: {out_path}")
    print(f"Parámetros: NP={NP}, G={G}, F={F}, CR={CR}, seed={seed}")

if __name__ == "__main__":
    main()
