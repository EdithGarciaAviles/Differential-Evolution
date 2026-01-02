import numpy as np
import matplotlib.pyplot as plt

from benchmarks import rosenbrock, beale, rastrigin
from de import de_rand_1_bin
from utils import ensure_dir

def bounds_for(name: str, D: int) -> np.ndarray:
    if name == "beale":
        # típico: [-4.5, 4.5]
        return np.array([[-4.5, 4.5]] * D, dtype=float)
    if name == "rosenbrock":
        # típico: [-5, 10]
        return np.array([[-5.0, 10.0]] * D, dtype=float)
    if name == "rastrigin":
        # típico: [-5.12, 5.12]
        return np.array([[-5.12, 5.12]] * D, dtype=float)
    raise ValueError(f"Benchmark desconocido: {name}")

def run_one_experiment(name: str, fobj, D: int, NP: int, G: int, F: float, CR: float, runs: int, seed_base: int):
    bounds = bounds_for(name, D)

    best_fs = []
    histories = []

    for r in range(runs):
        seed = seed_base + 1000 * (hash((name, D)) % 1000) + r  # reproducible y distinto
        res = de_rand_1_bin(
            fobj=fobj,
            bounds=bounds,
            NP=NP,
            G=G,
            F=F,
            CR=CR,
            seed=seed,
        )
        best_fs.append(res.best_f)
        histories.append(res.best_history)

    best_fs = np.array(best_fs, dtype=float)

    # Guardar gráfica de convergencia (usamos la corrida 0)
    ensure_dir("results")
    conv_path = f"results/{name}_D{D}_conv.png"

    plt.figure()
    plt.plot(histories[0])
    plt.xlabel("Generación")
    plt.ylabel("Mejor f(x)")
    plt.yscale("log")
    plt.title(f"DE/rand/1/bin en {name} D={D} | NP={NP}, G={G}, F={F}, CR={CR}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(conv_path, dpi=200)
    plt.close()

    return {
        "benchmark": name,
        "D": D,
        "best": float(np.min(best_fs)),
        "mean": float(np.mean(best_fs)),
        "std": float(np.std(best_fs, ddof=1)) if runs > 1 else 0.0,
        "conv_path": conv_path,
    }

def main():
    # Parámetros DE 
    NP = 50
    G = 500
    F = 0.8
    CR = 0.9

    runs = 10
    seed_base = 42

    experiments = [
        ("beale", beale, 2),
        ("rosenbrock", rosenbrock, 2),
        ("rosenbrock", rosenbrock, 3),
        ("rosenbrock", rosenbrock, 10),
        ("rastrigin", rastrigin, 2),
        ("rastrigin", rastrigin, 10),
    ]

    results = []
    for (name, fobj, D) in experiments:
        print(f"\n=== Ejecutando {name} D={D} ({runs} corridas) ===")
        row = run_one_experiment(name, fobj, D, NP, G, F, CR, runs, seed_base)
        results.append(row)
        print(f"  -> best={row['best']:.6e}, mean={row['mean']:.6e}, std={row['std']:.6e}")
        print(f"  -> conv: {row['conv_path']}")

    ensure_dir("results")
    csv_path = "results/summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("benchmark,D,best,mean,std\n")
        for r in results:
            f.write(f"{r['benchmark']},{r['D']},{r['best']},{r['mean']},{r['std']}\n")

    
    print("\nTabla Markdown:\n")
    print("| Benchmark | D | Best | Mean | Std |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['benchmark']} | {r['D']} | {r['best']:.3e} | {r['mean']:.3e} | {r['std']:.3e} |")

    print(f"\n Listo. CSV guardado en: {csv_path}")
    print(" Revisa results/ para las gráficas.")

if __name__ == "__main__":
    main()
