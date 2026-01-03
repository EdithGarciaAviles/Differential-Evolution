# Differential Evolution (DE/rand/1/bin)

Implementación del algoritmo Differential Evolution (DE/rand/1/bin) y evaluación experimental en benchmarks clásicos de optimización continua.

---

## Objetivo

Ejecutar y analizar el desempeño del algoritmo de Evolución Diferencial sobre benchmarks estándar, documentando los resultados y publicando un repositorio reproducible.

---

## Ejecución

Desde la raíz del repositorio:

```bash
pip install -r requirements.txt
python run_experiments.py
```

El script ejecuta automáticamente todos los experimentos y guarda los resultados en la carpeta `results/`.

---

## Benchmarks y configuraciones

Se realizaron **10 corridas independientes** para cada combinación de benchmark y dimensión.

- **Beale**
  - D = 2

- **Rosenbrock**
  - D = 2  
  - D = 3  
  - D = 10  

- **Rastrigin**
  - D = 2  
  - D = 10  

---

## Resultados experimentales

La siguiente tabla resume los resultados obtenidos.  
Se reporta el mejor valor alcanzado (**Best**), el promedio (**Mean**) y la desviación estándar (**Std**) sobre 10 corridas.

| Benchmark | D | Best | Mean | Std |
|---|---:|---:|---:|---:|
| beale | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| rosenbrock | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| rosenbrock | 3 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| rosenbrock | 10 | 2.399e+00 | 3.781e+00 | 1.130e+00 |
| rastrigin | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| rastrigin | 10 | 3.170e+01 | 3.942e+01 | 5.152e+00 |

---

## Gráficas de convergencia

Las gráficas de convergencia se generan automáticamente y se guardan en la carpeta:

```
results/
```

Entre ellas se incluyen obligatoriamente:

- `rosenbrock_D2_conv.png`
- `rastrigin_D2_conv.png`

---

## Reproducibilidad

- Se utilizan semillas controladas para garantizar resultados reproducibles.
- Todos los experimentos pueden repetirse ejecutando el mismo script principal.

---

## Estructura del repositorio

```
.
├── benchmarks.py
├── de.py
├── run_experiments.py
├── utils.py
├── requirements.txt
├── README.md
└── results/
```
