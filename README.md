# Adaptive Aspiration Dynamics on Graphs — Monte Carlo Validation

Simulation code, data, and figures accompanying

> **Stationary Analysis of Adaptive Aspiration Dynamics in Graph-Structured Games**
> Dhaker Kroumi, Department of Mathematics, King Fahd University of Petroleum and Minerals

The code implements the finite-population aspiration-learning process of the paper on
finite connected graphs and validates the first-order weak-selection predictions

$$
\mathbb{E}_\beta[p_C] \;=\; \tfrac12 + \tfrac{\beta}{16}\,(R+S-T-P) + o(\beta),
\qquad
N\,\mathrm{Var}_\beta(p_C) \;=\; \tfrac14 + \tfrac{\beta}{32}\big[(R-S)-(T-P)\big] + o(\beta).
$$

## Model

Each elementary update follows the microscopic rule of Section 2 of the manuscript:
a vertex is selected uniformly at random; its payoff (averaged over its $d_i$ pairwise
interactions, or accumulated when stated) and its payoff–aspiration surplus
$r_i = \pi_i - A_i$ are computed; the strategy self-switches with the Fermi probability
$\sigma_i = \big(1 + e^{\beta r_i}\big)^{-1}$; and the aspiration is then updated with the
payoff computed before switching, $A_i \leftarrow (1-\rho)A_i + \rho\,\pi_i$.

Four standard pairwise games are simulated:

| Game | $(R,S,T,P)$ | Mean contrast $R{+}S{-}T{-}P$ | Variance contrast $(R{-}S){-}(T{-}P)$ |
|---|---|---|---|
| Donation | $(1,-1,2,0)$ | $-2$ | $0$ |
| Snowdrift | $(3,2,4,0)$ | $1$ | $-2$ |
| Stag hunt | $(3,-1,0,0)$ | $2$ | $4$ |
| Harmony | $(3,2,1,0)$ | $4$ | $0$ |

## Repository layout

```
├── Makefile
├── src/
│   └── simulate.c              # self-contained C simulation (OpenMP)
├── data/                       # per-run CSV output (one row per replication)
│   ├── weak_selection_kN.csv
│   ├── topology_averaged.csv
│   ├── topology_accumulated.csv
│   └── learning_rate.csv
├── notebooks/
│   └── plot_figures.ipynb      # reads data/, writes figures/
└── figures/                    # five publication figures (PDF + PNG)
```

## Building and reproducing everything

```bash
make            # compile:  gcc -O3 -march=native -fopenmp src/simulate.c -lm
make data       # run all 12,096 simulation runs (a few minutes on a laptop)
make figures    # execute the notebook and regenerate the five figures
```

Requirements: a C compiler with OpenMP, Python 3 with `numpy`, `pandas`,
`matplotlib`, and `jupyter` (`nbconvert`) for the figures.

The program is fully deterministic: graph seeds depend only on the graph family,
its parameters, and the replication index. Random graph realizations are therefore
**independent across replications and paired across parameter values** ($\beta$,
$\rho$, game, payoff convention), as described in the manuscript; the wheel graph
is fixed. Dynamics seeds are distinct for every run (xoshiro256++ seeded through
splitmix64).

## Simulation protocol

* Burn-in $\mathcal{T}_{\mathrm{burn}} = 2000N$ updates, measurement
  $\mathcal{T}_{\mathrm{meas}} = 10000N$ updates, eight independent replications
  per parameter configuration.
* Initial strategies i.i.d. uniform on $\{C, D\}$. Averaged payoffs: all
  aspirations start at $\bar\pi = (R+S+T+P)/4$; accumulated payoffs: the initial
  aspiration of vertex $i$ is uniform on $[d_i\pi_-,\, d_i\pi_+]$.
* Estimators: time averages
  $\widehat{p}_C = \mathcal{T}_{\mathrm{meas}}^{-1}\sum_t p_C(t)$ and
  $\widehat{\mathrm{Var}}(p_C) = \mathcal{T}_{\mathrm{meas}}^{-1}\sum_t p_C(t)^2 - \widehat{p}_C^{\,2}$.
* Graph families ($N = 50$ unless varied): random $k$-regular (circulant
  $C_N(1,\dots,k/2)$ randomized by $20Nk$ accepted double edge swaps, connectivity
  checked by BFS), connected Erdős–Rényi $G(N, 4/(N-1))$, Barabási–Albert with
  $m = 2$, and the wheel graph.

## Datasets and figures

| Dataset (CSV) | Sweep | Figure(s) |
|---|---|---|
| `weak_selection_kN.csv` | $k\in\{4,6,8\}$ at $N=50$, and $N\in\{20,50,100\}$ at $k=4$; averaged payoffs, $\rho=0.5$, $\beta\in\{0,0.025,\dots,0.5\}$ | `figure_weak_selection_mean.pdf`, `figure_weak_selection_variance.pdf` |
| `topology_averaged.csv` | four graph families, averaged payoffs, $N=50$, $\rho=0.5$ | `figure_topology_mean_comparison.pdf`, `figure_topology_variance_comparison.pdf` (top rows) |
| `topology_accumulated.csv` | four graph families, accumulated payoffs, $N=50$, $\rho=0.5$ | `figure_topology_mean_comparison.pdf`, `figure_topology_variance_comparison.pdf` (bottom rows) |
| `learning_rate.csv` | $\rho\in\{0.1,0.5,1\}$, random 4-regular, averaged payoffs, $\beta$ up to $4$ | `figure_learning_rate_strong_selection.pdf` |

CSV schema (one row per run):

```
game, graph, payoff_mode, N, k, rho, beta, rep, mean_pc, var_pc
```

`mean_pc` and `var_pc` are the per-run stationary time averages defined above;
the notebook aggregates them into replication means and standard errors and
scales the variance by $N$. The within-run observations are temporally correlated,
so uncertainty is computed across the eight independent replications rather than
by treating individual time points as independent observations.

In each figure, markers are means over the eight replications and, where shown,
error bars represent one standard error. Solid black lines in Figures 1 and 2
are the first-order predictions of Theorems 3.1 and 3.4. Dotted gray lines mark
the neutral values $1/2$ and $1/4$. The topology-comparison and learning-rate
figures show the neutral reference lines but omit the first-order prediction
lines, consistently with the presentation in the paper.

## Contact

Dhaker Kroumi — `dhaker.kroumi@kfupm.edu.sa`
