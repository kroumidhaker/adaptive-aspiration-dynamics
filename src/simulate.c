/* ==========================================================================
 * Monte Carlo simulation of adaptive aspiration dynamics on finite
 * connected graphs.
 *
 * Companion code for:
 *   "Stationary Analysis of Adaptive Aspiration Dynamics in
 *    Graph-Structured Games"  (D. Kroumi)
 *
 * Microscopic rule (Section 2 of the manuscript):
 *   1) a vertex I is selected uniformly at random;
 *   2) its payoff pi_I is computed from its current strategy and its
 *      neighborhood (averaged payoff  pi = x R + (1-x) S  for a
 *      cooperator, pi = x T + (1-x) P for a defector, with x = n_I/d_I;
 *      accumulated payoff multiplies by the degree d_I);
 *   3) the surplus r_I = pi_I - A_I is computed;
 *   4) the strategy self-switches with the Fermi probability
 *          sigma_I = 1 / (1 + exp(beta * r_I));
 *   5) the aspiration is updated with the payoff computed BEFORE the
 *      switch:  A_I <- (1 - rho) A_I + rho pi_I.
 *
 * Estimators (Section "Monte Carlo validation"):
 *   burn-in  T_burn = 2000 N elementary updates,
 *   measure  T_meas = 10000 N elementary updates,
 *   hat p_C     = time average of p_C(t),
 *   hat Var(p_C) = time average of p_C(t)^2  -  (hat p_C)^2 .
 *   Eight independent replications per parameter configuration; graph
 *   realizations are independent across replications and paired across
 *   parameter values (the graph seed depends on the replication index
 *   and the graph parameters, not on beta, rho, the game, or the payoff
 *   convention).
 *
 * Graph families:
 *   REG   random k-regular graph: circulant C_N(1,...,k/2) randomized by
 *         20*N*k accepted double edge swaps, connectivity checked by BFS
 *         (reset to the circulant in the rare disconnected case);
 *   ER    Erdos-Renyi G(N, 4/(N-1)) conditioned on connectivity;
 *   BA    Barabasi-Albert preferential attachment with m = 2;
 *   WHEEL wheel graph (one hub of degree N-1 plus an (N-1)-cycle).
 *
 * Output: one CSV file per dataset with one row per (configuration,
 * replication):  per-run stationary time averages of the mean and the
 * variance of p_C.
 *
 * Build:   gcc -O3 -march=native -fopenmp -o simulate simulate.c -lm
 * Run:     ./simulate [outdir]          (default outdir = "data")
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* Pseudo-random numbers: xoshiro256++ seeded through splitmix64.      */
/* ------------------------------------------------------------------ */
typedef struct { uint64_t s[4]; } rng_t;

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void rng_seed(rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&seed);
}
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s;
    uint64_t result = rotl(s[0] + s[3], 23) + s[0];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}
/* uniform double in [0,1) with 53 random bits */
static inline double rng_u01(rng_t *r) {
    return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}
/* uniform integer in {0,...,n-1} (Lemire's multiply-shift) */
static inline uint32_t rng_uint(rng_t *r, uint32_t n) {
    return (uint32_t)(((__uint128_t)(uint64_t)rng_next(r) * n) >> 64);
}
/* deterministic 64-bit hash of a small integer vector */
static uint64_t hash_fields(const uint64_t *v, int len) {
    uint64_t h = 0x243F6A8885A308D3ULL;
    for (int i = 0; i < len; i++) { h ^= v[i] + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); h = splitmix64(&h); }
    return h;
}

/* ------------------------------------------------------------------ */
/* Graphs                                                              */
/* ------------------------------------------------------------------ */
#define NMAX 128
#define EMAX (NMAX * NMAX / 2)

typedef struct {
    int N;
    int deg[NMAX];
    int off[NMAX + 1];       /* CSR offsets            */
    int adj[2 * EMAX];       /* CSR adjacency          */
} graph_t;

typedef struct {             /* work structure during generation */
    int N, ne;
    unsigned char a[NMAX][NMAX];
    int eu[EMAX], ev[EMAX];  /* edge list */
} gwork_t;

static void gw_clear(gwork_t *g, int N) {
    g->N = N; g->ne = 0;
    memset(g->a, 0, sizeof g->a);
}
static void gw_add(gwork_t *g, int u, int v) {
    g->a[u][v] = g->a[v][u] = 1;
    g->eu[g->ne] = u; g->ev[g->ne] = v; g->ne++;
}
static int gw_connected(const gwork_t *g) {
    int N = g->N, q[NMAX], head = 0, tail = 0, seen = 1;
    unsigned char vis[NMAX] = {0};
    q[tail++] = 0; vis[0] = 1;
    while (head < tail) {
        int u = q[head++];
        for (int v = 0; v < N; v++)
            if (g->a[u][v] && !vis[v]) { vis[v] = 1; q[tail++] = v; seen++; }
    }
    return seen == N;
}
static void gw_to_graph(const gwork_t *g, graph_t *G) {
    int N = g->N;
    G->N = N;
    for (int i = 0; i < N; i++) G->deg[i] = 0;
    for (int e = 0; e < g->ne; e++) { G->deg[g->eu[e]]++; G->deg[g->ev[e]]++; }
    G->off[0] = 0;
    for (int i = 0; i < N; i++) G->off[i + 1] = G->off[i] + G->deg[i];
    int pos[NMAX];
    memcpy(pos, G->off, sizeof(int) * N);
    for (int e = 0; e < g->ne; e++) {
        int u = g->eu[e], v = g->ev[e];
        G->adj[pos[u]++] = v;
        G->adj[pos[v]++] = u;
    }
}

/* circulant C_N(1,...,k/2) */
static void gw_circulant(gwork_t *g, int N, int k) {
    gw_clear(g, N);
    for (int d = 1; d <= k / 2; d++)
        for (int i = 0; i < N; i++) {
            int j = (i + d) % N;
            if (!g->a[i][j]) gw_add(g, i, j);
        }
}

/* random k-regular graph: circulant + 20*N*k accepted double edge swaps */
static void gen_regular(graph_t *G, int N, int k, uint64_t seed) {
    gwork_t g;
    rng_t r; rng_seed(&r, seed);
    for (;;) {
        gw_circulant(&g, N, k);
        long target = 20L * N * k, accepted = 0, guard = 0, guard_max = 4000L * N * k;
        while (accepted < target && guard < guard_max) {
            guard++;
            int e1 = rng_uint(&r, g.ne), e2 = rng_uint(&r, g.ne);
            if (e1 == e2) continue;
            int i = g.eu[e1], j = g.ev[e1], u = g.eu[e2], v = g.ev[e2];
            if (rng_next(&r) & 1) { int t = u; u = v; v = t; }   /* random pairing */
            if (i == u || i == v || j == u || j == v) continue;   /* vertex-disjoint */
            if (g.a[i][u] || g.a[j][v]) continue;                 /* no multi-edges  */
            /* replace {i,j},{u,v} by {i,u},{j,v} */
            g.a[i][j] = g.a[j][i] = 0; g.a[u][v] = g.a[v][u] = 0;
            g.a[i][u] = g.a[u][i] = 1; g.a[j][v] = g.a[v][j] = 1;
            g.eu[e1] = i; g.ev[e1] = u;
            g.eu[e2] = j; g.ev[e2] = v;
            accepted++;
        }
        if (gw_connected(&g)) break;   /* rare failure: regenerate from scratch */
    }
    gw_to_graph(&g, G);
}

/* connected Erdos-Renyi G(N, p) with p = 4/(N-1) */
static void gen_er(graph_t *G, int N, uint64_t seed) {
    gwork_t g;
    rng_t r; rng_seed(&r, seed);
    double p = 4.0 / (double)(N - 1);
    do {
        gw_clear(&g, N);
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++)
                if (rng_u01(&r) < p) gw_add(&g, i, j);
    } while (!gw_connected(&g));
    gw_to_graph(&g, G);
}

/* Barabasi-Albert with attachment parameter m = 2 (seed: triangle) */
static void gen_ba(graph_t *G, int N, uint64_t seed) {
    gwork_t g;
    rng_t r; rng_seed(&r, seed);
    gw_clear(&g, N);
    gw_add(&g, 0, 1); gw_add(&g, 1, 2); gw_add(&g, 0, 2);
    int ends[4 * NMAX];          /* local endpoint list: at most 4*N - 6 entries */
    int nend = 6;
    int tmp[6] = {0, 1, 1, 2, 0, 2};
    memcpy(ends, tmp, sizeof tmp);
    for (int v = 3; v < N; v++) {
        int t1, t2;
        do { t1 = ends[rng_uint(&r, nend)]; } while (0);
        do { t2 = ends[rng_uint(&r, nend)]; } while (t2 == t1);
        gw_add(&g, v, t1); gw_add(&g, v, t2);
        ends[nend++] = v; ends[nend++] = t1;
        ends[nend++] = v; ends[nend++] = t2;
    }
    gw_to_graph(&g, G);   /* BA graphs are connected by construction */
}

/* wheel graph: hub 0, cycle on 1..N-1 */
static void gen_wheel(graph_t *G, int N) {
    gwork_t g;
    gw_clear(&g, N);
    for (int i = 1; i < N; i++) gw_add(&g, 0, i);
    for (int i = 1; i < N - 1; i++) gw_add(&g, i, i + 1);
    gw_add(&g, N - 1, 1);
    gw_to_graph(&g, G);
}

/* ------------------------------------------------------------------ */
/* Games                                                               */
/* ------------------------------------------------------------------ */
typedef struct { const char *name; double R, S, T, P; } game_t;
static const game_t GAMES[4] = {
    {"donation",  1.0, -1.0, 2.0, 0.0},
    {"snowdrift", 3.0,  2.0, 4.0, 0.0},
    {"staghunt",  3.0, -1.0, 0.0, 0.0},
    {"harmony",   3.0,  2.0, 1.0, 0.0},
};

/* ------------------------------------------------------------------ */
/* One simulation run                                                  */
/* ------------------------------------------------------------------ */
enum { MODE_AVG = 0, MODE_ACC = 1 };
enum { GT_REG = 0, GT_ER = 1, GT_BA = 2, GT_WHEEL = 3 };
static const char *GT_NAMES[4] = {"regular", "erdos_renyi", "barabasi_albert", "wheel"};

static void run_one(const graph_t *G, const game_t *gm, int mode,
                    double rho, double beta, uint64_t seed,
                    double *out_mean, double *out_var)
{
    const int N = G->N;
    rng_t r; rng_seed(&r, seed);

    unsigned char strat[NMAX];   /* 1 = C, 0 = D */
    int ncoop[NMAX];             /* cooperating neighbors */
    double A[NMAX];              /* aspirations */

    const double R = gm->R, S = gm->S, T = gm->T, P = gm->P;
    double pmin = R, pmax = R;
    if (S < pmin) pmin = S;
    if (T < pmin) pmin = T;
    if (P < pmin) pmin = P;
    if (S > pmax) pmax = S;
    if (T > pmax) pmax = T;
    if (P > pmax) pmax = P;
    const double pbar = (R + S + T + P) / 4.0;

    int X = 0;
    for (int i = 0; i < N; i++) { strat[i] = (unsigned char)(rng_next(&r) & 1); X += strat[i]; }
    for (int i = 0; i < N; i++) {
        int n = 0;
        for (int e = G->off[i]; e < G->off[i + 1]; e++) n += strat[G->adj[e]];
        ncoop[i] = n;
    }
    for (int i = 0; i < N; i++) {
        if (mode == MODE_AVG) A[i] = pbar;
        else {
            double d = (double)G->deg[i];
            A[i] = d * pmin + (d * pmax - d * pmin) * rng_u01(&r);
        }
    }

    const long Tburn = 2000L * N, Tmeas = 10000L * N;
    double sX = 0.0, sX2 = 0.0;

    for (long t = 0; t < Tburn + Tmeas; t++) {
        int i = (int)rng_uint(&r, (uint32_t)N);
        int d = G->deg[i], n = ncoop[i];
        double pi;
        if (mode == MODE_AVG) {
            double x = (double)n / (double)d;
            pi = strat[i] ? x * R + (1.0 - x) * S : x * T + (1.0 - x) * P;
        } else {
            pi = strat[i] ? n * R + (d - n) * S : n * T + (d - n) * P;
        }
        double rr = pi - A[i];
        double psw = 1.0 / (1.0 + exp(beta * rr));
        if (rng_u01(&r) < psw) {
            int news = 1 - strat[i];
            strat[i] = (unsigned char)news;
            int delta = news ? +1 : -1;
            X += delta;
            for (int e = G->off[i]; e < G->off[i + 1]; e++) ncoop[G->adj[e]] += delta;
        }
        A[i] += rho * rr;   /* aspiration update uses the pre-switch payoff */
        if (t >= Tburn) { sX += (double)X; sX2 += (double)X * (double)X; }
    }
    double m1 = sX / ((double)Tmeas * N);
    double m2 = sX2 / ((double)Tmeas * N * (double)N);
    *out_mean = m1;
    *out_var  = m2 - m1 * m1;
}

/* ------------------------------------------------------------------ */
/* Job list                                                            */
/* ------------------------------------------------------------------ */
typedef struct {
    int dataset;        /* 0 weak_kN, 1 topo_avg, 2 topo_acc, 3 learning */
    int game, gtype, mode, N, k, rep;
    double rho, beta;
    double mean, var;
} job_t;

#define MAXJOBS 20000
static job_t jobs[MAXJOBS];
static int njobs = 0;

static void add_job(int ds, int game, int gt, int mode, int N, int k,
                    double rho, double beta, int rep)
{
    job_t *j = &jobs[njobs++];
    j->dataset = ds; j->game = game; j->gtype = gt; j->mode = mode;
    j->N = N; j->k = k; j->rho = rho; j->beta = beta; j->rep = rep;
}

#define NREP 8
#define NBETA_WEAK 21          /* 0, 0.025, ..., 0.5   */
#define NBETA_STRONG 35        /* + 0.75, 1.0, ..., 4  */

static double beta_weak(int j)   { return 0.025 * j; }
static double beta_strong(int j) { return j < NBETA_WEAK ? 0.025 * j : 0.25 * (j - NBETA_WEAK + 3); }

int main(int argc, char **argv)
{
    const char *outdir = (argc > 1) ? argv[1] : "data";

    /* ---- dataset 0: random k-regular graphs, averaged payoffs, rho=0.5;
            (N=50, k in {4,6,8}) and (k=4, N in {20,50,100})            ---- */
    {
        const int cfgN[5] = {50, 50, 50, 20, 100};
        const int cfgK[5] = { 4,  6,  8,  4,   4};
        for (int c = 0; c < 5; c++)
            for (int gme = 0; gme < 4; gme++)
                for (int b = 0; b < NBETA_WEAK; b++)
                    for (int rep = 0; rep < NREP; rep++)
                        add_job(0, gme, GT_REG, MODE_AVG, cfgN[c], cfgK[c], 0.5, beta_weak(b), rep);
    }
    /* ---- datasets 1 & 2: four graph families, N=50, rho=0.5,
            averaged (1) and accumulated (2) payoffs                     ---- */
    for (int mode = 0; mode < 2; mode++)
        for (int gt = 0; gt < 4; gt++)
            for (int gme = 0; gme < 4; gme++)
                for (int b = 0; b < NBETA_WEAK; b++)
                    for (int rep = 0; rep < NREP; rep++)
                        add_job(1 + mode, gme, gt, mode, 50, 4, 0.5, beta_weak(b), rep);
    /* ---- dataset 3: random 4-regular, averaged, rho in {0.1,0.5,1},
            beta up to 4                                                  ---- */
    {
        const double rhos[3] = {0.1, 0.5, 1.0};
        for (int ir = 0; ir < 3; ir++)
            for (int gme = 0; gme < 4; gme++)
                for (int b = 0; b < NBETA_STRONG; b++)
                    for (int rep = 0; rep < NREP; rep++)
                        add_job(3, gme, GT_REG, MODE_AVG, 50, 4, rhos[ir], beta_strong(b), rep);
    }

    fprintf(stderr, "total runs: %d\n", njobs);

    /* ---- execute ---- */
    long done = 0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int idx = 0; idx < njobs; idx++) {
        job_t *j = &jobs[idx];
        /* graph seed: paired across beta, rho, game, and payoff mode */
        uint64_t gh[5] = {0xA11CEu, (uint64_t)j->gtype, (uint64_t)j->N, (uint64_t)j->k, (uint64_t)j->rep};
        uint64_t gseed = hash_fields(gh, 5);
        graph_t G;
        switch (j->gtype) {
            case GT_REG:   gen_regular(&G, j->N, j->k, gseed); break;
            case GT_ER:    gen_er(&G, j->N, gseed);            break;
            case GT_BA:    gen_ba(&G, j->N, gseed);            break;
            default:       gen_wheel(&G, j->N);                break;
        }
        /* dynamics seed: distinct for every run */
        uint64_t dh[9] = {0xD1CEu, (uint64_t)j->dataset, (uint64_t)j->game,
                          (uint64_t)j->gtype, (uint64_t)j->mode, (uint64_t)j->N,
                          (uint64_t)(j->rho * 1000.0 + 0.5),
                          (uint64_t)(j->beta * 1000.0 + 0.5), (uint64_t)j->rep};
        uint64_t dseed = hash_fields(dh, 9);
        run_one(&G, &GAMES[j->game], j->mode, j->rho, j->beta, dseed, &j->mean, &j->var);
        long done_now;
#ifdef _OPENMP
        #pragma omp atomic capture
#endif
        done_now = ++done;
        if (done_now % 1000 == 0)
            fprintf(stderr, "  %ld / %d runs done\n", done_now, njobs);
    }

    /* ---- write CSVs ---- */
    const char *fnames[4] = {"weak_selection_kN.csv", "topology_averaged.csv",
                             "topology_accumulated.csv", "learning_rate.csv"};
    char path[512];
    for (int ds = 0; ds < 4; ds++) {
        snprintf(path, sizeof path, "%s/%s", outdir, fnames[ds]);
        FILE *f = fopen(path, "w");
        if (!f) { perror(path); return 1; }
        fprintf(f, "game,graph,payoff_mode,N,k,rho,beta,rep,mean_pc,var_pc\n");
        for (int idx = 0; idx < njobs; idx++) {
            const job_t *j = &jobs[idx];
            if (j->dataset != ds) continue;
            fprintf(f, "%s,%s,%s,%d,%d,%.3f,%.3f,%d,%.10f,%.12e\n",
                    GAMES[j->game].name, GT_NAMES[j->gtype],
                    j->mode == MODE_AVG ? "averaged" : "accumulated",
                    j->N, j->k, j->rho, j->beta, j->rep, j->mean, j->var);
        }
        fclose(f);
        fprintf(stderr, "wrote %s\n", path);
    }
    return 0;
}
