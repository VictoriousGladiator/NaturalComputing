import os
import csv
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive — saves to files instead of displaying
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_image(file_path, dimensions=(128, 128)):
    img = io.imread(file_path)
    img = img / 255.0
    return resize(img, dimensions, anti_aliasing=True)


# ============================================================
# 2. SHARED METRIC
# ============================================================

def compression_error(pixels, palette):
    """
    Total squared distance from each pixel to its nearest palette color.
    pixels : (N, 3)
    palette: (K, 3)
    """
    dists = np.linalg.norm(pixels[:, None, :] - palette[None, :, :], axis=2)
    return float(np.sum(np.min(dists, axis=1) ** 2))


def compress_with_palette(pixels, palette):
    """Map each pixel to the nearest palette color. Returns (N, 3)."""
    dists = np.linalg.norm(pixels[:, None, :] - palette[None, :, :], axis=2)
    return palette[np.argmin(dists, axis=1)]


# ============================================================
# 3. PSO WITH OPTIONAL SNAPSHOTS
# ============================================================

class SwarmOptimizer:
    """
    PSO for color palette optimization.
    Pass snapshot_dir to save palette + compressed image at regular intervals.
    """

    def __init__(self, pixels, image_shape, K=8, agent_count=20,
                 inertia=0.7, cognitive=1.5, social=1.5, max_iterations=50,
                 snapshot_dir=None, snapshot_interval=10, run_id=0):

        self.pixels = pixels
        self.image_shape = image_shape
        self.K = K
        self.dim = 3 * K
        self.agent_count = agent_count
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.max_iterations = max_iterations
        self.snapshot_dir = snapshot_dir
        self.snapshot_interval = snapshot_interval
        self.run_id = run_id

        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)

        self._init_swarm()

    def _init_swarm(self):
        self.positions = np.random.rand(self.agent_count, self.dim)
        self.velocities = np.zeros((self.agent_count, self.dim))
        self.personal_best = self.positions.copy()
        self.personal_scores = np.array([
            compression_error(self.pixels, p.reshape(self.K, 3))
            for p in self.positions
        ])
        best_idx = np.argmin(self.personal_scores)
        self.global_best = self.personal_best[best_idx].copy()
        self.global_score = self.personal_scores[best_idx]

    def optimize(self):
        history = []

        for it in range(self.max_iterations):
            for i in range(self.agent_count):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                self.velocities[i] = (
                    self.inertia * self.velocities[i]
                    + self.cognitive * r1 * (self.personal_best[i] - self.positions[i])
                    + self.social   * r2 * (self.global_best        - self.positions[i])
                )
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], 0, 1)

                score = compression_error(self.pixels, self.positions[i].reshape(self.K, 3))

                if score < self.personal_scores[i]:
                    self.personal_scores[i] = score
                    self.personal_best[i] = self.positions[i].copy()

                if score < self.global_score:
                    self.global_score = score
                    self.global_best = self.positions[i].copy()

            history.append(self.global_score)

            if self.snapshot_dir:
                snap = (it == 0) or ((it + 1) % self.snapshot_interval == 0) \
                       or (it == self.max_iterations - 1)
                if snap:
                    self._save_snapshot(it + 1)

        return self.global_best.reshape(self.K, 3), history

    def _save_snapshot(self, iteration):
        palette = np.clip(self.global_best.reshape(self.K, 3), 0, 1)
        H, W = self.image_shape
        compressed = compress_with_palette(self.pixels, palette).reshape(H, W, 3)

        fig, axes = plt.subplots(2, 1, figsize=(5, 5),
                                 gridspec_kw={"height_ratios": [1, 5]})
        axes[0].imshow([palette])
        axes[0].set_title(f"Palette — iter {iteration}  error {self.global_score:.1f}", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(np.clip(compressed, 0, 1))
        axes[1].axis("off")
        fig.tight_layout(pad=0.5)

        fname = os.path.join(
            self.snapshot_dir,
            f"run{self.run_id:02d}_iter{iteration:03d}.png"
        )
        fig.savefig(fname, dpi=90, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# 4. K-MEANS WITH PER-ITERATION TRACKING
# ============================================================

def run_kmeans(pixels, K, max_iter=50, seed=None):
    """
    Manual K-means that returns (palette, error_history).
    If it converges early the last value is repeated up to max_iter.
    """
    rng = np.random.default_rng(seed)
    centers = pixels[rng.choice(len(pixels), K, replace=False)].copy()
    history = []

    for _ in range(max_iter):
        dists = np.linalg.norm(pixels[:, None] - centers[None], axis=2)
        labels = np.argmin(dists, axis=1)

        new_centers = np.array([
            pixels[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
            for k in range(K)
        ])

        err = compression_error(pixels, new_centers)
        history.append(err)

        if np.allclose(centers, new_centers, atol=1e-7):
            history += [err] * (max_iter - len(history))
            break

        centers = new_centers

    return centers, history


# ============================================================
# 5. PSO HYPERPARAMETER GRID SEARCH
# ============================================================

def grid_search_pso(pixels, image_shape, K=8,
                    inertia_vals=(0.4, 0.6, 0.8),
                    cognitive_vals=(1.0, 1.5, 2.0),
                    social_vals=(1.0, 1.5, 2.0),
                    agent_counts=(10, 20, 30),
                    search_iterations=30,
                    runs_per_combo=3):
    """
    Exhaustive grid search over PSO hyperparameters.
    Each combination is evaluated `runs_per_combo` times to smooth noise.
    Returns a list of result dicts sorted by mean_error ascending.
    """
    grid = list(itertools.product(inertia_vals, cognitive_vals, social_vals, agent_counts))
    total = len(grid)
    results = []

    for idx, (w, c, s, n) in enumerate(grid):
        print(f"  [{idx+1:3d}/{total}] inertia={w}  cognitive={c}  social={s}  agents={n}",
              flush=True)
        errors = []
        for _ in range(runs_per_combo):
            opt = SwarmOptimizer(pixels, image_shape, K=K,
                                 agent_count=n, inertia=w, cognitive=c, social=s,
                                 max_iterations=search_iterations)
            _, hist = opt.optimize()
            errors.append(hist[-1])

        results.append({
            "inertia":    w,
            "cognitive":  c,
            "social":     s,
            "agents":     n,
            "mean_error": float(np.mean(errors)),
            "std_error":  float(np.std(errors)),
        })

    results.sort(key=lambda x: x["mean_error"])
    return results


def save_grid_search_results(results, output_dir):
    """Save CSV + top-N bar chart + pairwise heatmaps."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- CSV ----
    csv_path = os.path.join(output_dir, "grid_search.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved grid search CSV → {csv_path}")

    # ---- Bar chart: top 20 combinations ----
    top = results[:20]
    labels = [
        f"w={r['inertia']} c={r['cognitive']}\ns={r['social']} n={r['agents']}"
        for r in top
    ]
    means = [r["mean_error"] for r in top]
    stds  = [r["std_error"]  for r in top]

    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(top))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean Final Error")
    ax.set_title("Top 20 PSO Hyperparameter Combinations (grid search)")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    bar_path = os.path.join(output_dir, "grid_search_top20.png")
    fig.savefig(bar_path, dpi=110)
    plt.close(fig)
    print(f"Saved top-20 bar chart → {bar_path}")

    # ---- Pairwise heatmaps (6 pairs of 4 parameters) ----
    param_names = ["inertia", "cognitive", "social", "agents"]
    param_vals  = {p: sorted(set(r[p] for r in results)) for p in param_names}
    pairs = list(itertools.combinations(param_names, 2))  # 6 pairs

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (p1, p2) in zip(axes, pairs):
        v1 = param_vals[p1]
        v2 = param_vals[p2]
        matrix = np.zeros((len(v1), len(v2)))

        for i, val1 in enumerate(v1):
            for j, val2 in enumerate(v2):
                # Average over all other-parameter combinations
                matching = [r["mean_error"] for r in results
                            if r[p1] == val1 and r[p2] == val2]
                matrix[i, j] = np.mean(matching)

        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(v2))); ax.set_xticklabels(v2, fontsize=8)
        ax.set_yticks(range(len(v1))); ax.set_yticklabels(v1, fontsize=8)
        ax.set_xlabel(p2, fontsize=9)
        ax.set_ylabel(p1, fontsize=9)
        ax.set_title(f"{p1} × {p2}", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(v1)):
            for j in range(len(v2)):
                ax.text(j, i, f"{matrix[i,j]:.0f}",
                        ha="center", va="center", fontsize=7, color="black")

    fig.suptitle("Pairwise hyperparameter interactions\n"
                 "(mean error, averaged over remaining parameters)", fontsize=11)
    fig.tight_layout()
    hm_path = os.path.join(output_dir, "grid_search_heatmaps.png")
    fig.savefig(hm_path, dpi=110)
    plt.close(fig)
    print(f"Saved pairwise heatmaps → {hm_path}")

    return csv_path, bar_path, hm_path


# ============================================================
# 6. MULTI-RUN COMPARISON
# ============================================================

def run_pso_trials(pixels, image_shape, n_runs, K=8, max_iterations=50, **pso_kwargs):
    histories = []
    for r in range(n_runs):
        print(f"  PSO run {r+1}/{n_runs}", flush=True)
        opt = SwarmOptimizer(pixels, image_shape, K=K, max_iterations=max_iterations,
                             **pso_kwargs)
        _, hist = opt.optimize()
        histories.append(hist)
    return np.array(histories)  # (n_runs, max_iterations)


def run_kmeans_trials(pixels, n_runs, K=8, max_iter=50):
    histories = []
    for r in range(n_runs):
        print(f"  K-Means run {r+1}/{n_runs}", flush=True)
        _, hist = run_kmeans(pixels, K, max_iter=max_iter, seed=r)
        histories.append(hist)
    return np.array(histories)  # (n_runs, max_iter)


# ============================================================
# 7. PLOTTING
# ============================================================

def plot_convergence(pso_histories, km_histories, output_path, pso_label="PSO"):
    fig, ax = plt.subplots(figsize=(9, 5))

    for histories, label, color in [
        (pso_histories, pso_label, "tab:blue"),
        (km_histories,  "K-Means", "tab:orange"),
    ]:
        iters = np.arange(1, histories.shape[1] + 1)
        mean  = histories.mean(axis=0)
        std   = histories.std(axis=0)
        ax.plot(iters, mean, color=color, lw=2, label=f"{label} mean")
        ax.fill_between(iters, mean - std, mean + std, alpha=0.25, color=color,
                        label=f"{label} ±1 std")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Compression Error (sum of squared distances)")
    ax.set_title(f"{pso_label} vs K-Means Convergence")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved convergence plot → {output_path}")


def plot_final_errors(pso_histories, km_histories, output_path, pso_label="PSO"):
    fig, ax = plt.subplots(figsize=(5, 4))
    data   = [pso_histories[:, -1], km_histories[:, -1]]
    labels = [pso_label, "K-Means"]
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor="lightblue"),
                    medianprops=dict(color="red", lw=2))
    bp["boxes"][1].set_facecolor("moccasin")
    ax.set_ylabel("Final Compression Error")
    ax.set_title("Final Error Distribution")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved final-error box-plot → {output_path}")


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    # ---------- configuration ----------
    IMAGE_PATH        = "A3/si-exercises/exercise_pso/image.png"
    IMAGE_SIZE        = (128, 128)
    K                 = 8
    MAX_ITER          = 50       # iterations for comparison runs
    N_RUNS            = 8        # runs for the statistical comparison
    SNAPSHOT_INTERVAL = 10       # save a snapshot every N PSO iterations

    # Grid search settings
    SEARCH_ITER       = 30       # shorter runs are enough to rank combinations
    RUNS_PER_COMBO    = 3        # repeats per combo to average out noise

    OUTPUT_DIR        = "A3/output"
    SNAPSHOT_DIR      = os.path.join(OUTPUT_DIR, "snapshots")
    GRID_DIR          = os.path.join(OUTPUT_DIR, "grid_search")
    os.makedirs(OUTPUT_DIR,   exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(GRID_DIR,     exist_ok=True)
    # -----------------------------------

    img = load_image(IMAGE_PATH, IMAGE_SIZE)
    H, W, _ = img.shape
    pixels = img.reshape(-1, 3)

    # ------------------------------------------------------------------
    # PART 0 — grid search to find the best PSO hyperparameters
    # ------------------------------------------------------------------
    # print("=== Part 0: PSO hyperparameter grid search ===")
    # print(f"Grid: inertia×cognitive×social×agents  ({SEARCH_ITER} iter, {RUNS_PER_COMBO} runs/combo)\n")

    # grid_results = grid_search_pso(
    #     pixels, (H, W), K=K,
    #     inertia_vals  = (0.4, 0.6, 0.8),
    #     cognitive_vals= (1.0, 1.5, 2.0),
    #     social_vals   = (1.0, 1.5, 2.0),
    #     agent_counts  = (20, 30, 40),
    #     search_iterations=SEARCH_ITER,
    #     runs_per_combo=RUNS_PER_COMBO,
    # )

    # save_grid_search_results(grid_results, GRID_DIR)

    # best = grid_results[0]
    # print(f"\nBest configuration found:")
    # print(f"  inertia={best['inertia']}  cognitive={best['cognitive']}  "
    #       f"social={best['social']}  agents={int(best['agents'])}")
    # print(f"  mean_error={best['mean_error']:.2f}  std={best['std_error']:.2f}")

    # best_pso_label = (f"PSO (w={best['inertia']} c={best['cognitive']} "
    #                   f"s={best['social']} n={int(best['agents'])})")
    # Best configuration found:
    # inertia=0.4  cognitive=1.5  social=1.0  agents=40
    # mean_error=188.34  std=8.02
    # ------------------------------------------------------------------
    # PART A — single PSO run with snapshots (using best params)
    # ------------------------------------------------------------------
    
    # Best configuration found:
    print("\n=== Part A: PSO with snapshots (best params) ===")
    opt = SwarmOptimizer(
        pixels, (H, W), K=K,
        agent_count=40,
        inertia=0.4,
        cognitive=1.5,
        social=1.0,
        max_iterations=MAX_ITER,
        snapshot_dir=SNAPSHOT_DIR,
        snapshot_interval=SNAPSHOT_INTERVAL,
        run_id=0,
    )
    pso_palette, _ = opt.optimize()
    print(f"Snapshots saved to: {SNAPSHOT_DIR}/")

    # ------------------------------------------------------------------
    # PART B — multi-run statistical comparison (best PSO vs K-Means)
    # ------------------------------------------------------------------
    print("\n=== Part B: PSO trials (best params) ===")
    pso_histories = run_pso_trials(
        pixels, (H, W), n_runs=N_RUNS, K=K, max_iterations=MAX_ITER,
        agent_count=40,
        inertia=0.4,
        cognitive=1.5,
        social=1.0
    )

    print("\n=== Part B: K-Means trials ===")
    km_histories = run_kmeans_trials(pixels, n_runs=N_RUNS, K=K, max_iter=MAX_ITER)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("\n--- Final error summary ---")
    print(f"PSO     final error: mean={pso_histories[:,-1].mean():.2f}  "
          f"std={pso_histories[:,-1].std():.2f}  "
          f"min={pso_histories[:,-1].min():.2f}")
    print(f"K-Means final error: mean={km_histories[:,-1].mean():.2f}  "
          f"std={km_histories[:,-1].std():.2f}  "
          f"min={km_histories[:,-1].min():.2f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    best_pso_label = (f"PSO (w={0.4} c={1.5} "
                      f"s={1.0} n={40})")
    plot_convergence(pso_histories, km_histories,
                     os.path.join(OUTPUT_DIR, "convergence.png"),
                     pso_label=best_pso_label)
    plot_final_errors(pso_histories, km_histories,
                      os.path.join(OUTPUT_DIR, "final_errors.png"),
                      pso_label=best_pso_label)

    print("\nDone. All output written to:", OUTPUT_DIR)
