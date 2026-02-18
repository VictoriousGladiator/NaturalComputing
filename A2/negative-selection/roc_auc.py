import argparse
import csv
import pathlib
import subprocess
import sys
from collections import defaultdict
import matplotlib.pyplot as plt


def run_negsel_scores(jar, self_file, input_file, n, r, use_count=True, log_scores=True, alphabet=None):
    jar = str(jar)
    self_file = str(self_file)
    input_file = str(input_file)

    cmd = ["java", "-jar", jar]

    if alphabet:
        alpha_path = pathlib.Path(alphabet).resolve()
        cmd += ["-alphabet", "file://" + alpha_path.as_posix()]

    cmd += ["-self", self_file, "-n", str(n), "-r", str(r)]
    if use_count:
        cmd.append("-c")
    if log_scores:
        cmd.append("-l")

    try:
        with open(input_file, "rb") as fin:
            p = subprocess.run(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not open input file: {input_file}\n{e}") from e

    if p.returncode != 0:
        err = p.stderr.decode(errors="replace")
        out = p.stdout.decode(errors="replace")
        raise RuntimeError(
            "negsel2.jar failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {p.returncode}\n\n"
            f"STDERR:\n{err}\n\n"
            f"STDOUT:\n{out}\n"
        )

    text = p.stdout.decode(errors="replace").strip()
    if not text:
        return []

    scores = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            scores.append(float(line))
        except ValueError:
            raise RuntimeError(f"Unexpected non-numeric output line: {line!r}")
    return scores


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def roc_points_and_auc(pos_scores, neg_scores):
    P = len(pos_scores)
    N = len(neg_scores)
    if P == 0 or N == 0:
        raise ValueError("Need at least 1 positive and 1 negative example.")

    counts = defaultdict(lambda: [0, 0])
    for s in pos_scores:
        counts[s][0] += 1
    for s in neg_scores:
        counts[s][1] += 1

    scores_desc = sorted(counts.keys(), reverse=True)

    tp = 0
    fp = 0
    points = [(0.0, 0.0)]

    for s in scores_desc:
        cpos, cneg = counts[s]
        tp += cpos
        fp += cneg
        points.append((fp / N, tp / P))

    auc = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        auc += (x1 - x0) * (y0 + y1) / 2.0

    return points, auc


def save_roc_csv(path, points):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr"])
        for x, y in points:
            w.writerow([x, y])


def plot_roc_png(path, points, title):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.figure()
    plt.plot(xs, ys)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("1 - specificity (FPR)")
    plt.ylabel("sensitivity (TPR)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def safe_label_from_path(p):
    p = pathlib.Path(p)
    # "spanish.test" -> "spanish"
    label = p.stem
    # keep it filesystem-friendly
    label = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
    return label


def collect_pos_files(pos_list):
    files = []

    if pos_list:
        for x in pos_list:
            files.append(pathlib.Path(x))


    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for p in files:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def main():
    ap = argparse.ArgumentParser(
        description="Compute ROC curves and AUC for negsel2.jar across one or many languages."
    )
    ap.add_argument("--jar", default="negsel2.jar", help="Path to negsel2.jar")
    ap.add_argument("--train", default="english.train", help="Training self file (e.g., english.train)")
    ap.add_argument("--neg", default="english.test", help="Negative/normal test file (e.g., english.test)")

    ap.add_argument("--pos", action="append", default=None,
                    help="Positive/anomalous test file (repeatable), e.g. --pos tagalog.test")
    ap.add_argument("--alphabet", default=None, help="Optional alphabet file")
    ap.add_argument("--n", type=int, default=10, help="n parameter (pattern length)")
    ap.add_argument("--r-min", type=int, default=1, help="Minimum r to evaluate")
    ap.add_argument("--r-max", type=int, default=9, help="Maximum r to evaluate")
    ap.add_argument("--no-log", action="store_true", help="Do NOT use -l (log2(1+x))")
    ap.add_argument("--no-count", action="store_true", help="Do NOT use -c (count matches)")

    ap.add_argument("--out-dir", default="roc_output", help="Output folder for ROC csv/png")
    ap.add_argument("--out-csv", default="auc_results_by_language.csv", help="Output CSV (all results)")
    ap.add_argument("--out-best-csv", default="auc_best_per_language.csv", help="Output CSV (best per language)")
    ap.add_argument("--save-rocs", action="store_true", help="Save ROC points as CSV")
    ap.add_argument("--plot", action="store_true", help="Save ROC plots as PNG")

    args = ap.parse_args()

    jar = pathlib.Path(args.jar)
    train = pathlib.Path(args.train)
    neg = pathlib.Path(args.neg)
    out_dir = pathlib.Path(args.out_dir)

    log_scores = not args.no_log
    use_count = not args.no_count

    pos_files = collect_pos_files(args.pos)
    if not pos_files:
        raise RuntimeError("No positive files given. Use --pos <file>")

    all_rows = []

    for pos in pos_files:
        lang = safe_label_from_path(pos)
        print(f"\n=== Language: {lang} (file: {pos}) ===")

        for r in range(args.r_min, args.r_max + 1):
            neg_scores = run_negsel_scores(
                jar=jar, self_file=train, input_file=neg, n=args.n, r=r,
                use_count=use_count, log_scores=log_scores, alphabet=args.alphabet
            )
            pos_scores = run_negsel_scores(
                jar=jar, self_file=train, input_file=pos, n=args.n, r=r,
                use_count=use_count, log_scores=log_scores, alphabet=args.alphabet
            )

            points, auc = roc_points_and_auc(pos_scores, neg_scores)

            row = {
                "language": lang,
                "pos_file": str(pos),
                "r": r,
                "auc": auc,
                "mean_neg": mean(neg_scores),
                "mean_pos": mean(pos_scores),
                "n_neg": len(neg_scores),
                "n_pos": len(pos_scores),
            }
            all_rows.append(row)

            print(
                f"r={r:2d}  AUC={auc:.4f}  mean(neg)={row['mean_neg']:.4f}  mean(pos)={row['mean_pos']:.4f}"
            )

            if args.save_rocs:
                save_roc_csv(out_dir / lang / f"roc_{lang}_r{r}.csv", points)

            if args.plot:
                plot_roc_png(
                    out_dir / lang / f"roc_{lang}_r{r}.png",
                    points,
                    title=f"ROC {lang} vs English (n={args.n}, r={r})  AUC={auc:.4f}"
                )

    # Write all results
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "pos_file", "r", "auc", "mean_neg", "mean_pos", "n_neg", "n_pos"])
        for row in all_rows:
            w.writerow([row["language"], row["pos_file"], row["r"], row["auc"],
                        row["mean_neg"], row["mean_pos"], row["n_neg"], row["n_pos"]])

    # Best per language
    best = {}
    for row in all_rows:
        lang = row["language"]
        if (lang not in best) or (row["auc"] > best[lang]["auc"]):
            best[lang] = row

    best_list = sorted(best.values(), key=lambda x: x["auc"], reverse=True)

    with open(args.out_best_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "pos_file", "best_r", "best_auc"])
        for row in best_list:
            w.writerow([row["language"], row["pos_file"], row["r"], row["auc"]])

    print("\n=== Ranking by best AUC (higher = easier to discriminate from English) ===")
    for row in best_list:
        print(f"{row['language']:>15s}  best AUC={row['auc']:.4f} at r={row['r']}")

    print(f"\nWrote: {args.out_csv}")
    print(f"Wrote: {args.out_best_csv}")
    print(f"ROC outputs in: {out_dir}")


if __name__ == "__main__":
    main()
