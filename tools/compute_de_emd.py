


import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist, cdist
from sklearn.decomposition import PCA
import anndata as ad

warnings.filterwarnings("ignore")





PAPER_DIR = Path(__file__).resolve().parent.parent / "paper_results"
MAX_CELLS_PERT = 2000
N_PCA = 100


def _to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X, dtype=np.float64)






def compute_emd_per_gene(X_pred, X_real):
    
    G = X_pred.shape[1]
    vals = np.empty(G, dtype=np.float64)
    for g in range(G):
        vals[g] = wasserstein_distance(X_pred[:, g], X_real[:, g])
    return float(np.mean(vals)), float(np.std(vals)), vals


def compute_energy_distance(X_pred, X_real):
    
    cross = cdist(X_pred, X_real, metric="euclidean").mean()
    wp = pdist(X_pred, metric="euclidean").mean() if X_pred.shape[0] > 1 else 0.0
    wr = pdist(X_real, metric="euclidean").mean() if X_real.shape[0] > 1 else 0.0
    return float(2.0 * cross - wp - wr)


def subsample(X, max_cells):
    if X.shape[0] <= max_cells:
        return X
    rng = np.random.default_rng(42)
    return X[rng.choice(X.shape[0], size=max_cells, replace=False)]






def get_perturbation_column(adata, dataset, model):
    if model == "state":
        for col in ["product_name", "gene"]:
            if col in adata.obs.columns:
                return col
    for col in ["condition", "product_name", "gene"]:
        if col in adata.obs.columns:
            return col
    return None


def compute_per_perturbation(pred, real, pert_col, max_cells, n_pca):
    Xp = _to_dense(pred.X)
    Xr = _to_dense(real.X)

    pred_perts = set(pred.obs[pert_col].unique())
    real_perts = set(real.obs[pert_col].unique())
    shared = sorted(pred_perts & real_perts)

    if not shared:
        return []

    results = []
    for pert in shared:
        mp = pred.obs[pert_col] == pert
        mr = real.obs[pert_col] == pert
        Xp_pert = Xp[mp.values]
        Xr_pert = Xr[mr.values]

        if Xp_pert.shape[0] < 5 or Xr_pert.shape[0] < 5:
            continue

        
        emd_mean, emd_std, _ = compute_emd_per_gene(Xp_pert, Xr_pert)

        
        s1 = subsample(Xp_pert, max_cells)
        s2 = subsample(Xr_pert, max_cells)
        combined = np.vstack([s1, s2])
        nc = min(n_pca, min(combined.shape) - 1)
        pca = PCA(n_components=nc, random_state=42)
        cp = pca.fit_transform(combined)
        edist = compute_energy_distance(cp[: s1.shape[0]], cp[s1.shape[0]:])

        results.append({
            "perturbation": str(pert),
            "n_pred": Xp_pert.shape[0],
            "n_real": Xr_pert.shape[0],
            "emd_mean": emd_mean,
            "emd_std": emd_std,
            "edist_pca": edist,
        })

    return results


def compute_global(pred, real, max_cells, n_pca):
    Xp = _to_dense(pred.X)
    Xr = _to_dense(real.X)

    emd_mean, emd_std, _ = compute_emd_per_gene(Xp, Xr)

    s1 = subsample(Xp, max_cells)
    s2 = subsample(Xr, max_cells)
    combined = np.vstack([s1, s2])
    nc = min(n_pca, min(combined.shape) - 1)
    pca = PCA(n_components=nc, random_state=42)
    cp = pca.fit_transform(combined)
    edist = compute_energy_distance(cp[: s1.shape[0]], cp[s1.shape[0]:])

    return [{
        "perturbation": "__global__",
        "n_pred": Xp.shape[0],
        "n_real": Xr.shape[0],
        "emd_mean": emd_mean,
        "emd_std": emd_std,
        "edist_pca": edist,
    }]






def main():
    parser = argparse.ArgumentParser(description="EMD + E-Distance on all genes (eval_raw style)")
    parser.add_argument("--max-cells-pert", type=int, default=MAX_CELLS_PERT)
    parser.add_argument("--n-pca", type=int, default=N_PCA)
    parser.add_argument("--datasets", type=str, nargs="*", default=None)
    parser.add_argument("--models", type=str, nargs="*", default=None)
    args = parser.parse_args()

    max_cells_pert = args.max_cells_pert
    n_pca = args.n_pca

    print(f"Configuration:")
    print(f"  EMD:    per-gene Wasserstein-1 on ALL genes")
    print(f"  EDist:  PCA {n_pca}d on ALL genes")
    print(f"  max_cells_per_pert = {max_cells_pert}")
    print()

    summary_rows = []
    all_pert_rows = []

    datasets = sorted(
        d for d in PAPER_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if args.datasets:
        datasets = [d for d in datasets if d.name in args.datasets]

    for ds_dir in datasets:
        dataset = ds_dir.name
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset}")
        print(f"{'='*70}")

        models = {}
        for model_dir in sorted(ds_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            p = model_dir / "adata_pred.h5ad"
            r = model_dir / "adata_real.h5ad"
            if p.exists() and r.exists():
                m = model_dir.name
                if args.models and m not in args.models:
                    continue
                models[m] = (str(p), str(r))

        if not models:
            print("  No models found.")
            continue

        model_list = sorted(models.keys())
        print(f"  Models: {len(models)} ({', '.join(model_list)})")

        n_done = 0
        for model in model_list:
            pred_path, real_path = models[model]
            n_done += 1
            print(f"  [{n_done:2d}/{len(models)}] {model:16s} ...", end=" ", flush=True)

            try:
                pred = ad.read_h5ad(pred_path)
                real = ad.read_h5ad(real_path)
            except Exception as e:
                print(f"ERROR loading: {e}")
                continue

            pert_col = get_perturbation_column(pred, dataset, model)
            n_genes = pred.shape[1]

            if pert_col:
                pert_results = compute_per_perturbation(pred, real, pert_col, max_cells_pert, n_pca)
            else:
                pert_results = []

            if not pert_results:
                pert_results = compute_global(pred, real, max_cells_pert, n_pca)

            emd_vals = [r["emd_mean"] for r in pert_results]
            edist_vals = [r["edist_pca"] for r in pert_results]

            summary_rows.append({
                "dataset": dataset,
                "model": model,
                "n_perturbations": len(pert_results),
                "n_genes": n_genes,
                "emd_mean": np.mean(emd_vals),
                "emd_std": np.std(emd_vals),
                "emd_median": np.median(emd_vals),
                "edist_pca_mean": np.mean(edist_vals),
                "edist_pca_std": np.std(edist_vals),
                "edist_pca_median": np.median(edist_vals),
            })

            for r in pert_results:
                r["dataset"] = dataset
                r["model"] = model
                all_pert_rows.append(r)

            print(f"EMD={np.mean(emd_vals):.4f}  EDist={np.mean(edist_vals):.4f}  "
                  f"({len(pert_results)} perts, {n_genes} genes)")

    
    summary_df = pd.DataFrame(summary_rows)
    pert_df = pd.DataFrame(all_pert_rows)

    summary_path = PAPER_DIR / "emd_edistance_summary.csv"
    pert_path = PAPER_DIR / "emd_edistance_per_pert.csv"

    computed_keys = set(zip(summary_df["dataset"], summary_df["model"]))
    for path, df_new in [(summary_path, summary_df), (pert_path, pert_df)]:
        if path.exists():
            old = pd.read_csv(path)
            old = old[~old.apply(lambda r: (r["dataset"], r["model"]) in computed_keys, axis=1)]
            df_new = pd.concat([old, df_new], ignore_index=True)
        df_new = df_new.sort_values(["dataset", "model"]).reset_index(drop=True)
        df_new.to_csv(path, index=False)

    print(f"\n✅ {summary_path}")
    print(f"✅ {pert_path}")

    
    print("\n" + "=" * 85)
    print("  EMD & E-Distance — all genes, per-perturbation")
    print("=" * 85)
    hdr = f"{'Dataset':<12} {'Model':<16} {'Perts':>6} {'Genes':>6} {'EMD':>10} {'EMD_std':>10} {'EDist':>12} {'EDist_std':>12}"
    print(hdr)
    print("-" * 85)
    for _, row in summary_df.iterrows():
        print(
            f"{row['dataset']:<12} {row['model']:<16} "
            f"{int(row['n_perturbations']):>6} {int(row['n_genes']):>6} "
            f"{row['emd_mean']:>10.4f} {row['emd_std']:>10.4f} "
            f"{row['edist_pca_mean']:>12.4f} {row['edist_pca_std']:>12.4f}"
        )
    print("-" * 85)
    print(f"  {len(summary_df)} combinations")
    print("=" * 85)


if __name__ == "__main__":
    main()
