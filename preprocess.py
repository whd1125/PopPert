import torch
import numpy as np
import anndata
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import tempfile
import shutil

from joblib import Parallel, delayed

from gmm_utils import (
    fit_batch_gmm, expr_to_gaussian, learn_factor_loading, factor_variance,
)


try:
    import tomllib as toml
except ImportError:
    try:
        import tomli as toml
    except ImportError:
        import toml


def parse_args():
    ROOT = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Unified preprocessing: h5ad → ZI-GMM processed .pt")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. replogle, adamson, norman)")
    parser.add_argument("--h5ad_path", type=str, default=None, help="Path to input .h5ad file")
    parser.add_argument("--toml_path", type=str, default=None, help="Path to TOML split config")
    parser.add_argument("--vocab_path", type=str, default=str(ROOT / "vocab.json"))
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--cell_line", type=str, default=None,
                        help="Cell line for controlled split (auto-detected from TOML if not set)")
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128, help="Cells per GMM batch")
    parser.add_argument("--min_batches_per_label", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=16, help="Parallel workers")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio (fallback if no TOML)")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio (fallback if no TOML)")
    parser.add_argument("--split_seed", type=int, default=42, help="Random seed for split")
    
    parser.add_argument("--perturbation_col", type=str, default="gene", help="obs column for perturbation label")
    parser.add_argument("--cell_line_col", type=str, default="cell_line", help="obs column for cell line")
    parser.add_argument("--control_value", type=str, default="non-targeting", help="Value in perturbation_col that marks controls")
    parser.add_argument("--n_hvg", type=int, default=5000, help="Number of highly variable genes to select")
    
    parser.add_argument("--perturb_mode", type=str, default="gene", choices=["gene", "drug"],
                        help="'gene' = CRISPR (default, unchanged). 'drug' = chemical perturbation (SMILES + dose).")
    parser.add_argument("--dose_col", type=str, default="dose_val", help="obs column for continuous dosage (drug mode)")
    parser.add_argument("--cell_type_col", type=str, default="cell_type", help="obs column for cell type (drug mode)")
    parser.add_argument("--smiles_col", type=str, default="SMILES", help="obs column for drug SMILES (drug mode)")
    parser.add_argument("--split_column", type=str, default=None,
                        help="obs column with pre-assigned splits for drug mode (e.g. split_ood_finetuning). "
                             "Values mapped: train->train, test->val, ood->test (zero-shot new drugs).")
    parser.add_argument("--fp_dim", type=int, default=1024, help="Morgan fingerprint bit size (drug mode)")
    parser.add_argument("--fp_radius", type=int, default=4, help="Morgan fingerprint radius (drug mode)")
    
    
    parser.add_argument("--copula_rank", type=int, default=32,
                        help="Number of shared copula factors r (Gaussian Copula). 0 disables copula.")
    parser.add_argument("--copula_eps", type=float, default=1e-6, help="Numerical epsilon for copula transforms")
    parser.add_argument("--copula_max_ctrl_cells", type=int, default=50000,
                        help="Cap on control cells used to learn the shared factor loading A")
    parser.add_argument("--copula_diag", action="store_true", default=True,
                        help="Enable per-gene residual variance psi (factor+diag model). "
                             "Use --no-copula_diag to disable.")
    parser.add_argument("--no-copula_diag", action="store_false", dest="copula_diag",
                        help="Disable per-gene residual psi (use factor-only model).")
    
    
    parser.add_argument("--truncated", action="store_true", default=True,
                        help="Use zero-inflated truncated Gaussian mixture (support [0,∞)). "
                             "Use --no-truncated to disable.")
    parser.add_argument("--no-truncated", action="store_false", dest="truncated",
                        help="Disable truncated Gaussian (use standard ZI-GMM).")
    args = parser.parse_args()

    if args.h5ad_path is None:
        default_h5ad = ROOT / "h5ad_data" / f"{args.dataset}_hvg.h5ad"
        nested_h5ad = ROOT / "data" / args.dataset / f"{args.dataset}_hvg.h5ad"
        if default_h5ad.exists():
            args.h5ad_path = str(default_h5ad)
        elif nested_h5ad.exists():
            args.h5ad_path = str(nested_h5ad)
    if args.output_path is None:
        args.output_path = str(ROOT / "data" / "processed" / f"processed_{args.dataset}.pt")
    if args.toml_path is None:
        default_toml = ROOT / "data" / f"{args.dataset}.toml"
        nested_toml = ROOT / "data" / args.dataset / f"{args.dataset}.toml"
        if default_toml.exists():
            args.toml_path = str(default_toml)
        elif nested_toml.exists():
            args.toml_path = str(nested_toml)
    return args


def build_split_config(toml_path, cell_line_override, val_ratio, test_ratio, split_seed):
    
    split_groups = {}  
    split_method = None

    if toml_path and Path(toml_path).exists():
        with open(toml_path, 'rb') as f:
            config = toml.load(f)
        known_keys = {'datasets', 'training', 'split'}
        for section_name in config:
            if section_name in known_keys:
                continue
            section = config[section_name]
            if isinstance(section, dict) and ('val' in section or 'test' in section):
                val_set = set(section.get('val', []))
                test_set = set(section.get('test', []))
                overlap = val_set & test_set
                if overlap:
                    print(f"WARNING: {len(overlap)} genes appear in both val and test for [{section_name}]")
                split_groups[section_name] = {'val': val_set, 'test': test_set}
                split_method = 'explicit'
                print(f"  [TOMl] {section_name}: val={len(val_set)}, test={len(test_set)}")

        if 'split' in config:
            sc = config['split']
            if sc.get('method') == 'random':
                split_method = 'random'
                val_ratio = float(sc.get('val_ratio', val_ratio))
                test_ratio = float(sc.get('test_ratio', test_ratio))
                split_seed = int(sc.get('seed', split_seed))
                
                split_groups['__random__'] = {
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'seed': split_seed,
                }
                print(f"  [TOMl] Random split: val={val_ratio}, test={test_ratio}, seed={split_seed}")

    if not split_groups:
        split_groups['__random__'] = {
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'seed': split_seed,
        }
        split_method = 'random'
        print(f"  [Fallback] Random split: val={val_ratio}, test={test_ratio}, seed={split_seed}")

    
    if cell_line_override:
        controlled_line = cell_line_override
    elif split_method == 'explicit':
        controlled_line = next(iter(split_groups.keys()))
    else:
        controlled_line = None

    return split_groups, split_method, controlled_line


def resolve_split(cell_line, gene_str, is_control, split_groups, split_method,
                  controlled_line, label_to_split, rng_for_random):
    
    if is_control:
        return 'train'

    if split_method == 'explicit':
        if cell_line == controlled_line:
            groups = split_groups.get(cell_line, {})
            if gene_str in groups.get('val', set()):
                return 'val'
            elif gene_str in groups.get('test', set()):
                return 'test'
            else:
                return 'train'
        else:
            return 'train'

    elif split_method == 'random':
        if label_to_split is not None:
            return label_to_split.get(gene_str, 'train')
        else:
            return 'train'

    return 'train'


def process_group_task_memmap(task, X_path, X_shape, gene_to_idx, split_groups,
                               split_method, controlled_line, label_to_split, rng_state,
                               n_comp, b_size, min_b, control_value,
                               copula_A=None, copula_cm=None, copula_cs=None, copula_eps=1e-6,
                               copula_d=None, copula_truncated=True):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    
    
    
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    X = np.memmap(X_path, dtype=np.float32, mode='r', shape=X_shape)
    (cell_line, gene), group_indices = task

    gene_str = str(gene)
    is_control = (gene_str == str(control_value))

    split = resolve_split(cell_line, gene_str, is_control, split_groups,
                          split_method, controlled_line, label_to_split, None)

    if is_control:
        label_idx = -1
    else:
        label_idx = gene_to_idx.get(gene_str, -1)

    batch_X = X[group_indices]
    n_samples = batch_X.shape[0]
    if n_samples < n_comp * 2:
        return [], [], []

    natural_batches = n_samples // b_size
    target_batches = max(min_b, natural_batches)
    local_results = []

    for i in range(target_batches):
        if i < natural_batches:
            start = i * b_size
            end = start + b_size
            batch_idx = np.arange(start, end)
        else:
            if n_samples >= b_size:
                batch_idx = np.random.choice(n_samples, b_size, replace=False)
            else:
                batch_idx = np.random.choice(n_samples, b_size, replace=True)

        try:
            params = fit_batch_gmm(batch_X[batch_idx], n_components=n_comp, n_jobs=1)
        except Exception as e:
            import warnings
            warnings.warn(f"Batch GMM fit failed for group {task[0]}: {e}")
            continue

        factor_var = _copula_factor_var(batch_X[batch_idx], params, copula_A, copula_cm, copula_cs, eps=copula_eps, d=copula_d, truncated=copula_truncated)

        local_results.append({
            'params': torch.tensor(params, dtype=torch.float32),
            'label': int(label_idx),
            'is_control': is_control,
            'cell_line': cell_line,
            'gene': gene_str,
            'factor_var': factor_var,
        })

    if split == 'train':
        return local_results, [], []
    elif split == 'val':
        return [], local_results, []
    else:
        return [], [], local_results


def load_and_filter_adata(data_path, vocab_path, perturbation_col, control_value, n_hvg=5000):
    
    if data_path.endswith('.h5ad'):
        return _load_h5ad(data_path, vocab_path, perturbation_col, control_value, n_hvg)
    else:
        return _load_legacy_pt(data_path, vocab_path, perturbation_col, control_value, n_hvg)


def _load_h5ad(h5ad_path, vocab_path, perturbation_col, control_value, n_hvg=5000):
    
    import scanpy as sc

    print(f"Loading h5ad: {h5ad_path}")
    adata = anndata.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    with open(vocab_path, 'r') as f:
        gene_to_idx = json.load(f)
    valid_genes = set(gene_to_idx.keys())

    if 'highly_variable' in adata.var.columns:
        hvg_mask = adata.var['highly_variable']
        hvg_genes = adata.var_names[hvg_mask].tolist()
        print(f"  Using {len(hvg_genes)} pre-computed HVGs from h5ad")
    else:
        n_select = min(n_hvg, adata.shape[1])
        print(f"  Running scanpy HVG selection (n_top_genes={n_select})...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_select, subset=False)
        hvg_mask = adata.var['highly_variable']
        hvg_genes = adata.var_names[hvg_mask].tolist()
        print(f"  HVG selected: {len(hvg_genes)} genes")

    hvg_genes = [g for g in hvg_genes if g in valid_genes]
    print(f"  HVGs in vocab: {len(hvg_genes)}")

    hvg_indices = [gene_to_idx[g] for g in hvg_genes]

    def _keep(g):
        s = str(g)
        return s in valid_genes or s == str(control_value)
    keep_mask = adata.obs[perturbation_col].apply(_keep).astype(bool)
    adata = adata[keep_mask].copy()
    print(f"  Cells after vocab filter: {adata.shape[0]}")

    orig_hvg_map = {g: i for i, g in enumerate(adata.var_names)}
    keep_cols = [orig_hvg_map[g] for g in hvg_genes if g in orig_hvg_map]

    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X[:, keep_cols].astype(np.float32)
    print(f"  Final data shape: {X.shape}")

    return X, hvg_indices, gene_to_idx, adata


def _load_legacy_pt(pt_path, vocab_path, perturbation_col, control_value, n_hvg):
    
    import scanpy as sc
    from scipy import sparse

    print(f"Loading legacy .pt: {pt_path}")
    data = torch.load(pt_path, weights_only=False)

    with open(vocab_path, 'r') as f:
        gene_to_idx = json.load(f)
    vocab_size = len(gene_to_idx)

    genes_list = data['genes']
    expr_list = data['expressions']
    n_cells = len(genes_list)

    print(f"  Building sparse matrix for {n_cells} cells...")
    all_genes, all_expr, row_indices = [], [], []
    for i in range(n_cells):
        g = genes_list[i].cpu().numpy() if hasattr(genes_list[i], 'cpu') else np.array(genes_list[i])
        e = expr_list[i].cpu().numpy() if hasattr(expr_list[i], 'cpu') else np.array(expr_list[i])
        all_genes.append(g)
        all_expr.append(e)
        row_indices.append(np.full(len(g), i))

    csr = sparse.csr_matrix(
        (np.concatenate(all_expr), (np.concatenate(row_indices), np.concatenate(all_genes))),
        shape=(n_cells, vocab_size))

    adata = sc.AnnData(csr)
    print(f"  Full matrix shape: {adata.shape}")

    
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=False)
    hvg_mask = adata.var['highly_variable']
    hvg_genes = adata.var_names[hvg_mask].tolist()
    hvg_indices = [int(x) for x in hvg_genes]
    adata = adata[:, hvg_genes].copy()
    print(f"  HVG selection: {adata.shape}")

    
    labels = np.array(data['single_ctrl'], dtype=int)
    
    idx_to_gene = {v: k for k, v in gene_to_idx.items()}
    gene_names = []
    for lbl in labels:
        if lbl == -1:
            gene_names.append(str(control_value))
        elif lbl in idx_to_gene:
            gene_names.append(idx_to_gene[lbl])
        else:
            gene_names.append(str(lbl))
    adata.obs[perturbation_col] = gene_names
    adata.obs['transcript'] = gene_names  
    adata.obs['cell_line'] = 'K562'  

    X = adata.X.toarray().astype(np.float32)
    print(f"  Final data shape: {X.shape}")
    print(f"  Added obs: {perturbation_col}, transcript, cell_line")

    return X, hvg_indices, gene_to_idx, adata


def build_random_label_split(adata, gene_to_idx, controlled_cell_line,
                              perturbation_col, cell_line_col, control_value,
                              val_ratio, test_ratio, seed):
    rng = np.random.RandomState(seed)
    cl_mask = adata.obs[cell_line_col] == controlled_cell_line
    genes_in_cl = adata[cl_mask].obs[perturbation_col].unique()
    control_str = str(control_value)
    pert_genes = [g for g in genes_in_cl if str(g) != control_str and str(g) in gene_to_idx]
    pert_genes = sorted(set(str(g) for g in pert_genes))
    rng.shuffle(pert_genes)

    n_val = max(1, int(len(pert_genes) * val_ratio))
    n_test = max(1, int(len(pert_genes) * test_ratio))
    label_to_split = {}
    for g in pert_genes[:n_val]:
        label_to_split[g] = 'val'
    for g in pert_genes[n_val:n_val + n_test]:
        label_to_split[g] = 'test'
    for g in pert_genes[n_val + n_test:]:
        label_to_split[g] = 'train'

    print(f"  Random split for {controlled_cell_line}: "
          f"{len(pert_genes)} genes → train={len(pert_genes)-n_val-n_test}, "
          f"val={n_val}, test={n_test}")
    return label_to_split


def compute_morgan_fingerprints(smiles_list, radius=4, n_bits=1024):
    
    try:
        from rdkit import Chem
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError:
        raise ImportError(
            "rdkit is required for chemical-perturbation (drug) preprocessing. "
            "Install it with: pip install rdkit"
        )
    mfpgen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = {}
    for smi in smiles_list:
        if smi is None:
            continue
        s = str(smi).strip()
        if s == "" or s == "nan":
            continue
        if s in fps:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            print(f"WARNING: could not parse SMILES (skipped): {s}")
            continue
        fps[s] = np.asarray(mfpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    return fps


def learn_copula_loading(X, ctrl_mask, n_components, rank, n_jobs=16,
                         max_ctrl_cells=50000, eps=1e-6, diag=True,
                         truncated=True):
    
    if rank is None or rank <= 0:
        return None, None, None, None, None
    ctrl_idx = np.where(ctrl_mask)[0]
    if len(ctrl_idx) == 0:
        print("  [copula] WARNING: no control cells found; copula disabled.")
        return None, None, None, None, None
    if len(ctrl_idx) > max_ctrl_cells:
        rng = np.random.RandomState(42)
        ctrl_idx = rng.choice(ctrl_idx, max_ctrl_cells, replace=False)
        print(f"  [copula] subsampled controls to {max_ctrl_cells} cells for A learning.")
    ctrl_X = X[ctrl_idx]
    print(f"  [copula] pooled control ZI-GMM on {ctrl_X.shape[0]} cells x {ctrl_X.shape[1]} genes...")
    ctrl_params = fit_batch_gmm(ctrl_X, n_components=n_components, n_jobs=n_jobs)
    ctrl_params_t = torch.as_tensor(ctrl_params, dtype=torch.float32)
    print("  [copula] transforming controls -> Gaussian copula latents...")
    z_ctrl = expr_to_gaussian(ctrl_X, ctrl_params_t, eps=eps, truncated=truncated)
    print(f"  [copula] learning factor loading A (rank={rank}) via PCA on correlation matrix...")
    A, cm, cs, psi, d = learn_factor_loading(z_ctrl, rank, eps=eps)
    ortho = float((A.t() @ A - torch.eye(rank)).abs().max())
    psi_mean = float(psi.mean())
    explained = float(1.0 - psi_mean)  
    
    raw_mean = float(factor_variance(z_ctrl, A, cm, cs).mean())
    s_mean = float(factor_variance(z_ctrl, A, cm, cs, d=d).mean())
    print(f"  [copula] A={tuple(A.shape)} | AᵀA-I max={ortho:.2e} | "
          f"factor_var raw_mean={raw_mean:.4f} s_mean={s_mean:.4f}")
    print(f"  [copula] psi (per-gene residual): mean={psi_mean:.4f}, "
          f"d (projected residual): mean={float(d.mean()):.4f}, "
          f"fraction_explained={explained:.4f} | diag={'on' if diag else 'off'}")
    if not diag:
        psi = None
        d = None
    return A, cm, cs, psi, d


def _copula_factor_var(batch_X, params_np, A, cm, cs, eps=1e-6, d=None,
                       truncated=True):
    
    if A is None:
        return None
    try:
        z = expr_to_gaussian(batch_X, torch.as_tensor(params_np, dtype=torch.float32), eps=eps,
                             truncated=truncated)
        s = factor_variance(z, A, cm, cs, eps=eps, d=d)
        return s.detach().to(torch.float32).clone()
    except Exception as e:
        import warnings
        warnings.warn(f"[copula] factor_var failed for a batch: {e}")
        return None


def _load_h5ad_drug(h5ad_path, vocab_path, n_hvg=5000):
    
    import scanpy as sc

    print(f"Loading h5ad (drug mode): {h5ad_path}")
    adata = anndata.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    with open(vocab_path, 'r') as f:
        gene_to_idx = json.load(f)
    valid_genes = set(gene_to_idx.keys())

    if 'highly_variable' in adata.var.columns:
        hvg_mask = adata.var['highly_variable']
        hvg_genes = adata.var_names[hvg_mask].tolist()
        print(f"  Using {len(hvg_genes)} pre-computed HVGs from h5ad")
    else:
        n_select = min(n_hvg, adata.shape[1])
        print(f"  Running scanpy HVG selection (n_top_genes={n_select})...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_select, subset=False)
        hvg_mask = adata.var['highly_variable']
        hvg_genes = adata.var_names[hvg_mask].tolist()
        print(f"  HVG selected: {len(hvg_genes)} genes")

    hvg_genes = [g for g in hvg_genes if g in valid_genes]
    print(f"  HVGs in vocab: {len(hvg_genes)}")
    hvg_indices = [gene_to_idx[g] for g in hvg_genes]

    orig_hvg_map = {g: i for i, g in enumerate(adata.var_names)}
    keep_cols = [orig_hvg_map[g] for g in hvg_genes if g in orig_hvg_map]

    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X[:, keep_cols].astype(np.float32)
    print(f"  Final data shape: {X.shape}")
    return X, hvg_indices, gene_to_idx, adata






SPLIT_MAP_DRUG = {'train': 'train', 'test': 'val', 'val': 'val', 'ood': 'test'}


def load_drug_ood_drugs(toml_path):
    
    if toml_path is None or not Path(toml_path).exists():
        return None
    with open(toml_path, 'rb') as f:
        cfg = toml.load(f)
    section = cfg.get('ood_drugs', {})
    drugs = section.get('drugs', []) if isinstance(section, dict) else []
    ood = set(str(x) for x in drugs)
    if ood:
        print(f"  [toml] fixed OOD-drug allowlist: {len(ood)} drugs -> test split")
    else:
        print(f"  [toml] no [ood_drugs] allowlist in {toml_path}; test split driven by obs column only")
    return ood or None


def process_drug_group_task_memmap(task, X_path, X_shape, smiles_by_drug, fp_dim,
                                   split_column, control_value, n_comp, b_size, min_b,
                                   copula_A=None, copula_cm=None, copula_cs=None, copula_eps=1e-6,
                                   ood_drugs=None, copula_d=None, copula_truncated=True):
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    
    
    
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    X = np.memmap(X_path, dtype=np.float32, mode='r', shape=X_shape)
    (cell_type, drug, dose_val, split_val), group_indices = task

    drug_str = str(drug)
    is_control = (drug_str == str(control_value))
    smiles = smiles_by_drug.get(drug_str, None)
    drug_dose_name = f"{drug_str}_{dose_val}"

    
    
    
    if ood_drugs is not None and drug_str in ood_drugs:
        split_bucket = 'test'
    else:
        split_bucket = SPLIT_MAP_DRUG.get(str(split_val), 'train')

    batch_X = X[group_indices]
    n_samples = batch_X.shape[0]
    if n_samples < n_comp * 2:
        return [], [], []

    natural_batches = n_samples // b_size
    target_batches = max(min_b, natural_batches)
    local_results = []

    for i in range(target_batches):
        if i < natural_batches:
            start = i * b_size
            batch_idx = np.arange(start, start + b_size)
        else:
            if n_samples >= b_size:
                batch_idx = np.random.choice(n_samples, b_size, replace=False)
            else:
                batch_idx = np.random.choice(n_samples, b_size, replace=True)

        try:
            params = fit_batch_gmm(batch_X[batch_idx], n_components=n_comp, n_jobs=1)
        except Exception as e:
            import warnings
            warnings.warn(f"Batch GMM fit failed for group {task[0]}: {e}")
            continue

        factor_var = _copula_factor_var(batch_X[batch_idx], params, copula_A, copula_cm, copula_cs, eps=copula_eps, d=copula_d, truncated=copula_truncated)

        local_results.append({
            'params': torch.tensor(params, dtype=torch.float32),
            'is_control': is_control,
            'cell_type': str(cell_type),
            'drug_name': drug_str,
            'dose_val': float(dose_val),
            'smiles': smiles,
            'drug_dose_name': drug_dose_name,
            'label': drug_dose_name,   
            'split': split_bucket,
            'factor_var': factor_var,
        })

    if split_bucket == 'train':
        return local_results, [], []
    elif split_bucket == 'val':
        return [], local_results, []
    else:
        return [], [], local_results


def run_drug_preprocess(args):
    
    print(f"=== Preprocessing (DRUG mode): {args.dataset} ===")
    pert_col = args.perturbation_col
    ctrl_val = args.control_value
    ct_col = args.cell_type_col
    dose_col = args.dose_col
    smiles_col = args.smiles_col

    
    if args.h5ad_path and args.h5ad_path.endswith('.h5ad'):
        _peek = anndata.read_h5ad(args.h5ad_path)
        if pert_col not in _peek.obs.columns:
            for c in ['condition', 'perturbation', 'product_name', 'drug', 'gene']:
                if c in _peek.obs.columns:
                    print(f"Auto-detected perturbation column: '{c}'")
                    pert_col = c
                    break
        if pert_col in _peek.obs.columns:
            unique_vals = set(str(v) for v in _peek.obs[pert_col].unique())
            if ctrl_val not in unique_vals:
                for cv in ['control', 'ctrl', 'vehicle', 'non-targeting']:
                    if cv in unique_vals:
                        print(f"Auto-detected control value: '{cv}'")
                        ctrl_val = cv
                        break
        for col, candidates, name in [
            (ct_col, ['cell_type', 'cell_line', 'cell_type_id'], 'cell type'),
            (dose_col, ['dose_val', 'dose', 'dose_character'], 'dose'),
            (smiles_col, ['SMILES', 'smiles'], 'SMILES'),
        ]:
            if col not in _peek.obs.columns:
                for c in candidates:
                    if c in _peek.obs.columns:
                        print(f"Auto-detected {name} column: '{c}'")
                        if name == 'cell type':
                            ct_col = c
                        elif name == 'dose':
                            dose_col = c
                        else:
                            smiles_col = c
                        break
        del _peek

    if args.split_column is None:
        
        args.split_column = 'split_ood_finetuning'
        print(f"  No --split_column given; defaulting to '{args.split_column}'")

    
    
    ood_drugs = load_drug_ood_drugs(args.toml_path)

    
    X, hvg_indices, gene_to_idx, adata = _load_h5ad_drug(
        args.h5ad_path, args.vocab_path, n_hvg=args.n_hvg,
    )

    
    obs = adata.obs
    smiles_by_drug = {}
    for drug, sub in obs.groupby(pert_col, observed=True):
        s = sub[smiles_col].dropna()
        s = s[s.astype(str).str.strip() != '']
        if len(s) > 0:
            smiles_by_drug[str(drug)] = str(s.iloc[0])
    if ctrl_val not in smiles_by_drug:
        
        smiles_by_drug[str(ctrl_val)] = 'CS(C)=O'
    print(f"  Built SMILES map for {len(smiles_by_drug)} drugs (control='{ctrl_val}').")

    
    ctrl_mask = (obs[pert_col].astype(str).values == str(ctrl_val))
    A, copula_cm, copula_cs, copula_psi, copula_d = learn_copula_loading(
        X, ctrl_mask, args.n_components, args.copula_rank,
        n_jobs=args.n_jobs, max_ctrl_cells=args.copula_max_ctrl_cells, eps=args.copula_eps,
        diag=args.copula_diag,
        truncated=args.truncated,
    )

    
    unique_smiles = sorted(set(smiles_by_drug.values()))
    print(f"  Computing Morgan fingerprints (radius={args.fp_radius}, bits={args.fp_dim}) "
          f"for {len(unique_smiles)} SMILES...")
    smiles_to_fp = compute_morgan_fingerprints(unique_smiles, radius=args.fp_radius, n_bits=args.fp_dim)
    print(f"  Fingerprints computed: {len(smiles_to_fp)}/{len(unique_smiles)}")

    
    if args.split_column not in obs.columns:
        raise ValueError(f"split column '{args.split_column}' not found in obs. "
                         f"Available: {list(obs.columns)}")
    groupby_cols = [ct_col, pert_col, dose_col, args.split_column]
    groups = obs.groupby(groupby_cols, observed=True)
    tasks = [(key, group_indices) for key, group_indices in groups.indices.items()]
    print(f"  Total groups (groupby={groupby_cols}): {len(tasks)}")

    
    n_jobs = min(args.n_jobs, len(tasks))
    print(f"  Processing with {n_jobs} workers...")

    temp_dir = tempfile.mkdtemp()
    try:
        memmap_path = os.path.join(temp_dir, 'X.mmap')
        X_memmap = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=X.shape)
        X_memmap[:] = X[:]
        X_memmap.flush()

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_drug_group_task_memmap)(
                task, memmap_path, X.shape, smiles_by_drug, args.fp_dim,
                args.split_column, ctrl_val,
                args.n_components, args.batch_size, args.min_batches_per_label,
                copula_A=A, copula_cm=copula_cm, copula_cs=copula_cs, copula_eps=args.copula_eps,
                ood_drugs=ood_drugs, copula_d=copula_d,
                copula_truncated=args.truncated,
            )
            for task in tqdm(tasks)
        )
    finally:
        shutil.rmtree(temp_dir)

    
    train_data, val_data, test_data = [], [], []
    for t, v, te in results:
        train_data.extend(t)
        val_data.extend(v)
        test_data.extend(te)
    stats = {'train': len(train_data), 'val': len(val_data), 'test': len(test_data)}
    print(f"  Split stats: {stats}")

    
    cell_types = sorted(set(d['cell_type'] for d in train_data + val_data + test_data))
    print(f"  Cell types: {cell_types}")

    
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'hvg_indices': hvg_indices,
        'vocab': gene_to_idx,
        'n_components': args.n_components,
        'perturb_mode': 'drug',
        'smiles_to_fp': smiles_to_fp,
        'fp_dim': args.fp_dim,
        'cell_types': cell_types,
        'n_cell_types': len(cell_types),
        'split_column': args.split_column,
        'factor_loadings': A,
        'copula_ctrl_mean': copula_cm,
        'copula_ctrl_std': copula_cs,
        'ctrl_psi': copula_psi,
        'copula_d': copula_d,
        'copula_rank': int(args.copula_rank) if A is not None else 0,
        'truncated': args.truncated,
    }
    torch.save(save_dict, args.output_path)
    print(f"  Saved: {args.output_path}")
    print("Done.")


def main():
    args = parse_args()
    
    if args.perturb_mode == 'drug':
        run_drug_preprocess(args)
        return
    print(f"=== Preprocessing: {args.dataset} ===")

    
    split_groups, split_method, controlled_line = build_split_config(
        args.toml_path, args.cell_line, args.val_ratio, args.test_ratio, args.split_seed
    )

    
    pert_col = args.perturbation_col
    ctrl_val = args.control_value
    cl_col = args.cell_line_col
    if args.h5ad_path and args.h5ad_path.endswith('.h5ad'):
        import anndata as _ad
        _peek = _ad.read_h5ad(args.h5ad_path)
        
        if pert_col not in _peek.obs.columns:
            for c in ['gene', 'condition', 'perturbation', 'target_gene', 'transcript']:
                if c in _peek.obs.columns:
                    print(f"Auto-detected perturbation column: '{c}' (specified '{pert_col}' not found)")
                    pert_col = c
                    break
        
        if pert_col in _peek.obs.columns:
            unique_vals = set(str(v) for v in _peek.obs[pert_col].unique())
            if ctrl_val not in unique_vals:
                for cv in ['non-targeting', 'ctrl', 'control', 'ntc']:
                    if cv in unique_vals:
                        print(f"Auto-detected control value: '{cv}' (specified '{ctrl_val}' not found)")
                        ctrl_val = cv
                        break
        
        if cl_col not in _peek.obs.columns:
            for c in ['cell_line', 'cell_type', 'cell_line_id']:
                if c in _peek.obs.columns:
                    print(f"Auto-detected cell_line column: '{c}' (specified '{cl_col}' not found)")
                    cl_col = c
                    break

    
    X, hvg_indices, gene_to_idx, adata = load_and_filter_adata(
        args.h5ad_path, args.vocab_path,
        pert_col, ctrl_val,
        n_hvg=args.n_hvg,
    )

    
    ctrl_mask = (adata.obs[pert_col].astype(str).values == str(ctrl_val))
    A, copula_cm, copula_cs, copula_psi, copula_d = learn_copula_loading(
        X, ctrl_mask, args.n_components, args.copula_rank,
        n_jobs=args.n_jobs, max_ctrl_cells=args.copula_max_ctrl_cells, eps=args.copula_eps,
        diag=args.copula_diag,
        truncated=args.truncated,
    )

    
    if cl_col in adata.obs.columns:
        cell_lines = sorted(adata.obs[cl_col].unique())
    else:
        cell_lines = ['unknown']
    print(f"  Cell lines ({cl_col}): {cell_lines}")

    if controlled_line is None:
        controlled_line = cell_lines[0] if cell_lines else 'unknown'
    if controlled_line not in cell_lines:
        print(f"  WARNING: controlled cell line '{controlled_line}' not found. "
              f"Falling back to '{cell_lines[0]}'")
        controlled_line = cell_lines[0]

    
    label_to_split = None
    if split_method == 'random':
        rg = split_groups.get('__random__', {})
        label_to_split = build_random_label_split(
            adata, gene_to_idx, controlled_line,
            pert_col, cl_col, ctrl_val,
            rg.get('val_ratio', args.val_ratio),
            rg.get('test_ratio', args.test_ratio),
            rg.get('seed', args.split_seed),
        )

    
    
    
    groupby_cols = [cl_col, pert_col]

    groups = adata.obs.groupby(groupby_cols, observed=True)
    tasks = [(key, group_indices) for key, group_indices in groups.indices.items()]
    print(f"  Total groups (groupby={groupby_cols}): {len(tasks)}")

    
    n_jobs = min(args.n_jobs, len(tasks))
    print(f"  Processing with {n_jobs} workers...")

    temp_dir = tempfile.mkdtemp()
    try:
        memmap_path = os.path.join(temp_dir, 'X.mmap')
        X_memmap = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=X.shape)
        X_memmap[:] = X[:]
        X_memmap.flush()

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_group_task_memmap)(
                task, memmap_path, X.shape, gene_to_idx, split_groups,
                split_method, controlled_line, label_to_split, None,
                args.n_components, args.batch_size, args.min_batches_per_label,
                ctrl_val,
                copula_A=A, copula_cm=copula_cm, copula_cs=copula_cs, copula_eps=args.copula_eps,
                copula_d=copula_d,
                copula_truncated=args.truncated,
            )
            for task in tqdm(tasks)
        )
    finally:
        shutil.rmtree(temp_dir)

    
    train_data, val_data, test_data = [], [], []
    for t, v, te in results:
        train_data.extend(t)
        val_data.extend(v)
        test_data.extend(te)

    stats = {'train': len(train_data), 'val': len(val_data), 'test': len(test_data)}
    print(f"  Split stats: {stats}")

    
    
    all_data = train_data + val_data + test_data
    all_lines = sorted(set(
        d.get('cell_line', 'unknown') for d in all_data
        if isinstance(d, dict)
    ))
    cell_line_to_idx = {cl: i for i, cl in enumerate(all_lines)}
    n_cell_types = len(all_lines)
    print(f"  Cell-line vocab (gene-mode context adapter): {cell_line_to_idx}")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'hvg_indices': hvg_indices,
        'vocab': gene_to_idx,
        'n_components': args.n_components,
        'factor_loadings': A,
        'copula_ctrl_mean': copula_cm,
        'copula_ctrl_std': copula_cs,
        'ctrl_psi': copula_psi,
        'copula_d': copula_d,
        'copula_rank': int(args.copula_rank) if A is not None else 0,
        'cell_line_vocab': cell_line_to_idx,
        'n_cell_types': n_cell_types,
        'truncated': args.truncated,
    }
    torch.save(save_dict, args.output_path)
    print(f"  Saved: {args.output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
