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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from zinb.zinb_utils import fit_batch_zinb, expr_to_gaussian

from gmm_utils import learn_factor_loading, factor_variance

try:
    import tomllib as toml
except ImportError:
    try:
        import tomli as toml
    except ImportError:
        import toml


def parse_args():
    ROOT = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="ZINB preprocessing: h5ad → processed .pt")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--h5ad_path", type=str, default=None)
    parser.add_argument("--toml_path", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default=str(ROOT / "vocab.json"))
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--cell_line", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128, help="Cells per ZINB batch")
    parser.add_argument("--min_batches_per_label", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--perturbation_col", type=str, default="gene")
    parser.add_argument("--cell_line_col", type=str, default="cell_line")
    parser.add_argument("--control_value", type=str, default="non-targeting")
    parser.add_argument("--n_hvg", type=int, default=5000)
    parser.add_argument("--default_cell_line", type=str, default="K562",
                        help="Default cell line name for legacy .pt data (no cell_line column)")
    parser.add_argument("--copula_rank", type=int, default=32)
    parser.add_argument("--copula_eps", type=float, default=1e-6)
    parser.add_argument("--copula_max_ctrl_cells", type=int, default=50000)
    args = parser.parse_args()

    if args.h5ad_path is None:
        
        raw_h5ad = ROOT / "data" / args.dataset / f"{args.dataset}_raw.h5ad"
        default_h5ad = ROOT / "data" / args.dataset / f"{args.dataset}.h5ad"
        nested_h5ad = ROOT / "data" / args.dataset / f"{args.dataset}_hvg.h5ad"
        if raw_h5ad.exists():
            args.h5ad_path = str(raw_h5ad)
        elif default_h5ad.exists():
            args.h5ad_path = str(default_h5ad)
        elif nested_h5ad.exists():
            args.h5ad_path = str(nested_h5ad)
        if default_h5ad.exists():
            args.h5ad_path = str(default_h5ad)
        elif nested_h5ad.exists():
            args.h5ad_path = str(nested_h5ad)
    if args.output_path is None:
        args.output_path = str(ROOT / "data" / "processed" / f"processed_{args.dataset}_zinb.pt")
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
                split_groups[section_name] = {'val': val_set, 'test': test_set}
                split_method = 'explicit'
                print(f"  [TOML] {section_name}: val={len(val_set)}, test={len(test_set)}")

        if 'split' in config:
            sc = config['split']
            if sc.get('method') == 'random':
                split_method = 'random'
                val_ratio = float(sc.get('val_ratio', val_ratio))
                test_ratio = float(sc.get('test_ratio', test_ratio))
                split_seed = int(sc.get('seed', split_seed))
                split_groups['__random__'] = {
                    'val_ratio': val_ratio, 'test_ratio': test_ratio, 'seed': split_seed,
                }
                print(f"  [TOML] Random split: val={val_ratio}, test={test_ratio}, seed={split_seed}")

    if not split_groups:
        split_groups['__random__'] = {
            'val_ratio': val_ratio, 'test_ratio': test_ratio, 'seed': split_seed,
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
                  controlled_line, label_to_split):
    if is_control:
        return 'train'

    
    def _match_gene_set(gene_str, gene_set):
        if gene_str in gene_set:
            return True
        if '+' in gene_str:
            for part in gene_str.split('+'):
                if part.strip() in gene_set:
                    return True
        return False

    if split_method == 'explicit':
        if cell_line == controlled_line:
            groups = split_groups.get(cell_line, {})
            if _match_gene_set(gene_str, groups.get('val', set())):
                return 'val'
            elif _match_gene_set(gene_str, groups.get('test', set())):
                return 'test'
            else:
                return 'train'
        else:
            return 'train'
    elif split_method == 'random':
        if label_to_split is not None:
            
            if gene_str in label_to_split:
                return label_to_split[gene_str]
            if '+' in gene_str:
                for part in gene_str.split('+'):
                    part = part.strip()
                    if part in label_to_split:
                        return label_to_split[part]
            return 'train'
        else:
            return 'train'
    return 'train'


def process_group_task_memmap(task, X_path, X_shape, gene_to_idx, split_groups,
                               split_method, controlled_line, label_to_split,
                               b_size, min_b, control_value,
                               copula_A=None, copula_cm=None, copula_cs=None,
                               copula_eps=1e-6, copula_d=None):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    X = np.memmap(X_path, dtype=np.float32, mode='r', shape=X_shape)
    (cell_line, gene, transcript), group_indices = task

    is_control = (str(transcript) == str(control_value))
    gene_str = str(gene)

    
    
    primary_gene = gene_str
    if '+' in gene_str and not is_control:
        parts = [p.strip() for p in gene_str.split('+')]
        non_ctrl = [p for p in parts if p != str(control_value)]
        if non_ctrl:
            primary_gene = non_ctrl[0]  

    split = resolve_split(cell_line, primary_gene, is_control, split_groups,
                          split_method, controlled_line, label_to_split)

    if is_control:
        label_idx = -1
    else:
        
        label_idx = gene_to_idx.get(gene_str, -1)
        if label_idx == -1 and '+' in gene_str:
            for part in gene_str.split('+'):
                part = part.strip()
                idx = gene_to_idx.get(part, -1)
                if idx >= 0:
                    label_idx = idx
                    break

    batch_X = X[group_indices]
    n_samples = batch_X.shape[0]
    if n_samples < 2:
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
            params = fit_batch_zinb(batch_X[batch_idx], n_jobs=1)
        except Exception as e:
            import warnings
            warnings.warn(f"ZINB fit failed for group {task[0]}: {e}")
            continue

        factor_var = _copula_factor_var(batch_X[batch_idx], params, copula_A, copula_cm, copula_cs, eps=copula_eps, d=copula_d)

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


def load_and_filter_adata(data_path, vocab_path, perturbation_col, control_value,
                          n_hvg=5000, default_cell_line='K562'):
    if data_path.endswith('.h5ad'):
        return _load_h5ad(data_path, vocab_path, perturbation_col, control_value, n_hvg)
    else:
        return _load_legacy_pt(data_path, vocab_path, perturbation_col, control_value,
                               n_hvg, default_cell_line)


def _load_h5ad(h5ad_path, vocab_path, perturbation_col, control_value, n_hvg=5000):
    import scanpy as sc

    print(f"Loading h5ad: {h5ad_path}")
    adata = anndata.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    with open(vocab_path, 'r') as f:
        gene_to_idx = json.load(f)
    valid_genes = set(gene_to_idx.keys())

    
    use_gene_name_col = 'gene_name' in adata.var.columns
    if use_gene_name_col:
        gene_symbols = adata.var['gene_name'].tolist()
        print(f"  Using var['gene_name'] for gene symbols (var_names are e.g. {list(adata.var_names[:2])})")
    else:
        gene_symbols = adata.var_names.tolist()
        print(f"  Using var_names for gene symbols")

    
    gene_to_col = {g: i for i, g in enumerate(gene_symbols)}

    
    if 'highly_variable' in adata.var.columns:
        hvg_mask = adata.var['highly_variable']
        hvg_genes = [gene_symbols[i] for i in range(len(gene_symbols)) if hvg_mask.iloc[i]]
        print(f"  Using {len(hvg_genes)} pre-computed HVGs from h5ad")
    else:
        n_select = min(n_hvg, adata.shape[1])
        print(f"  Running scanpy HVG selection (n_top_genes={n_select})...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_select, subset=False)
        hvg_mask = adata.var['highly_variable']
        hvg_genes = [gene_symbols[i] for i in range(len(gene_symbols)) if hvg_mask.iloc[i]]
        print(f"  HVG selected: {len(hvg_genes)} genes")

    hvg_genes_in_vocab = [g for g in hvg_genes if g in valid_genes]
    n_dropped = len(hvg_genes) - len(hvg_genes_in_vocab)
    print(f"  HVGs in vocab: {len(hvg_genes_in_vocab)} ({n_dropped} dropped)")
    hvg_indices = [gene_to_idx[g] for g in hvg_genes_in_vocab]

    
    keep_cols = [gene_to_col[g] for g in hvg_genes_in_vocab]

    def _keep(g):
        s = str(g)
        if s == str(control_value):
            return True
        
        if '+' in s:
            parts = s.split('+')
            return any(p in valid_genes for p in parts)
        return s in valid_genes
    keep_mask = adata.obs[perturbation_col].apply(_keep).astype(bool)
    adata = adata[keep_mask].copy()
    print(f"  Cells after vocab filter: {adata.shape[0]}")

    
    if 'counts' in adata.layers:
        raw = adata.layers['counts']
        print("  Using raw counts from layers['counts']")
    elif 'count' in adata.layers:
        raw = adata.layers['count']
        print("  Using raw counts from layers['count']")
    else:
        print("  WARNING: no raw counts layer — converting from log1p X back to integer counts")
        if hasattr(adata.X, 'toarray'):
            raw = adata.X.toarray()
        else:
            raw = adata.X
        
        raw = np.round(np.expm1(raw)).astype(np.int32)
    if hasattr(raw, 'toarray'):
        raw = raw.toarray()
    X = raw[:, keep_cols].astype(np.float32)
    print(f"  Final data shape: {X.shape}")
    print(f"  Count range: [{X.min():.1f}, {X.max():.1f}], dtype: {X.dtype}")

    
    adata.var_names = hvg_genes_in_vocab
    if 'gene_name' in adata.var.columns:
        adata.var = adata.var.drop(columns=['gene_name'])

    return X, hvg_indices, gene_to_idx, adata


def _load_legacy_pt(pt_path, vocab_path, perturbation_col, control_value, n_hvg,
                    default_cell_line='K562'):
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
    adata.obs['cell_line'] = default_cell_line

    X = adata.X.toarray().astype(np.float32)
    print(f"  Final data shape: {X.shape}")

    return X, hvg_indices, gene_to_idx, adata


def _copula_factor_var(batch_X, params_np, A, cm, cs, eps=1e-6, d=None):
    
    if A is None:
        return None
    try:
        z = expr_to_gaussian(batch_X, torch.as_tensor(params_np, dtype=torch.float32), eps=eps)
        s = factor_variance(z, A, cm, cs, eps=eps, d=d)
        return s.detach().to(torch.float32).clone()
    except Exception as e:
        import warnings
        warnings.warn(f"[copula] factor_var failed: {e}")
        return None


def _extract_gene_from_condition(cond_str, gene_to_idx):
    
    s = str(cond_str)
    if '+' in s:
        parts = [p.strip() for p in s.split('+')]
        for p in parts:
            if p in gene_to_idx:
                return p
    return s if s in gene_to_idx else None


def build_random_label_split(adata, gene_to_idx, controlled_cell_line,
                              perturbation_col, cell_line_col, control_value,
                              val_ratio, test_ratio, seed):
    rng = np.random.RandomState(seed)
    cl_mask = adata.obs[cell_line_col] == controlled_cell_line
    genes_in_cl = adata[cl_mask].obs[perturbation_col].unique()
    control_str = str(control_value)

    
    pert_conditions = []
    for g in genes_in_cl:
        s = str(g)
        if s == control_str:
            continue
        if _extract_gene_from_condition(s, gene_to_idx) is not None:
            pert_conditions.append(s)
    pert_conditions = sorted(set(pert_conditions))
    rng.shuffle(pert_conditions)

    n_total = len(pert_conditions)
    n_val = max(1, int(n_total * val_ratio))
    n_test = max(1, int(n_total * test_ratio))
    label_to_split = {}
    for g in pert_conditions[:n_val]:
        label_to_split[g] = 'val'
    for g in pert_conditions[n_val:n_val + n_test]:
        label_to_split[g] = 'test'
    for g in pert_conditions[n_val + n_test:]:
        label_to_split[g] = 'train'

    print(f"  Random split for {controlled_cell_line}: "
          f"{n_total} conditions -> train={n_total-n_val-n_test}, "
          f"val={n_val}, test={n_test}")
    return label_to_split


def main():
    args = parse_args()
    print(f"=== ZINB Preprocessing: {args.dataset} ===")

    split_groups, split_method, controlled_line = build_split_config(
        args.toml_path, args.cell_line, args.val_ratio, args.test_ratio, args.split_seed
    )

    X, hvg_indices, gene_to_idx, adata = load_and_filter_adata(
        args.h5ad_path, args.vocab_path,
        args.perturbation_col, args.control_value,
        n_hvg=args.n_hvg,
        default_cell_line=args.default_cell_line,
    )

    
    ctrl_mask_val = adata.obs[args.perturbation_col].astype(str).values == str(args.control_value)
    A, copula_cm, copula_cs = None, None, None
    if args.copula_rank > 0:
        ctrl_idx_cop = np.where(ctrl_mask_val)[0]
        if len(ctrl_idx_cop) == 0:
            print("  [copula] WARNING: no control cells; copula disabled.")
        else:
            if len(ctrl_idx_cop) > args.copula_max_ctrl_cells:
                rng = np.random.RandomState(42)
                ctrl_idx_cop = rng.choice(ctrl_idx_cop, args.copula_max_ctrl_cells, replace=False)
                print(f"  [copula] subsampled controls to {args.copula_max_ctrl_cells}")
            ctrl_X = X[ctrl_idx_cop]
            print(f"  [copula] pooled control ZINB on {ctrl_X.shape[0]} cells x {ctrl_X.shape[1]} genes...")
            ctrl_params = fit_batch_zinb(ctrl_X, n_jobs=args.n_jobs)
            ctrl_params_t = torch.as_tensor(ctrl_params, dtype=torch.float32)
            print("  [copula] transforming controls -> gaussian copula latents...")
            z_ctrl = expr_to_gaussian(ctrl_X, ctrl_params_t, eps=args.copula_eps)
            print(f"  [copula] learning factor loading A (rank={args.copula_rank})...")
            A, copula_cm, copula_cs, copula_psi, copula_d = learn_factor_loading(
                z_ctrl, args.copula_rank, eps=args.copula_eps)
            ortho = float((A.t() @ A - torch.eye(args.copula_rank)).abs().max())
            psi_mean = float(copula_psi.mean()) if copula_psi is not None else 0.0
            explained = float(1.0 - psi_mean)
            s_mean = float(factor_variance(z_ctrl, A, copula_cm, copula_cs, d=copula_d).mean())
            print(f"  [copula] A={tuple(A.shape)} | AᵀA-I max={ortho:.2e} | "
                  f"psi_mean={psi_mean:.4f} s_mean={s_mean:.4f} explained={explained:.4f}")

    if args.cell_line_col in adata.obs.columns:
        cell_lines = sorted(adata.obs[args.cell_line_col].unique())
    else:
        cell_lines = ['unknown']
    print(f"  Cell lines: {cell_lines}")

    if controlled_line is None:
        controlled_line = cell_lines[0] if cell_lines else 'unknown'
    if controlled_line not in cell_lines:
        controlled_line = cell_lines[0]

    label_to_split = None
    if split_method == 'random':
        rg = split_groups.get('__random__', {})
        label_to_split = build_random_label_split(
            adata, gene_to_idx, controlled_line,
            args.perturbation_col, args.cell_line_col, args.control_value,
            rg.get('val_ratio', args.val_ratio),
            rg.get('test_ratio', args.test_ratio),
            rg.get('seed', args.split_seed),
        )

    groupby_cols = [args.cell_line_col, args.perturbation_col]
    extra_cols = [c for c in adata.obs.columns
                  if c not in groupby_cols and c not in ('condition_name', 'dose_val', 'control')]
    if extra_cols:
        groupby_cols.append(extra_cols[0])

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
                split_method, controlled_line, label_to_split,
                args.batch_size, args.min_batches_per_label,
                args.control_value,
                copula_A=A, copula_cm=copula_cm, copula_cs=copula_cs,
                copula_eps=args.copula_eps, copula_d=copula_d,
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

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    
    all_data = train_data + val_data + test_data
    all_lines = sorted(set(d.get('cell_line', 'unknown') for d in all_data if isinstance(d, dict)))
    cell_line_vocab = {cl: i for i, cl in enumerate(all_lines)}

    torch.save({
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'hvg_indices': hvg_indices,
        'vocab': gene_to_idx,
        'distribution': 'zinb',
        'factor_loadings': A,
        'copula_ctrl_mean': copula_cm,
        'copula_ctrl_std': copula_cs,
        'ctrl_psi': copula_psi,
        'copula_d': copula_d,
        'copula_rank': int(args.copula_rank) if A is not None else 0,
        'cell_line_vocab': cell_line_vocab,
        'n_cell_types': len(all_lines),
    }, args.output_path)
    print(f"  Saved: {args.output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
