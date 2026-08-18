import argparse, json, random, sys
import anndata, pandas as pd, numpy as np
from scipy import sparse
from pathlib import Path
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ZINBTransformer
from dataset import ZINBDataset
from sampling import sample_from_params, unpack_params, sample_from_copula

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="ZINB inference for cell-eval")
    parser.add_argument("--dataset", type=str, default="norman")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--raw_data_path", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default=str(ROOT / "vocab.json"))
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--cell_line", type=str, default="K562")
    parser.add_argument("--control_value", type=str, default="ctrl")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--value_floor", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = str(ROOT / "results" / args.dataset / "ours_zinb" / "checkpoints" / "best_model.pt")
    if args.data_path is None:
        args.data_path = str(ROOT / "data" / "processed" / f"processed_{args.dataset}_zinb.pt")
    if args.raw_data_path is None:
        args.raw_data_path = str(ROOT / "data" / args.dataset / f"{args.dataset}.h5ad")
    if args.save_dir is None:
        args.save_dir = str(ROOT / "results" / args.dataset / "ours_zinb" / "inference")
    return args


def align_genes(hvg_indices, idx_to_gene, adata_var_names):
    hvg_genes = [(pos, gid, idx_to_gene[gid]) for pos, gid in enumerate(hvg_indices)
                 if gid in idx_to_gene and idx_to_gene[gid] in set(adata_var_names)]
    common_hvg = [g[1] for g in hvg_genes]
    common_names = [g[2] for g in hvg_genes]
    hvg_to_common = {g[0]: i for i, g in enumerate(hvg_genes)}
    n_missing = len(hvg_indices) - len(common_names)
    if n_missing > 0:
        print(f"Note: {n_missing}/{len(hvg_indices)} HVG genes not found in h5ad.")
    return common_hvg, common_names, hvg_to_common


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    
    print(f"Loading processed data: {args.data_path}")
    processed = torch.load(args.data_path, weights_only=False)
    hvg_indices = processed['hvg_indices']
    A = processed.get('factor_loadings')
    copula_rank = int(processed.get('copula_rank', 0) or 0)
    ctrl_psi = processed.get('ctrl_psi')  
    cell_line_vocab = processed.get('cell_line_vocab', {})
    n_cell_types = int(processed.get('n_cell_types', 0) or 0)

    with open(args.vocab_path) as f:
        vocab = json.load(f)
    idx_to_gene = {v: k for k, v in vocab.items()}
    gene_to_idx = vocab

    print(f"Loading raw data: {args.raw_data_path}")
    adata = anndata.read_h5ad(args.raw_data_path)
    print(f"Loaded h5ad: {adata.shape}")

    
    common_hvg, gene_names, hvg_to_common = align_genes(hvg_indices, idx_to_gene, adata.var_names)
    common_positions = sorted(hvg_to_common.keys())
    print(f"Aligned genes: {len(common_hvg)}")
    adata = adata[:, gene_names].copy()

    
    pert_col = 'gene'
    for c in ['gene', 'condition', 'perturbation', 'target_gene', 'transcript']:
        if c in adata.obs.columns: pert_col = c; break

    cl_col = 'cell_line'
    for c in ['cell_line', 'cell_type']:
        if c in adata.obs.columns: cl_col = c; break

    ctrl_val = args.control_value
    gene_vals = set(str(v) for v in adata.obs[pert_col].unique())
    if ctrl_val not in gene_vals:
        for c in ['non-targeting', 'ctrl', 'control']:
            if c in gene_vals: ctrl_val = c; break

    def _extract_label(raw_val):
        s = str(raw_val)
        if s == str(ctrl_val): return -1
        if '+' in s:
            parts = [p.strip() for p in s.split('+')]
            non_ctrl = [p for p in parts if p != str(ctrl_val)]
            for g in non_ctrl:
                idx = gene_to_idx.get(g, -1)
                if idx >= 0: return idx
        return gene_to_idx.get(s, -2)

    raw_labels = adata.obs[pert_col].apply(_extract_label).values

    
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)

    
    ckpt_d_model = sd['shared_embedding.weight'].shape[1]
    ckpt_nlayers = sum(1 for k in sd if k.startswith('transformer_decoder.layers.') and k.endswith('.norm1.weight'))
    has_copula = any('head_dep' in k for k in sd)
    ckpt_copula_rank = 0
    if has_copula:
        dw = sd.get('head_dep.2.weight')
        if dw is not None: ckpt_copula_rank = dw.shape[0]

    model = ZINBTransformer(
        vocab_size=len(vocab), d_model=ckpt_d_model,
        num_layers=ckpt_nlayers, nhead=args.nhead,
        copula_rank=ckpt_copula_rank, n_cell_types=n_cell_types,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    
    test_dataset = ZINBDataset(args.data_path, split='test', seed=args.seed)
    ctrl_by_cl = {}
    for cb in test_dataset.control_batches:
        cl = cb.get('cell_line', 'unknown')
        ctrl_by_cl.setdefault(cl, []).append(cb)

    rng_ctrl = np.random.RandomState(args.seed)
    ctrl_params_sel = {}; ctrl_s_sel = {}
    for cl, batches in ctrl_by_cl.items():
        sel = batches[rng_ctrl.randint(len(batches))]
        ctrl_params_sel[cl] = sel['params'][common_positions, :]
        fv = sel.get('factor_var')
        ctrl_s_sel[cl] = fv.float() if fv is not None else None

    A_common = A[common_positions, :].contiguous().to(device) if (A is not None and copula_rank > 0) else None
    psi_common = ctrl_psi[common_positions].contiguous().to(device) if (ctrl_psi is not None and copula_rank > 0) else None

    
    ctrl_indices = np.where(raw_labels == -1)[0]
    n_ctrl = min(len(ctrl_indices), 2000)
    rng = np.random.RandomState(args.seed)
    ctrl_idx = rng.choice(ctrl_indices, n_ctrl, replace=len(ctrl_indices) < n_ctrl)
    ctrl_X = adata[ctrl_idx].X
    if sparse.issparse(ctrl_X): ctrl_X = ctrl_X.toarray()
    ctrl_X = np.maximum(ctrl_X, 0)

    
    perturb_keys = sorted(set((cl, lbl) for _, lbl, cl, _ in test_dataset.perturb_samples))
    print(f"Test perturbations: {len(perturb_keys)}")

    cl_arr = adata.obs[cl_col].astype(str).values if cl_col in adata.obs.columns else np.full(adata.n_obs, 'unknown')

    
    torch.manual_seed(args.seed)
    gene_indices_t = torch.tensor(common_hvg, dtype=torch.long, device=device)

    pred_X_list, pred_obs_list = [], []
    real_X_list, real_obs_list = [], []

    pred_X_list.append(ctrl_X); real_X_list.append(ctrl_X)
    for _ in range(n_ctrl):
        pred_obs_list.append({'condition': ctrl_val, 'target_gene': ctrl_val, 'is_control': True})
        real_obs_list.append({'condition': ctrl_val, 'target_gene': ctrl_val, 'is_control': True})

    for cell_line, lbl in tqdm(perturb_keys, desc="Inference"):
        pert_idx = np.where((cl_arr == cell_line) & (raw_labels == lbl))[0]
        if len(pert_idx) < 5: continue

        gene_name = idx_to_gene.get(lbl, f"Gene_{lbl}")
        real_X = adata[pert_idx].X
        if sparse.issparse(real_X): real_X = real_X.toarray()
        real_X = np.maximum(real_X, 0)

        condition = f"{cell_line}__{gene_name}"
        real_X_list.append(real_X)
        for _ in range(len(pert_idx)):
            real_obs_list.append({'condition': condition, 'target_gene': gene_name, 'is_control': False, 'cell_line': cell_line})

        x_ctrl = ctrl_params_sel.get(cell_line)
        if x_ctrl is None: x_ctrl = next(iter(ctrl_params_sel.values()))
        x_ctrl = x_ctrl.unsqueeze(0).to(device)

        s_ctrl = None
        if copula_rank > 0:
            sc = ctrl_s_sel.get(cell_line)
            if sc is None: sc = next((v for v in ctrl_s_sel.values() if v is not None), None)
            s_ctrl = sc.unsqueeze(0).to(device) if sc is not None else None

        label_t = torch.tensor([lbl], device=device)
        ctx_id = torch.tensor([cell_line_vocab.get(cell_line, 0)], dtype=torch.long, device=device) if cell_line_vocab else None

        with torch.no_grad():
            preds = model(x_ctrl, gene_indices_t, label_t, s_ctrl=s_ctrl, cell_context_id=ctx_id)
            if 's' in preds and A_common is not None:
                samples = sample_from_copula(preds, A_common, args.n_samples, psi=psi_common)
            else:
                samples = sample_from_params(preds, args.n_samples)

        pred_X = samples.squeeze(0).cpu().numpy().T
        pred_X = np.clip(pred_X, 0, None)
        
        pred_X = np.log1p(pred_X)
        if args.value_floor > 0:
            pred_X[pred_X < args.value_floor] = args.value_floor

        pred_X_list.append(pred_X)
        for _ in range(args.n_samples):
            pred_obs_list.append({'condition': condition, 'target_gene': gene_name, 'is_control': False, 'cell_line': cell_line})

    
    print("Building AnnData...")
    adata_real_out = anndata.AnnData(X=np.vstack(real_X_list), obs=pd.DataFrame(real_obs_list), var=pd.DataFrame(index=gene_names))
    adata_pred_out = anndata.AnnData(X=np.vstack(pred_X_list), obs=pd.DataFrame(pred_obs_list), var=pd.DataFrame(index=gene_names))
    for ad in (adata_real_out, adata_pred_out):
        ad.obs_names = [f"cell_{i}" for i in range(ad.n_obs)]

    real_path = Path(args.save_dir) / "inference_real.h5ad"
    pred_path = Path(args.save_dir) / "inference_pred.h5ad"
    adata_real_out.write_h5ad(real_path)
    adata_pred_out.write_h5ad(pred_path)
    print(f"Saved: {real_path}  ({adata_real_out.shape})")
    print(f"Saved: {pred_path}  ({adata_pred_out.shape})")
    print("Done.")


if __name__ == "__main__":
    import torch
    main()
