import math
import torch
import argparse
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
from tqdm import tqdm
import sys
import random
import json

from model import DistributionTransformer
from dataset import DistDataset
from sampling import sample_from_params, unpack_params, sample_from_copula







DATASET_PRESETS = {
    "replogle": {
        "perturb_mode": "gene",
        "perturbation_col": "gene",
        "cell_line_col": "cell_line",
        "cell_line": "hepg2",
        "control_value": "non-targeting",
        
        "raw_data_subpath": "data/replogle/replogle.h5ad",
        
        "model_subdir": "ours_copula_large_lam0.2",
    },
    "norman": {
        "perturb_mode": "gene",
        "perturbation_col": "gene",
        "cell_line_col": "cell_line",
        "cell_line": "K562",
        "control_value": "ctrl",
        "raw_data_subpath": "data/norman/norman.h5ad",
        "model_subdir": "ours_copula_lam0.2",
    },
    "adamson": {
        "perturb_mode": "gene",
        "perturbation_col": "gene",
        "cell_line_col": "cell_line",
        "cell_line": "K562",
        "control_value": "ctrl",
        "raw_data_subpath": "data/adamson/adamson.h5ad",
        "model_subdir": "ours_copula_lam0.3",
    },
    "sciplex": {
        "perturb_mode": "drug",
        "perturbation_col": "condition",
        "control_value": "control",
        "dose_col": "dose_val",
        "cell_type_col": "cell_type",
        "smiles_col": "SMILES",
        "split_column": "split_ood_finetuning",
        "test_split_values": "ood",
        "raw_data_subpath": "data/sciplex/sciplex_complete_middle_subset.h5ad",
        "model_subdir": "ours_copula_lam0.2",
    },
}



_PRESET_KEYS = [
    "perturb_mode", "perturbation_col", "cell_line_col", "cell_line",
    "control_value", "dose_col", "cell_type_col", "smiles_col",
    "split_column", "test_split_values", "fp_dim",
]


def parse_args():
    ROOT = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate Predicted and Real AnnData for cell-eval")
    parser.add_argument("--dataset", type=str, default="replogle", help="Dataset name")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, default=None, help="Path to processed data .pt file")
    parser.add_argument("--raw_data_path", type=str, default=None, help="Path to raw h5ad file")
    parser.add_argument("--vocab_path", type=str, default=str(ROOT / "vocab.json"))
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for inference results")
    parser.add_argument("--n_samples", type=int, default=500, help="Cells to sample per perturbation")
    parser.add_argument("--cell_line", type=str, default=None, help="Cell line to filter (auto from preset)")
    parser.add_argument("--cell_line_col", type=str, default=None, help="obs column for cell line (gene mode, auto from preset)")
    parser.add_argument("--control_value", type=str, default=None, help="Value marking control cells (auto from preset)")
    parser.add_argument("--perturbation_col", type=str, default=None,
                        help="obs column for the perturbation/drug label (auto from preset)")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    
    parser.add_argument("--perturb_mode", type=str, default=None, choices=["gene", "drug"],
                        help="Perturbation mode. Auto-detected from processed .pt; override with this flag.")
    parser.add_argument("--fp_dim", type=int, default=1024, help="Morgan fingerprint dimension (drug mode)")
    parser.add_argument("--dose_col", type=str, default=None, help="obs column for dosage (drug mode, auto from preset)")
    parser.add_argument("--cell_type_col", type=str, default=None, help="obs column for cell type (drug mode, auto from preset)")
    parser.add_argument("--smiles_col", type=str, default=None, help="obs column for SMILES (drug mode, auto from preset)")
    parser.add_argument("--split_column", type=str, default=None,
                        help="obs column used to pick test cells from the raw h5ad (drug mode, auto from preset)")
    parser.add_argument("--test_split_values", type=str, default=None,
                        help="comma-separated split values defining the test set (drug mode), e.g. 'ood' or 'test'")
    parser.add_argument("--value_floor", type=float, default=0.0,
                        help="Positive floor applied to all output values before writing h5ad. "
                            "cell-eval/pdex computes fold change as log2(target/control) with NO pseudocount, "
                            "so control genes with mean==0 yield inf/NaN. A small floor (0.01) eliminates this "
                            "Set to a small positive value (e.g. 0.01) to suppress these. Default 0 (disabled).")
    parser.add_argument("--run_tag", type=str, default=None,
                        help="Training run tag (e.g. 'copula_lam0.2'). Auto-derived from preset if not set.")
    
    args = parser.parse_args()

    
    preset = DATASET_PRESETS.get(args.dataset, {})
    if preset:
        print(f"Dataset preset: {args.dataset}")
        for key in _PRESET_KEYS:
            if getattr(args, key, None) is None and key in preset:
                setattr(args, key, preset[key])

    
    if args.perturb_mode is None:
        args.perturb_mode = "gene"
    if args.perturbation_col is None:
        args.perturbation_col = "gene" if args.perturb_mode == "gene" else "condition"
    if args.cell_line_col is None:
        args.cell_line_col = "cell_line"
    if args.control_value is None:
        args.control_value = "non-targeting"
    if args.dose_col is None:
        args.dose_col = "dose_val"
    if args.cell_type_col is None:
        args.cell_type_col = "cell_type"
    if args.smiles_col is None:
        args.smiles_col = "SMILES"
    if args.split_column is None:
        args.split_column = "split_ood_finetuning"
    if args.test_split_values is None:
        args.test_split_values = "ood"

    
    run_tag = args.run_tag or preset.get("model_subdir", "ours")
    if args.model_path is None:
        args.model_path = str(ROOT / "results" / args.dataset / run_tag / "checkpoints" / "best_model.pt")
    if args.data_path is None:
        args.data_path = str(ROOT / "data" / "processed" / f"processed_{args.dataset}.pt")
    if args.raw_data_path is None:
        raw_sub = preset.get("raw_data_subpath")
        if raw_sub:
            args.raw_data_path = str(ROOT / raw_sub)
        else:
            args.raw_data_path = str(ROOT / "h5ad_data" / f"{args.dataset}_hvg.h5ad")
    if args.save_dir is None:
        args.save_dir = str(ROOT / "results" / args.dataset / run_tag / "inference")

    print(f"  control_value     = {args.control_value}")
    print(f"  perturbation_col  = {args.perturbation_col}")
    print(f"  cell_line_col     = {args.cell_line_col}")
    print(f"  cell_line         = {args.cell_line}")
    print(f"  perturb_mode      = {args.perturb_mode}")
    print(f"  model_path        = {args.model_path}")
    print(f"  data_path         = {args.data_path}")
    print(f"  raw_data_path     = {args.raw_data_path}")
    print(f"  save_dir          = {args.save_dir}")
    return args


def align_genes(hvg_indices, idx_to_gene, adata_var_names):
    
    
    hvg_genes = []
    for pos, gid in enumerate(hvg_indices):
        name = idx_to_gene.get(gid)
        if name:
            hvg_genes.append((pos, gid, name))

    adata_gene_set = set(adata_var_names)

    common_hvg_indices = []
    common_gene_names = []
    hvg_to_common = {}  

    for pos, gid, name in hvg_genes:
        if name in adata_gene_set:
            common_pos = len(common_hvg_indices)
            common_hvg_indices.append(gid)
            common_gene_names.append(name)
            hvg_to_common[pos] = common_pos

    n_missing = len(hvg_indices) - len(common_hvg_indices)
    if n_missing > 0:
        print(f"Note: {n_missing}/{len(hvg_indices)} HVG genes not found in h5ad, excluded.")

    return common_hvg_indices, common_gene_names, hvg_to_common


def load_model_strict(model_path, model_kwargs, device, pretrained_embedding_path=None):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    
    emb = state_dict.get('shared_embedding.weight')
    if emb is None:
        raise ValueError("Checkpoint missing 'shared_embedding.weight' — not a DistributionTransformer?")
    ckpt_vocab_size, ckpt_d_model = tuple(emb.shape)

    
    pe0 = state_dict.get('param_encoder.0.weight')
    ckpt_param_dim = int(pe0.shape[1]) if pe0 is not None else None
    ckpt_n_components = (ckpt_param_dim - 1) // 3 if ckpt_param_dim is not None else None

    ckpt_nlayers = sum(1 for k in state_dict
                       if k.startswith('transformer_decoder.layers.')
                       and k.endswith('.norm1.weight'))

    ckpt_has_copula = any('head_dep' in k for k in state_dict)
    ckpt_copula_rank = 0
    if ckpt_has_copula:
        dep_weight = state_dict.get('head_dep.2.weight')
        if dep_weight is not None:
            ckpt_copula_rank = int(dep_weight.shape[0])

    ckpt_has_cell_context = any('cell_context_embedding' in k for k in state_dict)

    
    ff0 = state_dict.get('transformer_decoder.layers.0.linear1.weight')
    ckpt_dim_ff = int(ff0.shape[0]) if ff0 is not None else None

    
    errors = []
    def _check(key, ckpt_val, arg_val):
        if ckpt_val is not None and arg_val is not None and ckpt_val != arg_val:
            errors.append(f"{key}: checkpoint={ckpt_val} vs requested={arg_val}")

    _check('vocab_size', ckpt_vocab_size, model_kwargs.get('vocab_size'))
    _check('d_model', ckpt_d_model, model_kwargs.get('d_model'))
    _check('n_components', ckpt_n_components, model_kwargs.get('n_components'))
    _check('num_layers', ckpt_nlayers, model_kwargs.get('num_layers'))
    _check('copula_rank', ckpt_copula_rank, model_kwargs.get('copula_rank'))

    if errors:
        print("ERROR: Checkpoint / model config mismatch:")
        for e in errors:
            print(f"  - {e}")
        print("  Ensure --d_model, --nlayers, --nhead, --n_components, and copula config "
              "match the training run that produced this checkpoint.")
        raise ValueError("Model config mismatch")

    
    model = DistributionTransformer(
        vocab_size=ckpt_vocab_size,
        n_components=ckpt_n_components,
        d_model=ckpt_d_model,
        num_layers=ckpt_nlayers,
        nhead=model_kwargs.get('nhead', 4),
        dim_feedforward=ckpt_dim_ff if ckpt_dim_ff else model_kwargs.get('dim_feedforward', 1024),
        perturb_mode=model_kwargs.get('perturb_mode', 'gene'),
        fp_dim=model_kwargs.get('fp_dim', 1024),
        n_cell_types=model_kwargs.get('n_cell_types', 0),
        copula_rank=ckpt_copula_rank,
    ).to(device)

    
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    
    if missing or unexpected:
        raise RuntimeError(f"Unexpected strict-load failure: missing={missing}, unexpected={unexpected}")

    print(f"Model loaded (strict): vocab={ckpt_vocab_size}, d_model={ckpt_d_model}, "
          f"K={ckpt_n_components}, layers={ckpt_nlayers}, ff={ckpt_dim_ff}, copula_rank={ckpt_copula_rank}"
          + (f", cell_context={ckpt_has_cell_context}" if ckpt_has_cell_context else ""))

    
    if pretrained_embedding_path and Path(pretrained_embedding_path).exists():
        n_restored = 0
        try:
            pt_state = torch.load(pretrained_embedding_path, map_location=device, weights_only=False)
            pt_emb = pt_state.get('module.embedding.weight')
            if pt_emb is not None and pt_emb.shape == model.shared_embedding.weight.shape:
                trained_emb = model.shared_embedding.weight.data
                zero_mask = trained_emb.norm(dim=1) < 1e-8
                n_zero = int(zero_mask.sum().item())
                if n_zero > 0:
                    trained_emb[zero_mask] = pt_emb[zero_mask].to(device)
                    n_restored = n_zero
        except Exception as e:
            print(f"  [load] pretrained embedding restore skipped: {e}")
        if n_restored > 0:
            print(f"  [load] restored {n_restored} zero-embedding rows from pretrained file")

    model.eval()
    return model


def run_drug_inference(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    
    print(f"Loading processed data: {args.data_path}")
    processed_info = torch.load(args.data_path, weights_only=False)
    hvg_indices = processed_info['hvg_indices']
    n_components = processed_info.get('n_components', 2)
    fp_dim = processed_info.get('fp_dim', args.fp_dim)
    cell_types = processed_info.get('cell_types', [])
    cell_type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    smiles_to_fp = processed_info.get('smiles_to_fp', {})
    truncated = processed_info.get('truncated', False)
    
    A = processed_info.get('factor_loadings')
    copula_rank = int(processed_info.get('copula_rank', 0) or 0)
    ctrl_psi = processed_info.get('ctrl_psi')  

    
    with open(args.vocab_path, 'r') as f:
        vocab_json = json.load(f)
    idx_to_gene = {v: k for k, v in vocab_json.items()}
    vocab_size = len(vocab_json)

    
    print(f"Loading raw data: {args.raw_data_path}")
    adata = anndata.read_h5ad(args.raw_data_path)
    print(f"Loaded h5ad: {adata.shape}")

    pert_col = args.perturbation_col
    ctrl_val = args.control_value
    dose_col = args.dose_col
    ct_col = args.cell_type_col

    
    _obs_cols = list(adata.obs.columns)

    if pert_col not in _obs_cols:
        for candidate in ['condition', 'perturbation', 'drug', 'gene', 'product_name']:
            if candidate in _obs_cols:
                print(f"Auto-detected perturbation_col: '{candidate}' "
                      f"(specified '{pert_col}' not in obs)")
                pert_col = candidate
                break

    if ct_col not in _obs_cols:
        for candidate in ['cell_type', 'cell_line']:
            if candidate in _obs_cols:
                print(f"Auto-detected cell_type_col: '{candidate}' "
                      f"(specified '{ct_col}' not in obs)")
                ct_col = candidate
                break

    if dose_col not in _obs_cols:
        for candidate in ['dose_val', 'dose', 'dose_character']:
            if candidate in _obs_cols:
                print(f"Auto-detected dose_col: '{candidate}' "
                      f"(specified '{dose_col}' not in obs)")
                dose_col = candidate
                break

    if ctrl_val not in set(str(v) for v in adata.obs[pert_col].unique()):
        for candidate in ['control', 'ctrl', 'vehicle', 'non-targeting']:
            if candidate in set(str(v) for v in adata.obs[pert_col].unique()):
                print(f"Auto-detected control_value: '{candidate}' "
                      f"(specified '{ctrl_val}' not found)")
                ctrl_val = candidate
                break

    
    args.perturbation_col = pert_col
    args.cell_type_col = ct_col
    args.dose_col = dose_col
    args.control_value = ctrl_val

    
    common_hvg_indices, gene_names, hvg_to_common = align_genes(
        hvg_indices, idx_to_gene, adata.var_names
    )
    print(f"Aligned genes: {len(common_hvg_indices)} common genes for inference")
    adata = adata[:, gene_names].copy()

    
    pretrained_path = str(Path(__file__).resolve().parent / "embedding_weight" / "best_model.pt")
    model = load_model_strict(
        args.model_path,
        model_kwargs=dict(
            vocab_size=vocab_size, n_components=n_components,
            d_model=args.d_model, num_layers=args.nlayers, nhead=args.nhead,
            n_cell_types=len(cell_types), copula_rank=copula_rank,
            perturb_mode='drug', fp_dim=fp_dim,
        ),
        device=device,
        pretrained_embedding_path=pretrained_path,
    )

    
    test_dataset = DistDataset(args.data_path, split='test', seed=42)
    common_positions = sorted(hvg_to_common.keys())

    
    
    ctrl_by_ct = {}
    for cb in test_dataset.control_batches:
        ct = cb.get('cell_type', 'unknown')
        ctrl_by_ct.setdefault(ct, []).append(cb)

    rng_ctrl = np.random.RandomState(42)
    ctrl_params_sel = {}   
    ctrl_s_sel = {}        
    for ct, batches in ctrl_by_ct.items():
        idx = rng_ctrl.randint(len(batches))
        sel = batches[idx]
        ctrl_params_sel[ct] = sel['params'][common_positions, :]
        fv = sel.get('factor_var')
        ctrl_s_sel[ct] = fv.float() if fv is not None else None
    print(f"Control selected (1 batch/cell_type, matching training): "
          f"{list(ctrl_params_sel.keys())}")

    
    A_common = A[common_positions, :].contiguous() if (A is not None and copula_rank > 0) else None
    psi_common = ctrl_psi[common_positions].contiguous() if (ctrl_psi is not None and copula_rank > 0) else None
    if A_common is not None:
        print(f"Copula A sliced to common genes: {tuple(A_common.shape)}")

    gene_indices = torch.tensor(common_hvg_indices, dtype=torch.long, device=device)

    
    ctrl_mask = (adata.obs[pert_col].astype(str) == str(ctrl_val))
    if 'control' in adata.obs.columns:
        ctrl_mask = ctrl_mask | (adata.obs['control'].astype(str) == '1')
    ctrl_indices_all = np.where(ctrl_mask.values)[0]
    rng = np.random.RandomState(42)
    n_ctrl = min(len(ctrl_indices_all), 2000)
    ctrl_idx = rng.choice(ctrl_indices_all, n_ctrl, replace=(len(ctrl_indices_all) < n_ctrl))
    ctrl_X = adata[ctrl_idx].X
    if sparse.issparse(ctrl_X):
        ctrl_X = ctrl_X.toarray()
    ctrl_X = np.maximum(ctrl_X, 0)
    
    
    if args.value_floor > 0:
        ctrl_X[ctrl_X < args.value_floor] = args.value_floor

    
    test_values = set(v.strip() for v in args.test_split_values.split(',') if v.strip())
    if args.split_column in adata.obs.columns:
        test_cell_mask = adata.obs[args.split_column].astype(str).isin(test_values).values
    else:
        print(f"WARNING: split column '{args.split_column}' not in obs; using all non-control cells as test.")
        test_cell_mask = (~ctrl_mask).values
    print(f"Test cells ({args.split_column} in {test_values}): {int(test_cell_mask.sum())}")

    obs = adata.obs
    ct_arr = obs[ct_col].astype(str).values
    cond_arr = obs[pert_col].astype(str).values
    dose_arr = obs[dose_col].astype(float).values

    
    test_groups = {}
    for params, drug_dose_name, ct, smiles, dose_val, _fv in test_dataset.perturb_samples:
        key = (ct, drug_dose_name)
        if key not in test_groups:
            test_groups[key] = {'smiles': smiles, 'dose_val': float(dose_val),
                                'drug_name': drug_dose_name.rsplit('_', 1)[0]}
    print(f"Test groups (cell_type x drug_dose): {len(test_groups)}")

    pred_X_list, pred_obs_list = [], []
    real_X_list, real_obs_list = [], []

    
    pred_X_list.append(ctrl_X)
    real_X_list.append(ctrl_X)
    for _ in range(n_ctrl):
        pred_obs_list.append({'condition': ctrl_val, 'target_gene': ctrl_val, 'is_control': True})
        real_obs_list.append({'condition': ctrl_val, 'target_gene': ctrl_val, 'is_control': True})

    
    
    torch.manual_seed(42)

    for (ct, drug_dose_name), info in tqdm(test_groups.items(), desc="Inference"):
        smiles = info['smiles']
        dose_val = info['dose_val']
        drug_name = info['drug_name']

        
        match = (test_cell_mask
                & (ct_arr == ct)
                & (cond_arr == drug_name)
                & np.isclose(dose_arr, dose_val, atol=1e-6))
        real_idx = np.where(match)[0]
        if len(real_idx) < 5:
            continue

        
        real_X = adata[real_idx].X
        if sparse.issparse(real_X):
            real_X = real_X.toarray()
        real_X = np.maximum(real_X, 0)
        if args.value_floor > 0:
            real_X[real_X < args.value_floor] = args.value_floor
        real_X_list.append(real_X)

        
        x_ctrl = ctrl_params_sel.get(ct)
        if x_ctrl is None:
            
            x_ctrl = next(iter(ctrl_params_sel.values()))
        x_ctrl = x_ctrl.unsqueeze(0).to(device)  

        s_ctrl = None
        if copula_rank > 0:
            sc = ctrl_s_sel.get(ct)
            if sc is None:
                sc = next((v for v in ctrl_s_sel.values() if v is not None), None)
            s_ctrl = sc.unsqueeze(0).to(device) if sc is not None else None

        fp = smiles_to_fp.get(smiles)
        if fp is None:
            
            fp = np.zeros(fp_dim, dtype=np.float32)
        drug_fp = torch.tensor(fp, dtype=torch.float32, device=device).unsqueeze(0)
        
        dose_log10 = math.log10(max(float(dose_val), 1e-6))
        dose_t = torch.tensor([dose_log10], dtype=torch.float32, device=device)
        ct_t = torch.tensor([cell_type_to_idx.get(ct, 0)], dtype=torch.long, device=device)

        with torch.no_grad():
            preds = model(x_ctrl, gene_indices, drug_fp=drug_fp, dose=dose_t, cell_type=ct_t, s_ctrl=s_ctrl)
            if 's' in preds and A_common is not None:
                samples = sample_from_copula(preds, A_common, args.n_samples, psi=psi_common,
                                             truncated=truncated)
            else:
                samples = sample_from_params(preds, args.n_samples, truncated=truncated)
        pred_X = samples.squeeze(0).cpu().numpy().T  
        
        
        pred_X = np.minimum(pred_X, 14.9) if truncated else np.clip(pred_X, 0, 14.9)
        
        
        if args.value_floor > 0:
            pred_X[pred_X < args.value_floor] = args.value_floor
        pred_X_list.append(pred_X)

        condition = f"{ct}__{drug_dose_name}"
        for _ in range(len(real_idx)):
            real_obs_list.append({'condition': condition, 'target_gene': drug_name,
                                  'is_control': False, 'cell_type': ct, 'dose': dose_val})
        for _ in range(args.n_samples):
            pred_obs_list.append({'condition': condition, 'target_gene': drug_name,
                                'is_control': False, 'cell_type': ct, 'dose': dose_val})

    
    n_perturb_conditions = sum(1 for d in real_obs_list if not d.get('is_control', False))
    if n_perturb_conditions == 0:
        raise RuntimeError(
            "No perturbation cells were matched!  Possible causes:\n"
            f"  - control_value mismatch: using '{args.control_value}' but h5ad may use a "
            f"different label (check obs['{args.perturbation_col}'].unique())\n"
            f"  - perturbation_col mismatch: using '{args.perturbation_col}' but h5ad "
            f"obs columns are {list(adata.obs.columns)}\n"
            f"  - test_split_values mismatch: using '{args.test_split_values}' "
            f"but split column '{args.split_column}' may have different values\n"
            f"  - h5ad/processed data gene mismatch: drugs in test split not found in h5ad\n"
            f"  - raw_data_path points to wrong file: '{args.raw_data_path}'\n"
            f"Run with the correct --dataset flag to auto-configure these values."
        )
    print("Building AnnData objects...")
    adata_real = anndata.AnnData(X=np.vstack(real_X_list), obs=pd.DataFrame(real_obs_list))
    adata_pred = anndata.AnnData(X=np.vstack(pred_X_list), obs=pd.DataFrame(pred_obs_list))
    for ad in (adata_real, adata_pred):
        assert ad.shape[1] == len(gene_names), \
            f"Gene count mismatch: data has {ad.shape[1]} columns, {len(gene_names)} gene names"
        ad.var_names = gene_names
        ad.obs_names = [f"cell_{i}" for i in range(ad.n_obs)]

    real_path = Path(args.save_dir) / "inference_real.h5ad"
    pred_path = Path(args.save_dir) / "inference_pred.h5ad"
    adata_real.write_h5ad(real_path)
    adata_pred.write_h5ad(pred_path)
    print(f"Saved: {real_path}  ({adata_real.shape})")
    print(f"Saved: {pred_path}  ({adata_pred.shape})")
    print(f"\nRun cell-eval:\n  cell-eval run -ap {pred_path} -ar {real_path} --control-pert {ctrl_val} "
        f"-o {Path(args.save_dir) / 'cell_eval'}")
    print("Done.")


def main():
    args = parse_args()
    
    processed_info_peek = torch.load(args.data_path, weights_only=False) \
        if Path(args.data_path).exists() else {}
    if processed_info_peek.get('perturb_mode', 'gene') == 'drug' or args.perturb_mode == 'drug':
        run_drug_inference(args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    
    print(f"Loading processed data: {args.data_path}")
    processed_info = torch.load(args.data_path, weights_only=False)
    hvg_indices = processed_info['hvg_indices']  
    n_components = processed_info.get('n_components', 2)
    truncated = processed_info.get('truncated', False)
    
    A = processed_info.get('factor_loadings')
    copula_rank = int(processed_info.get('copula_rank', 0) or 0)
    ctrl_psi = processed_info.get('ctrl_psi')  
    
    cell_line_vocab = processed_info.get('cell_line_vocab', {})
    n_cell_types = int(processed_info.get('n_cell_types', 0) or 0)

    
    with open(args.vocab_path, 'r') as f:
        vocab_json = json.load(f)
    idx_to_gene = {v: k for k, v in vocab_json.items()}
    gene_to_idx = vocab_json
    vocab_size = len(vocab_json)

    
    print(f"Loading raw data: {args.raw_data_path}")
    adata = anndata.read_h5ad(args.raw_data_path)
    print(f"Loaded h5ad: {adata.shape}")

    
    
    
    
    

    
    common_hvg_indices, gene_names, hvg_to_common = align_genes(
        hvg_indices, idx_to_gene, adata.var_names
    )
    print(f"Aligned genes: {len(common_hvg_indices)} common genes for inference")

    
    adata = adata[:, gene_names].copy()

    
    
    
    _obs_cols = list(adata.obs.columns)

    
    
    if args.perturbation_col not in _obs_cols:
        for candidate in ['gene', 'condition', 'perturbation', 'target_gene']:
            if candidate in _obs_cols:
                print(f"Auto-detected perturbation_col: '{candidate}' "
                      f"(specified '{args.perturbation_col}' not in obs)")
                args.perturbation_col = candidate
                break

    
    
    if args.cell_line_col not in _obs_cols:
        for candidate in ['cell_line', 'cell_type']:
            if candidate in _obs_cols:
                print(f"Auto-detected cell_line_col: '{candidate}' "
                      f"(specified '{args.cell_line_col}' not in obs)")
                args.cell_line_col = candidate
                break

    
    pert_col_name = args.perturbation_col
    gene_vals = set(str(v) for v in adata.obs[pert_col_name].unique())
    ctrl_val = args.control_value
    if ctrl_val not in gene_vals:
        for candidate in ['non-targeting', 'ctrl', 'control', 'ntc']:
            if candidate in gene_vals:
                print(f"Auto-detected control_value: '{candidate}' "
                      f"(specified '{ctrl_val}' not found)")
                ctrl_val = candidate
                break

    
    args.control_value = ctrl_val

    raw_labels = adata.obs[pert_col_name].apply(
        lambda g: -1 if str(g) == str(ctrl_val) else gene_to_idx.get(str(g), -2)
    ).values

    
    pretrained_path = str(Path(__file__).resolve().parent / "embedding_weight" / "best_model.pt")
    model = load_model_strict(
        args.model_path,
        model_kwargs=dict(
            vocab_size=vocab_size, n_components=n_components,
            d_model=args.d_model, num_layers=args.nlayers, nhead=args.nhead,
            n_cell_types=n_cell_types, copula_rank=copula_rank,
            perturb_mode='gene',
        ),
        device=device,
        pretrained_embedding_path=pretrained_path,
    )

    
    test_dataset = DistDataset(args.data_path, split='test', seed=42)

    common_positions = sorted(hvg_to_common.keys())  

    
    
    
    
    
    ctrl_by_cl = {}
    for cb in test_dataset.control_batches:
        cl = cb.get('cell_line', 'unknown')
        ctrl_by_cl.setdefault(cl, []).append(cb)

    rng_ctrl = np.random.RandomState(42)
    ctrl_params_sel = {}   
    ctrl_s_sel = {}        
    for cl, batches in ctrl_by_cl.items():
        idx = rng_ctrl.randint(len(batches))
        sel = batches[idx]
        ctrl_params_sel[cl] = sel['params'][common_positions, :]
        fv = sel.get('factor_var')
        ctrl_s_sel[cl] = fv.float() if fv is not None else None
    print(f"Control selected (1 batch/cell_line, matching training): "
          f"{list(ctrl_params_sel.keys())}")

    
    
    
    perturb_keys = sorted(set(
        (cl, lbl) for _, lbl, cl, _fv in test_dataset.perturb_samples
    ))
    print(f"Test perturbations: {len(perturb_keys)} unique (cell_line, label) pairs")

    
    A_common = A[common_positions, :].contiguous() if (A is not None and copula_rank > 0) else None
    psi_common = ctrl_psi[common_positions].contiguous() if (ctrl_psi is not None and copula_rank > 0) else None
    if A_common is not None:
        print(f"Copula A sliced to common genes: {tuple(A_common.shape)}")

    
    ctrl_indices = np.where(raw_labels == -1)[0]
    if len(ctrl_indices) == 0:
        print("Error: no control cells (label=-1) in raw data")
        return

    rng = np.random.RandomState(42)
    n_ctrl = min(len(ctrl_indices), 2000)
    ctrl_idx = rng.choice(ctrl_indices, n_ctrl,
                        replace=(len(ctrl_indices) < n_ctrl))
    ctrl_X = adata[ctrl_idx].X
    if sparse.issparse(ctrl_X):
        ctrl_X = ctrl_X.toarray()
    ctrl_X = np.maximum(ctrl_X, 0)
    
    
    if args.value_floor > 0:
        ctrl_X[ctrl_X < args.value_floor] = args.value_floor

    
    gene_indices = torch.tensor(common_hvg_indices, dtype=torch.long, device=device)

    
    torch.manual_seed(42)

    pred_X_list, pred_obs_list = [], []
    real_X_list, real_obs_list = [], []

    
    pred_X_list.append(ctrl_X)
    real_X_list.append(ctrl_X)
    for _ in range(n_ctrl):
        pred_obs_list.append({'condition': ctrl_val, 'target_gene': ctrl_val, 'is_control': True})
        real_obs_list.append({'condition': ctrl_val, 'target_gene': ctrl_val, 'is_control': True})

    
    cl_arr = adata.obs[args.cell_line_col].astype(str).values \
        if args.cell_line_col in adata.obs.columns \
        else np.full(adata.n_obs, 'unknown', dtype=str)

    for cell_line, lbl in tqdm(perturb_keys, desc="Inference"):
        gene_name = idx_to_gene.get(lbl, f"Gene_{lbl}")

        
        
        pert_idx = np.where((cl_arr == cell_line) & (raw_labels == lbl))[0]
        if len(pert_idx) < 5:
            continue

        
        real_X = adata[pert_idx].X
        if sparse.issparse(real_X):
            real_X = real_X.toarray()
        real_X = np.maximum(real_X, 0)
        if args.value_floor > 0:
            real_X[real_X < args.value_floor] = args.value_floor

        condition = f"{cell_line}__{gene_name}"
        real_X_list.append(real_X)
        for _ in range(len(pert_idx)):
            real_obs_list.append({'condition': condition, 'target_gene': gene_name,
                                  'is_control': False, 'cell_line': cell_line})

        
        x_ctrl = ctrl_params_sel.get(cell_line)
        if x_ctrl is None:
            
            print(f"  WARNING: no control for cell_line='{cell_line}', using first available.")
            x_ctrl = next(iter(ctrl_params_sel.values()))
        x_ctrl = x_ctrl.unsqueeze(0).to(device)  

        s_ctrl = None
        if copula_rank > 0:
            sc = ctrl_s_sel.get(cell_line)
            if sc is None:
                sc = next((v for v in ctrl_s_sel.values() if v is not None), None)
            s_ctrl = sc.unsqueeze(0).to(device) if sc is not None else None

        label_t = torch.tensor([lbl], device=device)
        
        
        
        ctx_id = torch.tensor(
            [cell_line_vocab.get(cell_line, 0)], dtype=torch.long, device=device
        ) if cell_line_vocab else None

        with torch.no_grad():
            preds = model(x_ctrl, gene_indices, label_t, s_ctrl=s_ctrl,
                        cell_context_id=ctx_id)
            if 's' in preds and A_common is not None:
                samples = sample_from_copula(preds, A_common, args.n_samples, psi=psi_common,
                                             truncated=truncated)
            else:
                samples = sample_from_params(preds, args.n_samples, truncated=truncated)
            pred_X = samples.squeeze(0).cpu().numpy().T
        
        
        pred_X = np.minimum(pred_X, 14.9) if truncated else np.clip(pred_X, 0, 14.9)
        if args.value_floor > 0:
            pred_X[pred_X < args.value_floor] = args.value_floor

        pred_X_list.append(pred_X)
        for _ in range(args.n_samples):
            pred_obs_list.append({'condition': condition, 'target_gene': gene_name,
                                  'is_control': False, 'cell_line': cell_line})

    
    n_perturb_conditions = sum(1 for d in real_obs_list if not d.get('is_control', False))
    if n_perturb_conditions == 0:
        raise RuntimeError(
            "No perturbation cells were matched!  Possible causes:\n"
            f"  - control_value mismatch: using '{args.control_value}' but h5ad may use a "
            f"different label (check obs['{args.perturbation_col}'].unique())\n"
            f"  - perturbation_col mismatch: using '{args.perturbation_col}' but h5ad "
            f"obs columns are {list(adata.obs.columns)}\n"
            f"  - cell_line mismatch: using cell_line='{args.cell_line}' but h5ad has "
            f"{list(adata.obs[args.cell_line_col].unique()) if args.cell_line_col in adata.obs.columns else 'N/A'}\n"
            f"  - h5ad/processed data gene mismatch: genes in test split not found in h5ad\n"
            f"  - raw_data_path points to wrong file: '{args.raw_data_path}'\n"
            f"Run with the correct --dataset flag to auto-configure these values."
        )
    print("Building AnnData objects...")
    adata_real = anndata.AnnData(
        X=np.vstack(real_X_list),
        obs=pd.DataFrame(real_obs_list),
    )
    assert adata_real.shape[1] == len(gene_names), \
        f"Gene count mismatch: real data has {adata_real.shape[1]} columns, {len(gene_names)} gene names"
    adata_real.var_names = gene_names
    adata_real.obs_names = [f"cell_{i}" for i in range(adata_real.n_obs)]

    adata_pred = anndata.AnnData(
        X=np.vstack(pred_X_list),
        obs=pd.DataFrame(pred_obs_list),
    )
    assert adata_pred.shape[1] == len(gene_names), \
        f"Gene count mismatch: pred data has {adata_pred.shape[1]} columns, {len(gene_names)} gene names"
    adata_pred.var_names = gene_names
    adata_pred.obs_names = [f"cell_{i}" for i in range(adata_pred.n_obs)]

    real_path = Path(args.save_dir) / "inference_real.h5ad"
    pred_path = Path(args.save_dir) / "inference_pred.h5ad"

    adata_real.write_h5ad(real_path)
    adata_pred.write_h5ad(pred_path)
    print(f"Saved: {real_path}  ({adata_real.shape})")
    print(f"Saved: {pred_path}  ({adata_pred.shape})")
    print(f"\nRun cell-eval:\n  cell-eval run -ap {pred_path} -ar {real_path}")
    print("Done.")


if __name__ == "__main__":
    main()
