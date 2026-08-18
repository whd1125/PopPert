import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import sys
import json

from dataset import DistDataset
from model import DistributionTransformer
from sampling import compute_loss, validate

def parse_args():
    ROOT = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="replogle", help="Dataset name")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default=str(ROOT / "vocab.json"))
    parser.add_argument("--pretrin_embedding_path", type=str, default=str(ROOT / "embedding_weight" / "best_model.pt"))

    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=None,
                        help="Transformer FF hidden dim. Default: auto = 4*d_model.")
    parser.add_argument("--n_components", type=int, default=2, help="Number of GMM components (K). param_dim = 1 + 3*K")
    parser.add_argument("--perturb_mode", type=str, default="gene", choices=["gene", "drug"],
                        help="Perturbation mode. 'gene'=CRISPR (default). 'drug'=chemical (SCI-plex). "
                             "Overridden by the mode recorded in the processed .pt file.")
    parser.add_argument("--fp_dim", type=int, default=1024, help="Morgan fingerprint dimension (drug mode)")
    
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                        help="Stop if no val improvement for N consecutive epochs. 0 disables.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--n_samples_loss", type=int, default=256, help="Number of points to sample for the NLL loss")
    parser.add_argument("--val_interval", type=int, default=400, help="Validate every N steps within an epoch")

    
    
    
    parser.add_argument("--copula_lambda", type=float, default=1.0,
                        help="Weight of the copula factor-variance dependence loss (lam_dep)")
    parser.add_argument("--no_copula", action="store_true",
                        help="Disable the copula dependency head/loss even if the data supports it")

    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda:0, cuda:1, cpu). Auto-detect if not set.")
    args = parser.parse_args()
    if args.data_path is None:
        args.data_path = str(ROOT / "data" / "processed" / f"processed_{args.dataset}.pt")
    if args.save_dir is None:
        args.save_dir = str(ROOT / "results" / args.dataset / "ours" / "checkpoints")
    return args


def main():
    args = parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    
    dataset = DistDataset(args.data_path, split='train')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = DistDataset(args.data_path, split='val')
    truncated = getattr(dataset, 'truncated', False)
    if truncated:
        print("Truncated ZI-GMM mode: ON")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    
    perturb_mode = getattr(dataset, 'perturb_mode', 'gene')
    if perturb_mode != args.perturb_mode:
        print(f"Note: --perturb_mode={args.perturb_mode} overridden by dataset mode='{perturb_mode}'.")
    print(f"Perturbation mode: {perturb_mode}")
    
    
    if hasattr(dataset, 'hvg_indices'):
        num_genes = len(dataset.hvg_indices)
        print(f"Detected {num_genes} genes from dataset.")
    else:
        
        if len(dataset.perturb_samples) > 0:
            num_genes = dataset.perturb_samples[0][0].shape[0]
            print(f"Inferred {num_genes} genes from data shape.")
        else:
            num_genes = 5000
            print("WARNING: Could not infer num_genes, defaulting to 5000.")
            
    
    vocab_path = args.vocab_path
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
        vocab_size = len(vocab)
        
    print(f"num_genes : {num_genes}")
    print(f"vocab_size : {vocab_size}")

    
    dim_feedforward = args.dim_feedforward if args.dim_feedforward else 4 * args.d_model
    print(f"dim_feedforward: {dim_feedforward}")

    
    
    copula_rank = 0 if args.no_copula else int(getattr(dataset, 'copula_rank', 0) or 0)
    copula_lambda = args.copula_lambda
    print(f"Copula: rank={copula_rank} lambda={copula_lambda}"
          + (" (disabled by --no_copula)" if (args.no_copula and copula_rank == 0 and getattr(dataset, 'copula_rank', 0)) else ""))

    model = DistributionTransformer(
        vocab_size=vocab_size,
        n_components=args.n_components,
        d_model=args.d_model,
        num_layers=args.nlayers,
        nhead=args.nhead,
        dim_feedforward=dim_feedforward,
        dropout=args.dropout,
        perturb_mode=perturb_mode,
        fp_dim=getattr(dataset, 'fp_dim', args.fp_dim) or args.fp_dim,
        n_cell_types=getattr(dataset, 'n_cell_types', 0) or 0,
        copula_rank=copula_rank,
    ).to(device)
    
    
    pretrained_path = args.pretrin_embedding_path
    print(f"Loading pretrained embeddings from {pretrained_path}...")
    try:
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
        
        if 'module.embedding.weight' in state_dict:
            pretrained_emb = state_dict['module.embedding.weight']
            
            if pretrained_emb.shape == model.shared_embedding.weight.shape:
                model.shared_embedding.weight.data.copy_(pretrained_emb)
                print("Pretrained embeddings loaded successfully!")
                
                model.shared_embedding.weight.requires_grad = True
            else:
                print(f"Shape mismatch: Pretrained {pretrained_emb.shape} vs Model {model.shared_embedding.weight.shape}")
        else:
            print("WARNING: Could not find 'module.embedding.weight' in checkpoint.")
    except Exception as e:
        print(f"Error loading pretrained embeddings: {e}")

    
    
    
    
    
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    writer = SummaryWriter(log_dir=Path(args.save_dir) / "tensorboard")
    
    print("Starting training...")
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    
    if hasattr(dataset, 'hvg_indices'):
        gene_indices = torch.tensor(dataset.hvg_indices, dtype=torch.long, device=device)
    else:
        print("WARNING: using arange for gene indices (might be wrong if shared embedding expects vocab IDs)")
        gene_indices = torch.arange(num_genes, device=device)

    
    print(f"Loss: NLL + lam_dep*L_dep  (lam_dep={copula_lambda}, copula_rank={copula_rank})")

    best_val_loss = float('inf')
    patience_counter = 0

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        steps = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            x_ctrl = batch['x_ctrl'].to(device)
            y_real = batch['y_target'].to(device)
            s_ctrl = batch.get('s_ctrl')
            s_ctrl = s_ctrl.to(device) if s_ctrl is not None else None
            s_target = batch.get('s_target')
            s_target = s_target.to(device) if s_target is not None else None
            cell_context_id = batch.get('cell_context_id')
            cell_context_id = cell_context_id.to(device) if cell_context_id is not None else None

            optimizer.zero_grad()

            if perturb_mode == 'drug':
                drug_fp = batch['drug_fp'].to(device)
                dose = batch['dose'].to(device)
                cell_type = batch['cell_type'].to(device)
                preds = model(x_ctrl, gene_indices, drug_fp=drug_fp, dose=dose,
                              cell_type=cell_type, s_ctrl=s_ctrl)
            else:
                label = batch['label'].to(device)
                preds = model(x_ctrl, gene_indices, perturb_indices=label, s_ctrl=s_ctrl,
                              cell_context_id=cell_context_id)

            loss, log = compute_loss(preds, y_real, x_ctrl, n_samples=args.n_samples_loss,
                                     s_target=s_target, lam_dep=copula_lambda,
                                     truncated=truncated)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            steps += 1
            global_step += 1

            pf = {'L': f"{loss.item():.3f}", 'nll': f"{log['nll']:.3f}"}
            if 'dep' in log:
                pf['dep'] = f"{log['dep']:.3f}"
            pbar.set_postfix(pf)
            writer.add_scalar("LossComp/nll", log['nll'], global_step)
            if 'dep' in log:
                writer.add_scalar("LossComp/dep", log['dep'], global_step)

            
            if global_step % args.val_interval == 0:
                avg_val_loss = validate(model, val_loader, gene_indices, device, args.n_samples_loss,
                                        lam_dep=copula_lambda, truncated=truncated)
                print(f"\nStep {global_step} Val Loss: {avg_val_loss:.4f}")
                writer.add_scalar("Loss/Val_Step", avg_val_loss, global_step)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), Path(args.save_dir) / "best_model.pt")
                    print(f"Saved best model (Val Loss: {best_val_loss:.4f})")
                
                model.train()
            
        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        
        avg_val_loss = validate(model, val_loader, gene_indices, device, args.n_samples_loss,
                                lam_dep=copula_lambda, truncated=truncated)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Path(args.save_dir) / "best_model.pt")
            print(f"Saved best model (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{args.early_stopping_patience} epochs")
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    
    
    import json as _json
    config_path = Path(args.save_dir) / "model_config.json"
    print(f"Saved model config to {config_path}")

    
    torch.save(model.state_dict(), Path(args.save_dir) / "model_final.pt")

if __name__ == "__main__":
    main()
