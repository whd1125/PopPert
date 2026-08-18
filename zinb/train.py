import argparse, json, random, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import ZINBDataset
from model import ZINBTransformer
from sampling import compute_loss, validate

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="norman")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default=str(ROOT / "vocab.json"))
    parser.add_argument("--pretrin_embedding_path", type=str,
                        default=str(ROOT / "embedding_weight" / "best_model.pt"))
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                        help="Stop if no val improvement for N consecutive epochs. 0 disables.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_samples_loss", type=int, default=256)
    parser.add_argument("--val_interval", type=int, default=400)
    parser.add_argument("--copula_lambda", type=float, default=0.2)
    parser.add_argument("--no_copula", action="store_true")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    if args.data_path is None:  
        args.data_path = str(ROOT / "data" / "processed" / f"processed_{args.dataset}_zinb.pt")
    if args.save_dir is None:
        args.save_dir = str(ROOT / "results" / args.dataset / "ours_zinb" / "checkpoints")
    return args


def main():
    args = parse_args()
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    dataset = ZINBDataset(args.data_path, split='train')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataset = ZINBDataset(args.data_path, split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    num_genes = len(dataset.hvg_indices) if hasattr(dataset, 'hvg_indices') else 5000
    with open(args.vocab_path, 'r') as f:
        vocab_size = len(json.load(f))
    print(f"num_genes: {num_genes}, vocab_size: {vocab_size}")

    copula_rank = 0 if args.no_copula else int(getattr(dataset, 'copula_rank', 0) or 0)
    copula_lambda = args.copula_lambda
    print(f"Copula: rank={copula_rank}, lambda={copula_lambda}")

    dim_ff = args.dim_feedforward if args.dim_feedforward else 4 * args.d_model
    model = ZINBTransformer(
        vocab_size=vocab_size, d_model=args.d_model,
        num_layers=args.nlayers, nhead=args.nhead,
        dim_feedforward=dim_ff, dropout=args.dropout,
        copula_rank=copula_rank,
        n_cell_types=getattr(dataset, 'n_cell_types', 0) or 0,
    ).to(device)

    
    try:
        sd = torch.load(args.pretrin_embedding_path, map_location=device, weights_only=False)
        emb = sd.get('module.embedding.weight')
        if emb is not None and emb.shape == model.shared_embedding.weight.shape:
            model.shared_embedding.weight.data.copy_(emb)
            model.shared_embedding.weight.requires_grad = True
            print("Pretrained embeddings loaded.")
    except Exception as e:
        print(f"Pretrained embedding skip: {e}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    writer = SummaryWriter(log_dir=Path(args.save_dir) / "tensorboard")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    gene_indices = torch.tensor(dataset.hvg_indices, dtype=torch.long, device=device) \
        if hasattr(dataset, 'hvg_indices') else torch.arange(num_genes, device=device)

    print(f"Training — L = L_nll + λ_dep·L_dep  (λ_dep={copula_lambda})")
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, steps = 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            x_ctrl = batch['x_ctrl'].to(device)
            y_real = batch['y_target'].to(device)
            label = batch['label'].to(device)
            s_ctrl = batch.get('s_ctrl'); s_ctrl = s_ctrl.to(device) if s_ctrl is not None else None
            s_target = batch.get('s_target'); s_target = s_target.to(device) if s_target is not None else None
            ctx = batch.get('cell_context_id'); ctx = ctx.to(device) if ctx is not None else None

            optimizer.zero_grad()
            preds = model(x_ctrl, gene_indices, label, s_ctrl=s_ctrl, cell_context_id=ctx)
            loss, log = compute_loss(preds, y_real, n_samples=args.n_samples_loss,
                                     s_target=s_target, lam_dep=copula_lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item(); steps += 1; global_step += 1
            pf = {'L': f"{loss.item():.3f}", 'nll': f"{log['nll']:.3f}"}
            if 'dep' in log: pf['dep'] = f"{log['dep']:.3f}"
            pbar.set_postfix(pf)

            if global_step % args.val_interval == 0:
                val_l = validate(model, val_loader, gene_indices, device, args.n_samples_loss,
                                 lam_dep=copula_lambda)
                if val_l < best_val_loss:
                    best_val_loss = val_l
                    torch.save(model.state_dict(), Path(args.save_dir) / "best_model.pt")
                    print(f"\nStep {global_step}: Saved best model (val={best_val_loss:.4f})")
                model.train()

        avg = total_loss / steps
        val_l = validate(model, val_loader, gene_indices, device, args.n_samples_loss,
                         lam_dep=copula_lambda)
        print(f"Epoch {epoch+1}: train={avg:.4f}  val={val_l:.4f}")
        writer.add_scalar("Loss/Train", avg, epoch)
        writer.add_scalar("Loss/Val", val_l, epoch)
        scheduler.step(val_l)
        if val_l < best_val_loss:
            best_val_loss = val_l
            patience_counter = 0
            torch.save(model.state_dict(), Path(args.save_dir) / "best_model.pt")
            print(f"  -> Saved best model (val={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{args.early_stopping_patience} epochs")
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), Path(args.save_dir) / "model_final.pt")


if __name__ == "__main__":
    main()
