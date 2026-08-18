

import torch
import numpy as np

_POISSON_THRESH = 10000




def _nb_log_prob(x, mu, theta, eps=1e-8):
    m = mu.expand_as(x)
    r = theta.expand_as(x)
    p_nb = torch.clamp(r / (r + m + eps), eps, 1 - eps)
    lp_nb = (
        torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1)
        + r * torch.log(p_nb) + x * torch.log(1 - p_nb + eps)
    )
    lp_pois = -m + x * torch.log(torch.clamp(m, min=eps)) - torch.lgamma(x + 1)
    use_pois = (theta > _POISSON_THRESH).expand_as(x)
    return torch.nan_to_num(torch.where(use_pois, lp_pois, lp_nb),
                            nan=-1e10, posinf=-1e10, neginf=-1e10)




def sample_from_params(params_dict, n_samples=100):
    pi = params_dict['pi']
    if pi.dim() == 2:
        pi = pi.unsqueeze(-1)
    mu = params_dict['mu']
    theta = params_dict['theta']
    B, N = mu.shape
    device = mu.device
    M = n_samples

    r = theta
    probs = torch.clamp(mu / (r + mu + 1e-8), 1e-6, 1 - 1e-6)
    nb_samples = torch.empty((B, N, M), device=device)
    large = theta > _POISSON_THRESH

    if large.any():
        rates = mu[large].unsqueeze(-1).expand(-1, M)
        pois = torch.poisson(torch.clamp(rates, min=0, max=1e6))
        nb_samples[large] = torch.clamp(pois, 0, 1e6)

    small = ~large
    if small.any():
        r_s = r[small].unsqueeze(-1).expand(-1, M)
        p_s = probs[small].unsqueeze(-1).expand(-1, M)
        raw = torch.distributions.NegativeBinomial(total_count=r_s, probs=p_s, validate_args=False).sample()
        nb_samples[small] = torch.clamp(raw, 0, 1e6)

    is_zero = torch.rand((B, N, M), device=device) < pi.expand(B, N, M)
    return torch.where(is_zero, torch.zeros((), device=device), nb_samples)


def unpack_params(tensor_params):
    return {'pi': tensor_params[..., 0], 'mu': tensor_params[..., 1], 'theta': tensor_params[..., 2]}




def zinb_nll_loss(pred_params, target_params_tensor, n_samples=256):
    with torch.no_grad():
        target = unpack_params(target_params_tensor)
        x = sample_from_params(target, n_samples)

    pi = pred_params['pi']
    if pi.dim() == 2:
        pi = pi.unsqueeze(-1)
    pi = pi.expand_as(x)
    mu = pred_params['mu'].unsqueeze(-1).expand_as(x)
    theta = pred_params['theta'].unsqueeze(-1).expand_as(x)

    eps = 1e-8
    is_zero = (x == 0)

    nb_log_p0 = -theta * torch.log1p(mu / (theta + eps))
    log_p_zero = torch.log(pi + (1 - pi + eps) * torch.exp(nb_log_p0) + eps)
    log_p_nonzero = torch.log(1 - pi + eps) + _nb_log_prob(x, mu, theta, eps)

    ll = torch.where(is_zero, log_p_zero, log_p_nonzero)
    loss = -ll.mean()
    if torch.isnan(loss) or torch.isinf(loss):
        loss = -torch.clamp(ll, min=-100, max=100).mean()
    return loss




def dependence_loss(s_pred, s_target, eps=1e-6):
    log_p = torch.log(s_pred.clamp(min=eps))
    log_t = torch.log(s_target.clamp(min=eps))
    return (log_p - log_t).pow(2).mean()


def sample_from_copula(preds, A, n_samples=100, eps=1e-6, max_count=5000,
                       psi=None):
    
    pi = preds['pi']; mu = preds['mu']; theta = preds['theta']
    s = preds['s']
    if pi.dim() == 2:
        pi = pi.unsqueeze(-1)
    B, N = mu.shape
    r = s.shape[-1]
    device = mu.device

    
    h = torch.randn(B, n_samples, r, device=device, dtype=mu.dtype) \
        * s.unsqueeze(1).clamp(min=eps).sqrt()

    
    A_dev = A.to(device=device, dtype=mu.dtype)
    z_latent = torch.einsum('bnr,gr->bng', h, A_dev)

    if psi is not None:
        psi_dev = psi.to(device=device, dtype=mu.dtype)
        xi = torch.randn(B, n_samples, N, device=device, dtype=mu.dtype) \
            * psi_dev.unsqueeze(0).unsqueeze(0).sqrt().clamp(min=eps)
        z_latent = z_latent + xi

    
    v = torch.einsum('gr,br->bg', A_dev.square(), s.clamp(min=eps))
    if psi is not None:
        v = v + psi_dev.unsqueeze(0)
    z_latent = z_latent / v.unsqueeze(1).sqrt().clamp(min=eps)

    
    z_latent = torch.clamp(z_latent, -5.0, 5.0)
    u = torch.special.ndtr(z_latent)

    
    
    u_flat = u.permute(1, 0, 2).reshape(n_samples, B * N).contiguous()

    
    pi_flat = pi.expand(B, N, 1).reshape(B * N, 1)
    mu_flat = mu.reshape(B * N, 1)
    theta_flat = theta.reshape(B * N, 1)
    params_flat = torch.cat([pi_flat, mu_flat, theta_flat], dim=-1)

    
    
    from zinb_utils import zinb_quantile

    x_chunks = []
    chunk_size = 128
    for i in range(0, B * N, chunk_size):
        end = min(i + chunk_size, B * N)
        p_chunk = params_flat[i:end]                     
        u_chunk = u_flat[:, i:end]                        
        
        x_chunk = zinb_quantile(u_chunk, p_chunk, max_count=max_count)
        x_chunks.append(x_chunk)

    x_flat = torch.cat(x_chunks, dim=1)                   
    x = x_flat.reshape(n_samples, B, N).permute(1, 2, 0).contiguous()
    return x


def compute_loss(preds, y_real, n_samples=256, s_target=None, lam_dep=1.0):
    nll = zinb_nll_loss(preds, y_real, n_samples)
    log = {'nll': nll.item()}
    if 's' in preds and s_target is not None and lam_dep != 0:
        dep = dependence_loss(preds['s'], s_target)
        log['dep'] = dep.item()
        total = nll + lam_dep * dep
    else:
        total = nll
    log['total'] = total.item()
    return total, log


def validate(model, val_loader, gene_indices, device, n_samples_loss, lam_dep=1.0):
    model.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            x_ctrl = batch['x_ctrl'].to(device)
            y_real = batch['y_target'].to(device)
            label = batch['label'].to(device)
            s_ctrl = batch.get('s_ctrl')
            s_ctrl = s_ctrl.to(device) if s_ctrl is not None else None
            s_target = batch.get('s_target')
            s_target = s_target.to(device) if s_target is not None else None
            cell_context_id = batch.get('cell_context_id')
            cell_context_id = cell_context_id.to(device) if cell_context_id is not None else None

            preds = model(x_ctrl, gene_indices, label, s_ctrl=s_ctrl, cell_context_id=cell_context_id)
            loss, _ = compute_loss(preds, y_real, n_samples=n_samples_loss,
                                   s_target=s_target, lam_dep=lam_dep)
            val_loss += loss.item()
            val_steps += 1
    return val_loss / val_steps if val_steps > 0 else 0.0
