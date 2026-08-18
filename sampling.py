import torch
import torch.nn.functional as F
import numpy as np

from gmm_utils import gmm_inverse_cdf


def sample_from_params(params_dict, n_samples=100, eps=1e-6, x_max=20.0, n_grid=2048,
                        return_mask=False, truncated=True):
    
    pi = params_dict['pi']  
    if pi.dim() == 2:
        pi = pi.unsqueeze(-1)

    w = params_dict['w']      
    mu = params_dict['mu']
    sigma = params_dict['sigma']

    B, N, K = w.shape
    device = w.device

    
    u = torch.rand((B, N, n_samples), device=device, dtype=w.dtype)
    is_zero = u < pi  

    
    
    params_bn = torch.cat([pi.expand(B, N, 1), w, mu, sigma], dim=-1)  
    params_flat = params_bn.reshape(B * N, -1)
    u_flat = u.permute(2, 0, 1).reshape(n_samples, B * N).contiguous()  
    x_flat = gmm_inverse_cdf(u_flat, params_flat, x_max=x_max, n_grid=n_grid, eps=eps,
                             truncated=truncated)
    x = x_flat.reshape(n_samples, B, N).permute(1, 2, 0).contiguous()   

    if return_mask:
        return x, is_zero
    return x


def unpack_params(tensor_params):
    
    D = tensor_params.shape[-1]
    K = (D - 1) // 3
    pi = tensor_params[..., 0]               
    w = tensor_params[..., 1:1+K]            
    mu = tensor_params[..., 1+K:1+2*K]       
    sigma = tensor_params[..., 1+2*K:1+3*K]  
    return {'pi': pi, 'w': w, 'mu': mu, 'sigma': sigma}


def zigmm_nll_loss(pred_params, target_params_tensor, n_samples=100, truncated=True):
    
    
    
    
    with torch.no_grad():
        target_dict = unpack_params(target_params_tensor)
        x_samples, is_zero = sample_from_params(target_dict, n_samples, return_mask=True)

    
    pi = pred_params['pi']  
    if pi.dim() == 2:
        pi = pi.unsqueeze(-1)  

    w = pred_params['w']  
    mu = pred_params['mu']  
    sigma = pred_params['sigma']  

    
    x = x_samples.unsqueeze(-1)

    
    w = w.unsqueeze(2)
    mu = mu.unsqueeze(2)
    sigma = sigma.unsqueeze(2)

    
    pi = pi.expand(x_samples.shape[0], x_samples.shape[1], x_samples.shape[2])

    
    
    

    
    eps = 1e-8
    nll_zero = -torch.log(pi + eps)

    
    var = sigma ** 2
    log_norm_const = -0.5 * torch.log(2 * torch.tensor(np.pi)) - torch.log(sigma + eps)
    log_exp_part = -0.5 * ((x - mu) ** 2) / (var + eps)
    log_gauss = log_norm_const + log_exp_part

    if truncated:
        
        
        log_Z = torch.special.log_ndtr(mu / sigma.clamp(min=eps))
        log_gauss = log_gauss - log_Z

    log_w = torch.log(w + eps)
    log_gmm_pdf = torch.logsumexp(log_w + log_gauss, dim=-1)  

    nll_nonzero = -(torch.log(1 - pi + eps) + log_gmm_pdf)

    
    loss_per_sample = torch.where(is_zero, nll_zero, nll_nonzero)

    return loss_per_sample.mean()


def wasserstein_loss(pred_params, real_params_tensor, n_samples=100):
    
    pred_samples = sample_from_params(pred_params, n_samples)

    
    real_dict = unpack_params(real_params_tensor)
    real_samples = sample_from_params(real_dict, n_samples)

    
    pred_sorted, _ = torch.sort(pred_samples, dim=-1)
    real_sorted, _ = torch.sort(real_samples, dim=-1)

    
    loss = torch.mean((pred_sorted - real_sorted) ** 2)
    return loss











def dependence_loss(s_pred, s_target, eps=1e-6):
    
    log_p = torch.log(s_pred.clamp(min=eps))
    log_t = torch.log(s_target.clamp(min=eps))
    return (log_p - log_t).pow(2).mean()


def sample_from_copula(preds, A, n_samples=100, eps=1e-6, x_max=20.0, n_grid=2048,
                       psi=None, truncated=True):
    
    pi = preds['pi']
    w = preds['w']; mu = preds['mu']; sigma = preds['sigma']
    s = preds['s']                                  
    if pi.dim() == 2:
        pi = pi.unsqueeze(-1)                       
    B, N, K = w.shape
    r = s.shape[-1]
    device = w.device

    
    h = torch.randn(B, n_samples, r, device=device, dtype=w.dtype) \
        * s.unsqueeze(1).clamp(min=eps).sqrt()
    
    A_dev = A.to(device=device, dtype=w.dtype)
    z = torch.einsum('bnr,gr->bng', h, A_dev)        
    
    if psi is not None:
        psi_dev = psi.to(device=device, dtype=w.dtype)
        xi = torch.randn(B, n_samples, N, device=device, dtype=w.dtype) \
            * psi_dev.unsqueeze(0).unsqueeze(0).sqrt().clamp(min=eps)  
        z = z + xi
    
    
    v = torch.einsum('gr,br->bg', A_dev.square(), s.clamp(min=eps))  
    if psi is not None:
        v = v + psi_dev.unsqueeze(0)                  
    z = z / v.unsqueeze(1).sqrt().clamp(min=eps)      
    
    u = torch.special.ndtr(z)                         

    
    
    params_bn = torch.cat([pi.expand(B, N, 1), w, mu, sigma], dim=-1)   
    params_flat = params_bn.reshape(B * N, -1)
    u_flat = u.permute(1, 0, 2).reshape(n_samples, B * N).contiguous()   
    x_flat = gmm_inverse_cdf(u_flat, params_flat, x_max=x_max, n_grid=n_grid, eps=eps,
                             truncated=truncated)  
    x = x_flat.reshape(n_samples, B, N).permute(1, 2, 0).contiguous()    
    return x


def compute_loss(preds, y_real, x_ctrl, n_samples=256, s_target=None, lam_dep=1.0,
                 truncated=True):
    
    nll = zigmm_nll_loss(preds, y_real, n_samples, truncated=truncated)
    log = {'nll': nll.item()}

    if 's' in preds and s_target is not None and lam_dep != 0:
        dep = dependence_loss(preds['s'], s_target)
        log['dep'] = dep.item()
        total = nll + lam_dep * dep
    else:
        total = nll
    log['total'] = total.item()
    return total, log


def validate(model, val_loader, gene_indices, device, n_samples_loss, lam_dep=1.0,
             truncated=True):
    model.eval()
    val_loss = 0
    val_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            x_ctrl = batch['x_ctrl'].to(device)
            y_real = batch['y_target'].to(device)
            s_ctrl = batch.get('s_ctrl')
            s_ctrl = s_ctrl.to(device) if s_ctrl is not None else None
            s_target = batch.get('s_target')
            s_target = s_target.to(device) if s_target is not None else None

            if getattr(model, 'perturb_mode', 'gene') == 'drug':
                drug_fp = batch['drug_fp'].to(device)
                dose = batch['dose'].to(device)
                cell_type = batch['cell_type'].to(device)
                preds = model(x_ctrl, gene_indices, drug_fp=drug_fp, dose=dose,
                              cell_type=cell_type, s_ctrl=s_ctrl)
            else:
                label = batch['label'].to(device)
                cell_context_id = batch.get('cell_context_id')
                cell_context_id = cell_context_id.to(device) if cell_context_id is not None else None
                preds = model(x_ctrl, gene_indices, perturb_indices=label, s_ctrl=s_ctrl,
                              cell_context_id=cell_context_id)

            loss, _ = compute_loss(preds, y_real, x_ctrl, n_samples=n_samples_loss,
                                   s_target=s_target, lam_dep=lam_dep,
                                   truncated=truncated)
            val_loss += loss.item()
            val_steps += 1

    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
    return avg_val_loss
