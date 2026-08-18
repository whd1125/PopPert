import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

class ZIGMMFitter:
    def __init__(self, n_components=2, min_samples=10):
        self.n_components = n_components
        self.min_samples = min_samples

    def fit(self, values):
        
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
            
        n_total = len(values)
        if n_total == 0:
            return self._default_params()

        
        n_zeros = np.sum(values == 0)
        pi_0 = n_zeros / n_total
        
        
        non_zeros = values[values > 0]
        
        
        if len(non_zeros) < self.min_samples:
            return self._default_params(pi_0=pi_0)
            
        
        try:
            
            X = non_zeros.reshape(-1, 1)
            
            gmm = GaussianMixture(n_components=self.n_components, covariance_type='diag', reg_covar=1e-5, random_state=42)
            gmm.fit(X)
            
            means = gmm.means_.flatten() 
            covariances = gmm.covariances_.flatten() 
            stds = np.sqrt(np.maximum(covariances, 1e-6))  
            stds = np.maximum(stds, 0.1)                  
            
            
            weights = gmm.weights_.flatten() 
            
            
            idx = np.argsort(means)
            means = means[idx]
            stds = stds[idx]
            weights = weights[idx]
            
            
            
            return np.concatenate([[pi_0], weights, means, stds])
            
        except Exception as e:
            
            warnings.warn(f"GMM fit failed for gene with {len(non_zeros)} non-zero values: {e}")
            return self._default_params(pi_0=pi_0)

    def _default_params(self, pi_0=1.0):
        
        
        w = np.ones(self.n_components) / self.n_components
        mu = np.zeros(self.n_components)
        sigma = np.ones(self.n_components)
        return np.concatenate([[pi_0], w, mu, sigma])

from joblib import Parallel, delayed

def fit_single_gene(fitter, gene_vals):
    return fitter.fit(gene_vals)

def fit_batch_gmm(batch_expr, n_components=2, n_jobs=16):
    
    n_genes = batch_expr.shape[1]
    fitter = ZIGMMFitter(n_components=n_components)
    
    
    
    
    
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_single_gene)(fitter, batch_expr[:, g])
        for g in range(n_genes)
    )

    result = np.array(results, dtype=np.float32)
    
    
    Kc = n_components
    result[:, 1 + 2*Kc : 1 + 3*Kc] = np.maximum(result[:, 1 + 2*Kc : 1 + 3*Kc], 0.1)
    return result






















def _unpack_gene_params(params, n_components=None):
    
    if n_components is None:
        n_components = (params.shape[-1] - 1) // 3
    K = n_components
    return {
        'pi': params[..., 0],
        'w': params[..., 1:1 + K],
        'mu': params[..., 1 + K:1 + 2 * K],
        'sigma': params[..., 1 + 2 * K:1 + 3 * K],
        'K': K,
    }


def zigmm_cdf(x, params, n_components=None, truncated=True):
    
    p = _unpack_gene_params(params, n_components)
    pi = p['pi']                       
    w, mu, sigma = p['w'], p['mu'], p['sigma']   
    z_score = (x.unsqueeze(-1) - mu) / sigma.clamp(min=1e-6)   
    comp_cdf = torch.special.ndtr(z_score)                       

    if truncated:
        
        
        alpha = -mu / sigma.clamp(min=1e-6)                      
        phi_alpha = torch.special.ndtr(alpha)                    
        
        Z = torch.special.ndtr(mu / sigma.clamp(min=1e-6))      
        comp_cdf = (comp_cdf - phi_alpha) / Z.clamp(min=1e-6)

    gmm_cdf = (w * comp_cdf).sum(-1)                             
    return pi + (1.0 - pi) * gmm_cdf


def expr_to_gaussian(X, params, n_components=None, eps=1e-6, seed=0,
                     truncated=True):
    
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)
    if not isinstance(params, torch.Tensor):
        params = torch.as_tensor(params, dtype=torch.float32)
    p = _unpack_gene_params(params, n_components)
    pi = p['pi']                       
    w, mu, sigma = p['w'], p['mu'], p['sigma']

    
    x_exp = X.unsqueeze(-1)                    
    mu_exp = mu.unsqueeze(0)                   
    sig_exp = sigma.unsqueeze(0).clamp(min=1e-6)
    comp_cdf = torch.special.ndtr((x_exp - mu_exp) / sig_exp)        

    if truncated:
        
        alpha = -mu_exp / sig_exp                                    
        phi_alpha = torch.special.ndtr(alpha)
        Z = torch.special.ndtr(mu_exp / sig_exp).clamp(min=1e-6)    
        comp_cdf = (comp_cdf - phi_alpha) / Z

    gmm_cdf = (w.unsqueeze(0) * comp_cdf).sum(-1)                    

    pi_exp = pi.unsqueeze(0)                   
    u_pos = pi_exp + (1.0 - pi_exp) * gmm_cdf  

    
    
    zero_mask = (X == 0)
    gen = torch.Generator(device=X.device).manual_seed(int(seed))
    u_rand = torch.rand(X.shape, generator=gen, device=X.device, dtype=X.dtype)
    u_zero = u_rand * pi_exp                   

    u = torch.where(zero_mask, u_zero, u_pos).clamp(eps, 1.0 - eps)
    return torch.special.ndtri(u)              


def gmm_inverse_cdf(u, params, n_components=None, x_max=20.0, n_grid=2048, eps=1e-6,
                    truncated=True):
    
    if not isinstance(u, torch.Tensor):
        u = torch.as_tensor(u, dtype=torch.float32)
    if not isinstance(params, torch.Tensor):
        params = torch.as_tensor(params, dtype=torch.float32)
    p = _unpack_gene_params(params, n_components)
    pi = p['pi']                       
    w, mu, sigma = p['w'], p['mu'], p['sigma']

    device = u.device
    grid = torch.linspace(0.0, x_max, n_grid, device=device, dtype=u.dtype)  

    
    g_exp = grid.unsqueeze(0).unsqueeze(-1)        
    mu_g = mu.unsqueeze(1)                         
    sig_g = sigma.unsqueeze(1).clamp(min=1e-6)
    w_g = w.unsqueeze(1)                           
    comp_cdf = torch.special.ndtr((g_exp - mu_g) / sig_g)           

    if truncated:
        
        alpha = -mu_g / sig_g                                       
        phi_alpha = torch.special.ndtr(alpha)
        Z = torch.special.ndtr(mu_g / sig_g).clamp(min=1e-6)       
        comp_cdf = (comp_cdf - phi_alpha) / Z
        

    cdf_grid = (w_g * comp_cdf).sum(-1)                              
    cdf_grid = cdf_grid.clamp(eps, 1.0 - eps)

    
    pi_col = pi.unsqueeze(1)                        
    u_t = u.t().contiguous()                        
    u_prime = ((u_t - pi_col) / (1.0 - pi_col).clamp(min=eps)).clamp(eps, 1.0 - eps)

    idx = torch.searchsorted(cdf_grid, u_prime, right=False).clamp(0, n_grid - 2)  
    cdf_lo = torch.gather(cdf_grid, 1, idx)
    cdf_hi = torch.gather(cdf_grid, 1, idx + 1)
    x_lo = grid[idx]
    x_hi = grid[idx + 1]
    
    
    
    
    frac = (u_prime - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=1e-12)
    x_pos = x_lo + frac * (x_hi - x_lo)
    x_pos = torch.clamp(x_pos, 0.0, x_max)          
    
    
    if truncated:
        x = x_pos.t().contiguous()                  
    else:
        x = x_pos.t().contiguous().clamp(min=0.0)   

    zero_mask = (u <= pi.unsqueeze(0))             
    return torch.where(zero_mask, torch.zeros_like(x), x)


def learn_factor_loading(Z, r, eps=1e-6):
    
    if not isinstance(Z, torch.Tensor):
        Z = torch.as_tensor(Z, dtype=torch.float32)
    ctrl_mean = Z.mean(dim=0)
    ctrl_std = Z.std(dim=0).clamp(min=eps)
    Zn = (Z - ctrl_mean) / ctrl_std
    n = Zn.shape[0]
    R = (Zn.t() @ Zn) / max(n - 1, 1)             
    R = 0.5 * (R + R.t())                          
    _, eigvecs = torch.linalg.eigh(R)              
    A = eigvecs[:, -r:].contiguous()               
    
    h = Zn @ A                                     
    z_recon = h @ A.t()                            
    residual = Zn - z_recon                        
    psi = residual.var(dim=0, unbiased=False).clamp(min=eps)  

    
    d = (A.t().square() @ psi.unsqueeze(-1)).squeeze(-1).clamp(min=eps)  

    return (A.contiguous(), ctrl_mean.contiguous(), ctrl_std.contiguous(),
            psi.contiguous(), d.contiguous())


def factor_variance(Z, A, ctrl_mean=None, ctrl_std=None, eps=1e-6, d=None):
    
    if not isinstance(Z, torch.Tensor):
        Z = torch.as_tensor(Z, dtype=torch.float32)
    if ctrl_mean is not None:
        Zn = (Z - ctrl_mean) / ctrl_std.clamp(min=eps)
    else:
        Zn = Z
    h = Zn @ A                                     
    raw = h.var(dim=0, unbiased=False)
    if d is not None:
        d_dev = d.to(device=raw.device, dtype=raw.dtype)
        return (raw - d_dev).clamp(min=eps)
    return raw.clamp(min=eps)
