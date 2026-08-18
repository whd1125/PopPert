import torch
import numpy as np
import warnings
from scipy.special import betainc as scipy_betainc
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)


class ZINBFitter:
    

    def __init__(self, min_samples=10):
        self.min_samples = min_samples

    def fit(self, values):
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        values = np.asarray(values, dtype=np.float64)
        n_total = len(values)
        if n_total == 0:
            return self._default_params()
        n_zeros = int(np.sum(values == 0))
        pi_0 = n_zeros / n_total
        non_zeros = values[values > 0]
        if len(non_zeros) < self.min_samples:
            return self._default_params(pi_0=pi_0)
        try:
            mu_hat = float(np.mean(non_zeros))
            var_hat = float(np.var(non_zeros, ddof=1))
            theta_hat = mu_hat * mu_hat / max(var_hat - mu_hat, 1e-6)
            theta_hat = float(np.clip(theta_hat, 1e-3, 1e6))
            mu_hat = float(np.clip(mu_hat, 1e-6, None))
            return np.array([pi_0, mu_hat, theta_hat], dtype=np.float32)
        except Exception as e:
            warnings.warn(f"ZINB fit failed: {e}")
            return self._default_params(pi_0=pi_0)

    def _default_params(self, pi_0=1.0):
        return np.array([pi_0, 1.0, 1.0], dtype=np.float32)


def fit_batch_zinb(batch_expr, n_jobs=16):
    n_genes = batch_expr.shape[1]
    fitter = ZINBFitter()
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fitter.fit)(batch_expr[:, g]) for g in range(n_genes)
    )
    return np.array(results, dtype=np.float32)







def _params_to_numpy(params):
    if isinstance(params, torch.Tensor):
        params = params.cpu().detach().numpy()
    pi = np.asarray(params[..., 0], dtype=np.float64)
    mu = np.asarray(params[..., 1], dtype=np.float64)
    theta = np.asarray(params[..., 2], dtype=np.float64)
    return pi, mu, theta


def _nb_cdf_numpy(k, r, mu, eps=1e-8):
    
    r = np.clip(r, eps, None)
    mu = np.clip(mu, eps, None)
    p = r / (r + mu)   
    
    return scipy_betainc(r, k + 1, p)


def zinb_cdf(k, params, eps=1e-8):
    
    pi, mu, theta = _params_to_numpy(params)
    r = theta
    nb_cdf = _nb_cdf_numpy(k, r, mu, eps)
    return np.clip(pi + (1.0 - pi) * nb_cdf, 0.0, 1.0)


def zinb_quantile(u, params, max_count=5000, eps=1e-8):
    
    was_torch_u = isinstance(u, torch.Tensor)
    if was_torch_u:
        device = u.device
        u = u.cpu().detach().numpy()
    u = np.asarray(u, dtype=np.float64)
    pi, mu, theta = _params_to_numpy(params)   
    r = np.clip(theta, eps, None)
    p = r / (r + mu + eps)

    
    f0 = pi + (1.0 - pi) * (p ** r)

    n, G = u.shape
    result = np.zeros((n, G), dtype=np.float32)
    for g in range(G):
        ug = u[:, g]
        zero_mask = ug <= f0[g]
        res_g = np.zeros(n, dtype=np.float32)

        
        nonz = ~zero_mask
        if nonz.any():
            lo = np.zeros(nonz.sum(), dtype=np.int32)
            hi = np.full(nonz.sum(), max_count, dtype=np.int32)
            u_sub = ug[nonz]
            r_g = r[g]
            p_g = p[g]
            pi_g = pi[g]
            for _ in range(14):
                mid = (lo + hi) // 2
                nb = scipy_betainc(r_g, mid.astype(np.float64) + 1, p_g)
                f_mid = np.clip(pi_g + (1.0 - pi_g) * nb, 0.0, 1.0)
                ge = f_mid >= u_sub
                hi = np.where(ge, mid, hi)
                lo = np.where(ge, lo, mid + 1)
            res_g[nonz] = lo.astype(np.float32)
        result[:, g] = res_g

    result = result.astype(np.float32)
    if was_torch_u:
        return torch.as_tensor(result, dtype=torch.float32, device=device)
    return result


def expr_to_gaussian(X, params, eps=1e-6, seed=0):
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)
    if not isinstance(params, torch.Tensor):
        params = torch.as_tensor(params, dtype=torch.float32)

    X_np = X.cpu().numpy().astype(np.float64)
    pi, mu, theta = _params_to_numpy(params)  
    r = np.clip(theta, eps, None)
    p = r / (r + mu + eps)
    n, G = X_np.shape

    k = X_np.astype(np.int64)
    kp1 = k + 1

    
    nb_cdf_k = scipy_betainc(r[None, :], kp1, p[None, :])
    f_k = np.clip(pi[None, :] + (1.0 - pi[None, :]) * nb_cdf_k, eps, 1.0 - eps)

    
    km1 = k - 1
    nb_cdf_km1 = scipy_betainc(r[None, :], km1 + 1, p[None, :])
    f_km1 = np.where(km1 >= 0,
                     pi[None, :] + (1.0 - pi[None, :]) * nb_cdf_km1,
                     0.0)
    f_km1 = np.clip(f_km1, 0.0, 1.0 - eps)

    
    rng = np.random.RandomState(int(seed))
    v = rng.rand(n, G).astype(np.float64)
    u = np.clip(f_km1 + v * (f_k - f_km1).clip(min=eps), eps, 1.0 - eps)

    
    z = torch.special.ndtri(torch.as_tensor(u, dtype=torch.float32))
    z = torch.clamp(z, -5.0, 5.0)  
    return z







try:
    from gmm_utils import learn_factor_loading, factor_variance
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gmm_utils import learn_factor_loading, factor_variance
