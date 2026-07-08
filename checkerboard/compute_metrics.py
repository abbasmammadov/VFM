import os
import math
import wandb
import itertools
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.func import jvp
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import gaussian_kde

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

def sample_checkerboard(n_samples, n_squares=4):
    assert n_squares % 2 == 0

    # sample candidate points
    x = np.random.rand(n_samples, 2)

    # map to square indices
    idx = (x * n_squares).astype(int)

    # checkerboard mask
    mask = (idx[:, 0] + idx[:, 1]) % 2 == 0

    # resample rejected points
    while not mask.all():
        n_bad = (~mask).sum()
        x_new = np.random.rand(n_bad, 2)
        idx_new = (x_new * n_squares).astype(int)

        x[~mask] = x_new
        idx[~mask] = idx_new
        mask = (idx[:, 0] + idx[:, 1]) % 2 == 0

    return 4.0 * (x - np.array([[0.5, 0.5]]))

class MeanFlowMLP(nn.Module):
    """
    u_theta(z, r, t) -> R^2
    Inputs: z in R^2, (t, r) scalars (two extra channels)
    """
    def __init__(self, hidden=256, depth=4):
        super().__init__()
        layers = []
        in_dim = 2 + 2  # z + (t, t-r)
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.SiLU()]
            dim = hidden
        layers += [nn.Linear(dim, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, r, t):
        # dt = t - r
        inp = torch.cat([z, t, r], dim=1)
        return self.net(inp)

class Adapter(nn.Module):
    def __init__(self, hidden=128, depth=3, z_dim=2):
        super().__init__()
        layers = []
        in_dim = 1  # y is scalar
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.SiLU()]
            dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(dim, z_dim)
        self.log_std = nn.Linear(dim, z_dim)  # log sigma

    def forward(self, y):
        h = self.backbone(y)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-6.0, 3.0)  # stability
        return mu, log_std

def f_theta(z, u_model, K=1):
    # z has shape (B,D) or (B,J,D)
    if z.ndim == 3:
        B, J, D = z.shape
        z = z.reshape(B * J, D)
    ts = torch.linspace(0.0, 1.0, K + 1, device=device)
    for i in range(K, 0, -1):
        t_i = torch.full((z.shape[0], 1), ts[i].item(), device=device)
        r_i = torch.full((z.shape[0], 1), ts[i-1].item(), device=device)
        z = z - (t_i - r_i) * u_model(z, r_i, t_i)
    if 'J' in locals():
        z = z.reshape(B, J, D)
    return z

def checkerboard_block_id(x01: torch.Tensor, n_squares=4, filled_parity=0):
    """
    x01: (N,2) points assumed in [0,1]^2
    returns:
      block_id: (N,) long tensor, -1 if outside [0,1]^2 or not a filled square
      ij: (N,2) long tensor of grid indices (i,j) (undefined when block_id=-1)
    """
    assert x01.ndim == 2 and x01.shape[1] == 2
    x = x01

    # inside unit box?
    inside = (x[:,0] >= 0) & (x[:,0] < 1) & (x[:,1] >= 0) & (x[:,1] < 1)

    ij = torch.floor(x * n_squares).long().clamp(0, n_squares-1)
    i, j = ij[:,0], ij[:,1]

    # checkerboard: keep only squares with (i+j) % 2 == filled_parity
    filled = ((i + j) % 2 == filled_parity)

    ok = inside & filled
    block_id = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
    block_id[ok] = i[ok] * n_squares + j[ok]  # unique id per grid cell

    return block_id, ij

def unscale_to_unit(x_scaled):
    return x_scaled / 4.0 + 0.5

def reparam_sample(mu, log_std, J=None):
    # mu, log_std: Shape (B, D)
    if J is None:
        eps = torch.randn_like(mu)
        z = mu + torch.exp(log_std) * eps
    else:
        # Get samples of shape (B, J, D)
        B, D = mu.shape
        eps = torch.randn(B, J, D, device=mu.device, dtype=mu.dtype)
        z = mu.unsqueeze(1) + torch.exp(log_std).unsqueeze(1) * eps
    return z
    
def sample_y(x, sigma_obs):
    # A = [1,0], so observe first coord
    y = x[:, :1] + sigma_obs * torch.randn(x.shape[0], 1, device=x.device, dtype=x.dtype)
    return y

@torch.no_grad()
def meanflow_one_step_with_latents(model, n_samples=10_000, device="cpu"):
    model.eval()
    z1 = torch.randn(n_samples, 2, device=device)
    t = torch.ones(n_samples, 1, device=device)
    r = torch.zeros(n_samples, 1, device=device)
    x0 = z1 - model(z1, r, t)
    return z1, x0

@torch.no_grad()
def meanflow_k_steps_with_latents(model, n_samples=10_000, K=10, device="cpu"):
    import math
    model.eval()
    z = torch.randn(n_samples, 2, device=device)
    z1 = z.clone()
    ts = torch.linspace(0.0, 1.0, K + 1, device=device)

    for i in range(K, 0, -1):
        t_i = torch.full((n_samples, 1), ts[i].item(), device=device)
        r_i = torch.full((n_samples, 1), ts[i-1].item(), device=device)
        z = z - (t_i - r_i) * model(z, r_i, t_i)

    x0 = z
    return z1, x0

@torch.no_grad()
def sample_posterior_rejection(
    y,                       # (B,1) or (B,)
    sigma_obs: float,
    J: int,
    prior_sampler,           # callable: prior_sampler(n, device=...) -> (n,2)
    proposal_K: int = 4096,  # proposals per batch item per iteration
    max_rounds: int = 10_000,
    device=None,
):
    """
    Rejection sampling for p(x|y) where prior is given by prior_sampler and
    likelihood is N(y | x[:,:1], sigma_obs^2).

    Returns:
      x_post: (B, J, 2)
    """
    if device is None:
        device = y.device
    if y.ndim == 1:
        y = y[:, None]
    y = y.to(device)

    B = y.shape[0]
    out = torch.empty(B, J, 2, device=device)
    filled = torch.zeros(B, dtype=torch.long, device=device)

    sigma = torch.tensor(sigma_obs, device=device, dtype=y.dtype)

    for _ in range(max_rounds):
        done = filled >= J
        if bool(done.all()):
            break

        # Propose B * proposal_K samples from the prior
        x_prop = prior_sampler(B * proposal_K).reshape(B, proposal_K, 2)
        x_prop = torch.tensor(x_prop, device=device, dtype=torch.float32)

        # Acceptance prob: exp(-(y - x1)^2 / (2 sigma^2))
        # y: (B,1) -> (B,proposal_K)
        diff = (y[:, 0:1] - x_prop[:, :, 0]) / sigma
        accept_prob = torch.exp(-0.5 * diff**2)  # (B, proposal_K)

        u = torch.rand(B, proposal_K, device=device, dtype=accept_prob.dtype)
        accept = u < accept_prob  # (B, proposal_K)

        # Fill output (loop over B only; still fast in practice)
        for b in range(B):
            if filled[b] >= J:
                continue
            xb_acc = x_prop[b][accept[b]]  # (n_acc,2)
            if xb_acc.numel() == 0:
                continue
            need = J - int(filled[b].item())
            take = min(need, xb_acc.shape[0])
            out[b, filled[b]:filled[b] + take] = xb_acc[:take]
            filled[b] += take

    if not bool((filled >= J).all()):
        # If this triggers, sigma is probably very small or proposal_K too small.
        raise RuntimeError(
            f"Rejection sampler did not finish: filled min={int(filled.min())}/{J}. "
            f"Try increasing proposal_K or sigma_obs, or use importance resampling."
        )

    return out

# Metrics
def nlpd(y, x_samples, sigma_obs):
    """
    Negative log predictive density (NLPD)

    y:         (B, 1)
    x_samples: (B, J, D)
    sigma_obs: float

    returns: (B,) tensor
    """
    B, J, D = x_samples.shape
    Ax = x_samples[..., 0]           # (B, J)
    y0 = y.squeeze(-1).unsqueeze(1)  # (B, 1)
    log_norm = -0.5 * math.log(2 * math.pi * (sigma_obs ** 2))
    log_probs = log_norm - 0.5 * (y0 - Ax) ** 2 / (sigma_obs ** 2)   # (B, J)
    log_py = torch.logsumexp(log_probs, dim=1) - math.log(J)
    return -log_py

def crps(x_true, x_samples):
    """
    x_true: (B, D)
    x_samples: (B, J, D)
    """
    B, J, D = x_samples.shape
    x_true_exp = x_true.unsqueeze(1).expand(B, J, D)  # (B, J, D)
    term1 = torch.mean(torch.norm(x_samples - x_true_exp, dim=2), dim=1)  # (B,)
    x1 = x_samples.unsqueeze(2)  # (B, J, 1, D)
    x2 = x_samples.unsqueeze(1)  # (B, 1, J, D)
    pairwise_dists = torch.norm(x1 - x2, dim=3)  # (B, J, J)
    term2 = 0.5 * torch.mean(pairwise_dists, dim=(1, 2))  # (B,)
    return term1 - term2  # (B,)

def support_accuracy(
    x01, n_squares=4, filled_parity=0, eps=0.0
):
    """
    Batched support accuracy for checkerboard.

    Parameters
    ----------
    x01 : torch.Tensor
        Shape (B, J, 2), points expected in [0,1]^2
    n_squares : int
        Number of checkerboard squares per side
    filled_parity : int
        Parity defining filled squares: (i + j) % 2 == filled_parity
    eps : float
        Optional margin to shrink squares slightly

    Returns
    -------
    acc : torch.Tensor
        Shape (B,), support accuracy per batch element
    """
    assert x01.ndim == 3 and x01.shape[-1] == 2
    x = x01
    
    # --- inside [0,1]^2 (with margin eps)
    inside = (
        (x[..., 0] >= 0.0 + eps) & (x[..., 0] < 1.0 - eps) &
        (x[..., 1] >= 0.0 + eps) & (x[..., 1] < 1.0 - eps)
    )  # (B, J)

    # --- checkerboard indices
    ij = torch.floor(x * n_squares).long()          # (B, J, 2)
    ij = ij.clamp(0, n_squares - 1)

    filled = ((ij[..., 0] + ij[..., 1]) % 2 == filled_parity)  # (B, J)

    # --- valid support mask
    support = inside & filled                         # (B, J)

    # --- accuracy per batch element
    return support.float().mean(dim=1)                # (B,)

@torch.no_grad()
def pairwise_sq_dists(x, y):
    # x: (N,D), y: (M,D)
    x2 = (x**2).sum(dim=1, keepdim=True)          # (N,1)
    y2 = (y**2).sum(dim=1, keepdim=True).T        # (1,M)
    return x2 + y2 - 2.0 * (x @ y.T)              # (N,M)

@torch.no_grad()
def median_heuristic_sigma(x, y, max_samples=2000, eps=1e-12):
    # Use a subsample to estimate median distance
    x_ = x[:max_samples]
    y_ = y[:max_samples]
    d2 = pairwise_sq_dists(x_, y_)                # (n,m)
    med = torch.median(d2)                        # median of squared distances
    return torch.sqrt(torch.clamp(med, min=eps))

@torch.no_grad()
def mmd_squared(x, y, sigma=None, max_sigma_samples=2000):
    """
    Unbiased MMD^2 with RBF kernel.
    x: (N,D), y: (M,D)
    returns scalar tensor (MMD^2)
    """
    assert x.ndim == 2 and y.ndim == 2
    N, M = x.shape[0], y.shape[0]
    assert N > 1 and M > 1

    if sigma is None:
        sigma = median_heuristic_sigma(x, y, max_samples=max_sigma_samples)
    gamma = 1.0 / (2.0 * sigma**2)

    Kxx = torch.exp(-gamma * pairwise_sq_dists(x, x))
    Kyy = torch.exp(-gamma * pairwise_sq_dists(y, y))
    Kxy = torch.exp(-gamma * pairwise_sq_dists(x, y))

    # remove diagonal for unbiased estimate
    Kxx = Kxx - torch.diag(torch.diag(Kxx))
    Kyy = Kyy - torch.diag(torch.diag(Kyy))

    mmd2 = Kxx.sum() / (N * (N - 1)) + Kyy.sum() / (M * (M - 1)) - 2.0 * Kxy.mean()
    return mmd2

if __name__  == "__main__":
    # Set parameters
    K = 1
    sigma = 0.1
    B = 10_000
    J = 100
    alpha = 1.0
    tau_list = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]
    ema = True

    # Get validation data
    prior_sampler = partial(sample_checkerboard, n_squares=4)
    x_prior = prior_sampler(B)
    x_prior = torch.tensor(x_prior, dtype=torch.float32).to(device)
    y_val = sample_y(x_prior, sigma) # Shape (B, 1)
    y_prime = sample_y(x_prior, sigma) # Shape (B, 1)

    # Get true posterior samples (limit to 1000 samples for speed)
    print("Sampling true posterior via rejection sampling...")
    x_posterior = sample_posterior_rejection(y_val[:1000], sigma, J, prior_sampler) # (1000, J, 2)

    for tau in tau_list:
        print(f"--- Evaluating tau={tau}, alpha={alpha} ---")

        # Load model. Adjust `dir_name` to the folder holding your trained
        # checkerboard adapters (under `ckpts/`); defaults to this folder.
        dir_name = os.path.dirname(os.path.abspath(__file__))
        if ema:
            if alpha == 0.5:
                fname = f"vfm_adapter_tau_{tau}_ema"
            elif alpha == 1.0:
                fname = f"vfm_adapter_tau_{tau}_alpha_{alpha}_ema"
        else:
            if alpha == 0.5:
                fname = f"vfm_adapter_tau_{tau}"
            elif alpha == 1.0:
                fname = f"vfm_adapter_tau_{tau}_alpha_{alpha}"
        ckpt_path = os.path.join(dir_name, "ckpts", f"{fname}.pt")
        ckpt = torch.load(ckpt_path)
        adapter = Adapter(hidden=256, depth=4, z_dim=2).to(device)
        model = MeanFlowMLP(hidden=512, depth=6).to(device)
        adapter.load_state_dict(ckpt["adapter"], strict=True)
        model.load_state_dict(ckpt["mf_model"], strict=True)
        adapter.eval()
        model.eval()

        # Compute metrics
        results = {}
        with torch.no_grad():
            mu_val, log_std_val = adapter(y_val) # Shape (B, 2)
            z_val = reparam_sample(mu_val, log_std_val, J=J) # Shape (B, J, 2)
            eps = torch.randn_like(z_val)
            xhat_val = f_theta(z_val, model, K) # Shape (B, J, 2)
            xhat_uncond = f_theta(eps, model, K) # Shape (B, J, 2)
            yhat_val = xhat_val[:, :, :1] # Shape (B, J, 1)

            val_nlpd = nlpd(y_prime, xhat_val, sigma)
            print(f"NLPD: {val_nlpd.mean():.3f} +/- {val_nlpd.std():.3f}")
            results['nlpd'] = val_nlpd.detach().cpu().numpy()

            val_crps = crps(x_prior, xhat_val)
            print(f"CRPS: {val_crps.mean():.3f} +/- {val_crps.std():.3f}")
            results['crps'] = val_crps.detach().cpu().numpy()

            x01 = unscale_to_unit(xhat_val)
            acc = support_accuracy(x01)
            print(f"Support Accuracy (posterior): {acc.mean():.3f} +/- {acc.std():.3f}")
            results['support_acc_post'] = acc.detach().cpu().numpy()

            x01_uncond = unscale_to_unit(xhat_uncond.transpose(0, 1)) # Shape (J, B, 1)
            acc_uncond = support_accuracy(x01_uncond)
            print(f"Support Accuracy (prior): {acc_uncond.mean():.3f} +/- {acc_uncond.std():.3f}")
            results['support_acc_prior'] = acc_uncond.detach().cpu().numpy()

            posterior_mmd_list = []
            for i in range(1000):
                posterior_mmd_list.append(torch.sqrt(torch.clamp(mmd_squared(x_posterior[i], xhat_val[i]), min=0)))
            posterior_mmds = torch.tensor(posterior_mmd_list)
            print(f"MMD (Posterior): {posterior_mmds.mean():.3f} +/- {posterior_mmds.std():.3f}")
            results['mmd_post'] = posterior_mmds.detach().cpu().numpy()

            mmd_list = []
            for _ in range(100):
                eps = torch.randn((B, 2), device=device, dtype=torch.float32)
                x_hat = f_theta(eps, model, K)
                prior_mmd = torch.sqrt(torch.clamp(mmd_squared(x_prior, x_hat), min=0))
                mmd_list.append(prior_mmd)
            mmd_list = torch.tensor(mmd_list)
            print(f"MMD (Prior): {mmd_list.mean():.3f} +/- {mmd_list.std():.3f}")
            results['mmd_prior'] = mmd_list.detach().cpu().numpy()
            print("\n")

        # Save results
        if ema:
            save_path = os.path.join(dir_name, "results", f"vfm_results_tau_{tau}_alpha_{alpha}_K_{K}_ema.npz")
        else:
            save_path = os.path.join(dir_name, "results", f"vfm_results_tau_{tau}_alpha_{alpha}_K_{K}.npz")
        np.savez(save_path, **results)

