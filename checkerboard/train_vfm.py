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

def sample_t_r(batch_size, device, ratio_r_neq_t=0.25):
    """
    Sample t,r in [0,1] with t>=r by swapping.
    Optionally set r=t for a fraction (1-ratio) of samples.
    """
    t = torch.rand(batch_size, 1, device=device)
    r = torch.rand(batch_size, 1, device=device)
    t, r = torch.maximum(t, r), torch.minimum(t, r)

    if ratio_r_neq_t < 1.0:
        keep = (torch.rand(batch_size, 1, device=device) < ratio_r_neq_t)
        r = torch.where(keep, r, t)
    return t, r

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


# Utils
def sample_y(x, sigma_obs):
    # A = [1,0], so observe first coord
    y = x[:, :1] + sigma_obs * torch.randn(x.shape[0], 1, device=x.device, dtype=x.dtype)
    return y

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

def kl(mu, log_std):
    return 0.5 * torch.sum(torch.exp(2*log_std) + mu**2 - 1.0 - 2*log_std, dim=1)

@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float = 0.9999,
    only_trainable: bool = True,   # skip params with requires_grad=False
    update_buffers: bool = True    # keep buffers (e.g., running stats) in sync
):
    """
    Exponential moving average update.
    - Parameters with requires_grad=False are skipped when only_trainable=True.
    - Buffers are copied directly (no decay) if update_buffers=True.
    """
    # Map names consistently (DDP may prepend "module.")
    def _strip(name: str) -> str:
        for pre in ("module.", "orig_mod.", "_orig_mod."):
            if name.startswith(pre):
                return name[len(pre):]
        return name

    # Build quick-lookup sets & dicts
    trainable = {_strip(n) for n, p in model.named_parameters() if p.requires_grad}
    ema_params = { _strip(n): p for n, p in ema_model.named_parameters() }
    model_params = { _strip(n): p for n, p in model.named_parameters() }

    # EMA on parameters
    for name, p_model in model_params.items():
        if only_trainable and name not in trainable:
            continue
        p_ema = ema_params.get(name, None)
        if p_ema is None:
            continue
        # Guard: only EMA float tensors
        if p_ema.is_floating_point():
            p_ema.mul_(decay).add_(p_model.data, alpha=1.0 - decay)
        else:
            p_ema.copy_(p_model.data)

    # Keep buffers in sync (no EMA decay)
    if update_buffers:
        ema_buffers = { _strip(n): b for n, b in ema_model.named_buffers() }
        for name, b_model in model.named_buffers():
            name = _strip(name)
            b_ema = ema_buffers.get(name, None)
            if b_ema is not None:
                b_ema.copy_(b_model)

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


# Plot utils
def compute_kde(data):
    # Ensure data is 2D and convert to numpy if it's a tensor
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # If data is 1D, make it 2D by adding a column of zeros
    if data.ndim == 1:
        data = np.vstack([data, np.zeros_like(data)])
    elif data.shape[1] == 1:
        data = np.vstack([data.flatten(), np.zeros_like(data.flatten())])
    else:
        data = data[:, :2].T # Transpose for kde (2, N)
    
    # If after processing, data is still not suitable for KDE
    if data.shape[0] != 2 or data.shape[1] < 2:
        print("Data dimensions not suitable for 2D KDE.")
        return None

    # Compute KDE
    try:
        kde = gaussian_kde(data)
        return kde
    except np.linalg.LinAlgError as e:
        print(f"KDE computation failed: {e}")
        return None

def get_kde_points(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    xmin, ymin = data.min(axis=0)
    xmax, ymax = data.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    xmin -= 0.05 * dx; xmax += 0.05 * dx
    ymin -= 0.05 * dy; ymax += 0.05 * dy

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 100),
        np.linspace(ymin, ymax, 100),
        indexing="xy",
    )
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return pts.T, [xmin, xmax, ymin, ymax]


if __name__ == "__main__":
    # ---------- CLI ----------
    parser = argparse.ArgumentParser(description="Train adapter with mean flow on checkerboard.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Observation noise std.")
    parser.add_argument("--tau", type=float, default=0.1, help="Data alignment parameter.")
    parser.add_argument("--tau-f", type=float, default=0.1, help="Data alignment parameter (final value if annealing).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Conditional/unconditional ratio.")
    parser.add_argument("--ratio-r-neq-t", type=float, default=0.25, help="r neq t ratio.") 
    parser.add_argument("--num-iter", type=int, default=20000, help="Training iterations.")
    parser.add_argument("--log-every", type=int, default=1000, help="Logging frequency.")
    parser.add_argument("--J", type=int, default=1000, help="Number of validation samples per input.")
    parser.add_argument("--K", type=int, default=4, help="Number of mean flow steps for sampling.")
    parser.add_argument("--p", type=float, default=1.0, help="Adaptive loss parameter p.")
    parser.add_argument("--c", type=float, default=0.01, help="Adaptive loss parameter c.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adapter learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adapter weight decay.")
    parser.add_argument("--project", type=str, default="checkerboard", help="wandb project name.")
    parser.add_argument("--dir-name", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Base directory for outputs and checkpoints (defaults to this folder).")
    parser.add_argument("--load-fm", type=str, default="checkpoints/fm_model.pt", help="Path to the pretrained flow-matching checkpoint, relative to --dir-name.")
    parser.add_argument("--ema", action="store_true", help="Use EMA in obs loss or not.")
    parser.add_argument("--no-kl", action="store_true", help="To not use KL regularisation in loss.")
    parser.add_argument("--cosine-annealing", action="store_true", help="To do cosine annealing.")
    args = parser.parse_args()

    # ---------- Hyperparameters ----------
    sigma = args.sigma
    tau = args.tau
    alpha = args.alpha
    num_iter = args.num_iter
    log_every = args.log_every
    J = args.J
    K = args.K
    dir_name = args.dir_name

    # For annealing
    tau_f = args.tau_f
    tau_0 = tau

    # ---------- wandb init ----------
    if args.ema:
        run_name = f"vfm_training_tau_{tau}_alpha_{alpha}_ema"
    else:
        run_name = f"vfm_training_tau_{tau}_alpha_{alpha}"
    if args.no_kl:
        run_name += "_no_kl"
    if args.cosine_annealing:
        run_name += "_cosine_annealing"
        
    wandb.init(project=args.project, name=run_name, config=vars(args))

    # Load models (fix path joins)
    fm_ckpt = os.path.join(dir_name, args.load_fm)
    if os.path.exists(fm_ckpt):
        fm_model = MeanFlowMLP(hidden=512, depth=6).to(device)
        fm_model.load_state_dict(torch.load(fm_ckpt, map_location=device))
    else:
        print("Flow matching model fm_model.pt not found. Skipping loading.")

    x = sample_checkerboard(20_000, n_squares=4)
    x = torch.tensor(x, dtype=torch.float32)
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Val loader
    x_val = sample_checkerboard(1000, n_squares=4)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)

    # models
    model = MeanFlowMLP(hidden=512, depth=6).to(device)
    model.load_state_dict(fm_model.state_dict())
    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)

    adapter = Adapter(hidden=256, depth=4, z_dim=2).to(device)
    opt = torch.optim.AdamW(itertools.chain(
                                adapter.parameters(),
                                model.parameters(),
                            ),
                            lr=args.lr, weight_decay=args.weight_decay)

    # Main training loop
    adapter.train()
    model.train()
    ema.eval()
    i = 0
    obs_losses = []
    kl_losses = []
    reconstruction_losses = []
    mf_losses = []
    losses = []
    data_iter = iter(loader)
    while i < num_iter:
        try:
            (xb,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (xb,) = next(data_iter)

        # Cosine annealing αt = αf + 0.5 * (α₀ - αf) * (1 + cos(π * current_step / total_steps))
        if args.cosine_annealing:
            tau = tau_f + 0.5 * (tau_0 - tau_f) * (1 + math.cos(math.pi * (i / num_iter)))
            if i % log_every == 0:
                wandb.log({"tau": tau})

        xb = xb.to(device)
        B = xb.shape[0]

        # sample observation
        y = sample_y(xb, sigma)  # (B,1)

        # q(z|y)
        mu, log_std = adapter(y)
        z = reparam_sample(mu, log_std)

        # predicted observation from frozen generator
        if args.ema:
            xhat = f_theta(z, ema, 1)          # (B,2) and detached
        else:
            xhat = f_theta(z, model, 1)
        yhat = xhat[:, :1]                  # A xhat

        # reconstruction term (Gaussian likelihood)
        obs_loss = (0.5 / (sigma**2)) * (y - yhat).pow(2).sum(dim=1)  # (B,)

        # KL term
        kl_loss = kl(mu, log_std)              # (B,)
        if args.no_kl:
            kl_loss = torch.zeros_like(kl_loss, device=device, dtype=torch.float32)

        # mean flow loss
        t, r = sample_t_r(B, device=device, ratio_r_neq_t=args.ratio_r_neq_t)
        mask = (torch.rand(B, 1, device=device) < alpha).float()
        eps = torch.randn_like(xb)
        z = mask * z + (1-mask) * eps
        xt = (1.0 - t) * xb + t * z
        vt = z - xb
        (u, dudt) = jvp(
            model,
            (xt, r, t),
            (vt, torch.zeros_like(r), torch.ones_like(t))
        )
        u_tgt = vt - (t - r) * dudt
        mf_loss_unweighted = F.mse_loss(u, u_tgt.detach()) / (t - r + 1e-6)
        mf_loss = (0.5 / (tau**2)) * mf_loss_unweighted

        # Adaptive loss
        loss = (mf_loss + obs_loss + kl_loss).mean()
        w = 1 / (loss + args.c)**args.p
        loss = w.detach() * loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        update_ema(ema, model, decay=0.0, only_trainable=False, update_buffers=True)

        # reconstruction loss
        with torch.no_grad():
            model.eval()
            zeros = torch.zeros((xb.shape[0],1), dtype=torch.float32, device=device)
            ones = torch.ones((xb.shape[0],1), dtype=torch.float32, device=device)
            reconstruction_loss = F.mse_loss(xb, model(z, zeros, ones).detach()) # Shape (B,)

        obs_losses.append(obs_loss.mean().item())
        kl_losses.append(kl_loss.mean().item())
        mf_losses.append(mf_loss_unweighted.mean().item())
        reconstruction_losses.append(reconstruction_loss.mean().item())
        losses.append(loss.item())
        i += 1
        if i % log_every == 0:
            avg_loss = sum(losses[-log_every:]) / log_every
            avg_obs = sum(obs_losses[-log_every:]) / log_every
            avg_kl = sum(kl_losses[-log_every:]) / log_every
            avg_mf = sum(mf_losses[-log_every:]) / log_every
            avg_rec = sum(reconstruction_losses[-log_every:]) / log_every
            print(f"iter {i:6d}/{num_iter} | loss {avg_loss:.6f}, obs_loss {avg_obs:.6f}, kl_loss {avg_kl:.6f}, mf_loss {avg_mf:.6f}")
            wandb.log({"total_loss": avg_loss, "obs_loss": avg_obs, "kl_loss": avg_kl, "mf_loss": avg_mf, "reconstruction_loss": avg_rec})

            # Validation
            adapter.eval()
            model.eval()
            with torch.no_grad():
                y_val = sample_y(x_val, sigma) # Shape (B, 1)
                mu_val, log_std_val = adapter(y_val) # Shape (B, 2)
                z_val = reparam_sample(mu_val, log_std_val, J=J) # Shape (B, J, 2)
                xhat_val = f_theta(z_val, model, 1) # Shape (B, J, 2)
                yhat_val = xhat_val[:, :, :1] # Shape (B, J, 1)

                # Compute validation metrics
                val_nlpd = nlpd(y_val, xhat_val, sigma).mean().item()
                val_crps = crps(x_val, xhat_val).mean().item()
                print(f"*** Validation NLPD: {val_nlpd:.6f}, CRPS: {val_crps:.6f} ***")
                wandb.log({"val_nlpd": val_nlpd, "val_crps": val_crps})

            fig0 = plt.figure()
            plt.plot(reconstruction_losses, label="reconstruction loss")
            plt.plot(mf_losses, label="mf loss")
            plt.legend()

            # Plot KDEs
            fig, axs = plt.subplots(2, 4, figsize=(16, 8))
            for j, y_val in enumerate([-1.5, -0.5, 0.5, 1.5]):
                with torch.no_grad():
                    y0 = torch.tensor([[y_val]], device=device, dtype=torch.float32)
                    mu, log_std = adapter(y0.repeat(J, 1)) # (J, 2)
                    z = reparam_sample(mu, log_std) # (J, 2)
                    x_samp = f_theta(z, model, K).cpu() # (J, 2)

                z_np = z.detach().cpu().numpy()

                kde_z_np = compute_kde(z_np)
                z_pts, z_range = get_kde_points(z_np)
                kde_x_samp = compute_kde(x_samp)
                x_pts, x_range = get_kde_points(x_samp)

                dens_z = kde_z_np(z_pts)
                dens_z = np.asarray(dens_z).reshape(100, 100)
                dens_x = kde_x_samp(x_pts)
                dens_x = np.asarray(dens_x).reshape(100, 100)

                if K == 1:
                    z1, x0 = meanflow_one_step_with_latents(model, n_samples=10_000, device=device)
                else:
                    z1, x0 = meanflow_k_steps_with_latents(model, n_samples=10_000, K=K, device=device)

                x01 = unscale_to_unit(x0)  # map to [0,1]^2 for block assignment
                block_id, ij = checkerboard_block_id(x01, n_squares=4, filled_parity=0)
                valid = block_id >= 0
                unique_blocks = torch.unique(block_id[valid]).tolist()
                block_to_compact = {b: k for k, b in enumerate(unique_blocks)}
                labels = torch.full_like(block_id, -1)
                for b, k in block_to_compact.items():
                    labels[block_id == b] = k

                z1_np = z1.detach().cpu().numpy()
                x01_np = x0.detach().cpu().numpy()
                lab_np = labels.detach().cpu().numpy()

                valid = lab_np >= 0
                invalid = ~valid
                cmap = plt.get_cmap("tab10")   # gives C0..C9
                colors = cmap(lab_np[valid] % cmap.N)

                # ---------- noise space ----------
                axs[0, j].scatter(z1_np[invalid, 0], z1_np[invalid, 1], s=1, c="lightgray", alpha=0.4, label="outside blocks")
                axs[0, j].scatter(z1_np[valid, 0], z1_np[valid, 1], s=1, c=colors)
                axs[0, j].contour(dens_z, origin='lower', extent=z_range)
                axs[0, j].set_title(f"y = {y_val:.1f}")
                if j == 0:
                    axs[0, j].set_ylabel("Latent space")

                # ---------- data space ----------
                axs[1, j].scatter(x01_np[invalid, 0], x01_np[invalid, 1], s=1,c="lightgray",alpha=0.4)
                axs[1, j].scatter(x01_np[valid, 0], x01_np[valid, 1], s=1, c=colors)
                axs[1, j].contour(dens_x, origin='lower', extent=x_range)
                axs[1, j].vlines(y_val, -2, 2, linestyles="dashed", colors="black")
                if j == 0:
                    axs[1, j].set_ylabel("Data space")

            plt.axis("equal")
            plt.tight_layout()

            # Log figure to wandb
            wandb.log({"plots/kde": wandb.Image(fig), "plots/loss_comparison": wandb.Image(fig0)})
            plt.close(fig)
            plt.close(fig0)

            adapter.train()
            model.train()

    # Save checkpoints
    if args.ema:
        fname = f"ckpts/vfm_adapter_tau_{tau}_alpha_{alpha}_ema"
    else:
        fname = f"ckpts/vfm_adapter_tau_{tau}_alpha_{alpha}"
    if args.no_kl:
        fname += "_no_kl"
    if args.cosine_annealing:
        fname += "_cosine_annealing"
    fname += ".pt"
    
        
    torch.save({
        "adapter": adapter.state_dict(),
        "mf_model": model.state_dict(),
    }, os.path.join(dir_name, fname))
    wandb.save(os.path.join(dir_name, fname))




