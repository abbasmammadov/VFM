from functools import partial

import torch
import numpy as np

import copy


def interpolant(x, t, path_type="linear", eps=None, reverse=False, mask=None):
    eps = torch.randn_like(x) if eps is None else eps
    if mask is not None:
        eps_pure_noise = torch.randn_like(x)
        eps = eps * mask + eps_pure_noise * (1 - mask)
    if path_type == "linear":
        alpha_t = 1 - t
        sigma_t = t
        d_alpha_t = -1
        d_sigma_t =  1
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = (-np.pi / 2) * torch.sin(t * np.pi / 2)
        d_sigma_t = (np.pi / 2) * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError()
    if reverse:
        alpha_t, sigma_t = sigma_t, alpha_t
        d_alpha_t, d_sigma_t = d_sigma_t, d_alpha_t
    x_t = alpha_t * x + sigma_t * eps
    v_t = d_alpha_t * x + d_sigma_t * eps
    return x_t, v_t


def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    return torch.sum(x, dim=list(range(1, len(x.size()))))

def mse_loss(x, y):
    err = (x - y) ** 2
    loss = torch.mean(err, dim=list(range(1, len(x.size()))))
    return loss, loss

def adaptive(loss, p=1.0, c=0.01, w=torch.tensor(0.0)):
    """Adaptive re-weighting of the loss by its own (detached) magnitude, as in
    the adaptive weighting used by MeanFlow, stabilising the multi-term objective."""
    weights = 1.0 / (loss + c) ** p
    weight = weights.detach()
    loss = loss * weight
    if (w != 0.0).all():
        loss = (1 / w.exp()) * loss + w
    return loss


def compute_guidance(model, x, t, y, v_t, drop_ids, omega=0.5, g_min=0.0, g_max=1.0, g_type="default"):
    if g_type == "default":
        return v_t
    elif g_type == "mg":
        drop_ids = drop_ids.reshape(-1, 1, 1, 1)
        cfg_ids = (t >= g_min) & (t <= g_max) & (~drop_ids)
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
        with torch.no_grad():
            x_2 = torch.cat([x, x], dim=0)
            t_2 = torch.cat([t, t], dim=0)
            y_2 = torch.cat([y, y_null], dim=0)
            v = model(x_2, t_2, t_2, y_2).to(torch.float32)
            v_cond, v_uncond = v.chunk(2, dim=0)
        v_tgt = v_t + omega * (v_cond - v_uncond)
        v_tgt = torch.where(cfg_ids, v_tgt, v_t)
        return v_tgt
    elif g_type == "mg_learn":
        drop_ids = drop_ids.reshape(-1, 1, 1, 1)
        cfg_ids = (t >= g_min) & (t <= g_max) & (~drop_ids)
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
        x_2 = torch.cat([x, x], dim=0)
        t_2 = torch.cat([t, t], dim=0)
        y_2 = torch.cat([y, y_null], dim=0)
        v = model(x_2, t_2, t_2, y_2).to(torch.float32)
        v_cond, v_uncond = v.chunk(2, dim=0)
        v_tgt = v_t + omega * (v_cond - v_uncond)
        v_tgt = torch.where(cfg_ids, v_tgt, v_t)
        return v_tgt
    elif g_type == "distill":
        cfg_ids = (t >= g_min) & (t <= g_max)
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
        with torch.no_grad():
            x_2 = torch.cat([x, x], dim=0)
            t_2 = torch.cat([t, t], dim=0)
            y_2 = torch.cat([y, y_null], dim=0)
            v = model(x_2, t_2, y_2).to(torch.float32)
            v_cond, v_uncond = v.chunk(2, dim=0)
        v_tgt = v_uncond + omega * (v_cond - v_uncond)
        v_tgt = torch.where(cfg_ids, v_tgt, v_cond)  # no class dropout for distillation
        return v_tgt
    else:
        raise NotImplementedError(f"Unknown guidance type: {g_type}")


def ln_sampler(z, path_type="linear"):
    if path_type == "linear":
        return torch.sigmoid(z)
    elif path_type == "cosine":
        return 2 / np.pi * torch.atan(z)
    else:
        raise NotImplementedError(f"Unknown path_type: {path_type}")


class AdapterLoss:
    def __init__(
        self,
        P_mean:         float = -0.4,
        P_std:          float = 1.0,
        P_mean_t:       float = -0.2,
        P_std_t:        float = 1.0,
        P_mean_r:       float = 0.2,
        P_std_r:        float = 1.0,
        cfg_prob:       float = 0.1,
        omega:          float = 0.0,
        g_min:          float = 0.0,
        g_max:          float = 1.0,
        path_type:      str   = "linear",
        g_type:         str   = "default"
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.P_mean_t = P_mean_t
        self.P_std_t = P_std_t
        self.P_mean_r = P_mean_r
        self.P_std_r = P_std_r
        self.cfg_prob = cfg_prob
        self.path_type = path_type
        self.guidance_kwargs = dict(
            omega=omega,
            g_min=g_min,
            g_max=g_max,
            g_type=g_type,
        )

    def cond_drop(self, y):
        bsz = y.shape[0]
        drop_ids = torch.rand(bsz, device=y.device) < self.cfg_prob
        y = torch.where(drop_ids, 1000, y)
        return y, drop_ids

    def sample_times(self, bsz, reverse=False):
        t_fm = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std + self.P_mean, self.path_type)
        ln_1 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_t + self.P_mean_t, self.path_type)
        ln_2 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_r + self.P_mean_r, self.path_type)
        t_mf, r_mf = torch.maximum(ln_1, ln_2), torch.minimum(ln_1, ln_2)
        if reverse:
            return 1 - t_fm, 1 - t_mf, 1 - r_mf # preserves DMF distribution just reversed literally on time axis along (0, 1)
            # return t_fm, r_mf, t_mf
        return t_fm, t_mf, r_mf

    def __call__(self, model, unwrapped_model, ema,
                 adapter_model, unwrapped_adapter_model, ema_adapter,
                 img, latent, label, model_t=None,
                 decode_func=None, inv_probs=None, data_loss_learn_fm="ema_path", 
                 train_adapter_only=False,
                 **kwargs):
        """Compute the joint VFM objective (Eq. 6 in the paper).

        For a batch of clean images we (i) sample a random inverse problem per
        example and form the observation ``y = A(x) + eps``; (ii) encode ``y``
        with the noise adapter to obtain ``q_phi(z|y) = N(mu, sigma^2 I)`` and a
        reparameterised sample ``z``; and (iii) sum three terms:

            1. ``data_consistency_loss`` -- observation likelihood
               ``||A(f_theta(z)) - y||^2`` (flow map run through ``ema`` so the
               observation term does not back-propagate into the flow map).
            2. ``kl_loss`` -- ``KL(q_phi(z|y) || N(0, I))``.
            3. ``flow_loss`` -- the flow-matching + MeanFlow objective that keeps
               the jointly-trained flow map a valid one/few-step generator.

        ``reverse`` selects the time convention of the pretrained backbone:
        ``True`` builds the interpolant from t=0 (data) to t=1 (noise), matching
        SiT/DMF checkpoints; ``False`` uses the t=1->0 convention.
        """
        reverse = kwargs.get('reverse', False)
        # data loss setup
        batch_size = img.shape[0]
        seeds_for_inv_problem = torch.randint(0, 2**32 - 1, (batch_size,)).tolist()
        inv_prob_ids = torch.randint(0, len(inv_probs), (batch_size,)).to(img.device)
        inv_prob_ids_list = inv_prob_ids.tolist()
        operators = [copy.copy(inv_probs[inv_prob_id]) for inv_prob_id in inv_prob_ids_list]
        for i, op in enumerate(operators):
            op.update(seeds_for_inv_problem[i])
        # Forward-degrade each image, add measurement noise, and map back to
        # image space via the operator transpose so the adapter always sees a
        # 3x256x256 tensor regardless of the inverse problem.
        y_list_tmp = [operators[i](img[i:i+1]) for i in range(batch_size)]
        y_list_fwd = [y_list_tmp[i] + torch.randn_like(y_list_tmp[i]) * 0.05 for i in range(batch_size)]
        y_list = [operators[i].transpose(y_list_fwd[i]) for i in range(batch_size)]
        ys = torch.cat(y_list, dim=0).to(img.device)

        # Amortised variational posterior q_phi(z|y) = N(mu, exp(logvar)) with a
        # reparameterised sample z.
        mu, logvar, z = adapter_model(ys, inv_prob_ids, return_sample=True)
        z = z.to(img.device)

        # KL(q_phi(z|y) || N(0, I)).
        kl_loss = 0.5 * (mu**2 + torch.exp(logvar) - logvar - 1)
        kl_loss = torch.mean(kl_loss, dim=list(range(1, len(kl_loss.size()))))
        kl_loss_weight = kwargs.get('KL_loss_weight', 1.0)

        # Observation term: map z one-step to data via the flow map, decode, and
        # re-measure. Using the EMA copy ("ema_path") stops the observation term
        # from back-propagating into the flow map, so f_theta is shaped only by
        # the flow objective while the adapter is free to fit the posterior.
        if data_loss_learn_fm == "ema_path":
            model_for_data_loss = ema
            model_for_data_loss.eval()
            for param in model_for_data_loss.parameters():
                param.requires_grad = False
        elif data_loss_learn_fm == "learn":
            model_for_data_loss = model
        else:
            raise NotImplementedError(f"Unknown data_loss_learn_fm: {data_loss_learn_fm}")

        t_ones = torch.ones((batch_size, 1, 1, 1), device=img.device)
        t_zeros = torch.zeros((batch_size, 1, 1, 1), device=img.device)
        t_start, t_end = (t_zeros, t_ones) if reverse else (t_ones, t_zeros)
        fm_velocity = model_for_data_loss(z, t_start, t_end, label, return_logvar=False)
        fm_recon = z + float((t_end - t_start).unique().item()) * fm_velocity
        x_rec = decode_func(fm_recon)
        y_rec_list = [operators[i].transpose(operators[i](x_rec[i:i+1])) for i in range(batch_size)]
        y_recs = torch.cat(y_rec_list, dim=0)
        data_consistency_loss = 200 * mean_flat((y_recs - ys) ** 2)
        data_loss_weight = kwargs.get('data_loss_weight', 1.0)

        if model_t is None:
            model_t = model
        y, drop_ids = self.cond_drop(label)
        t_fm, t_mf, r_mf = self.sample_times(latent.shape[0], reverse=reverse)
        t_fm, t_mf, r_mf = t_fm.to(latent.device), t_mf.to(latent.device), r_mf.to(latent.device)

        # Flow-matching term. `mask` mixes the adapter noise z (kept fraction) with
        # fresh noise so the flow map is trained on both adapter-coupled and i.i.d.
        # noise, preserving unconditional generation quality.
        ratio = 0.8
        mask = (torch.rand(latent.shape[0], 1, 1, 1, device=latent.device) < ratio).float()
        x_t_fm, v_t_fm = interpolant(latent, t_fm, path_type=self.path_type, eps=z, reverse=reverse, mask=mask)
        v_fm = model(x_t_fm, t_fm, t_fm, y, return_logvar=False)
        v_tgt_fm = compute_guidance(
            model_t, x_t_fm, t_fm, y, v_t_fm, drop_ids, **self.guidance_kwargs
        ).detach()
        fm_loss_log, fm_loss = mse_loss(v_fm, v_tgt_fm)

        # MeanFlow term: enforce the Eulerian identity via a JVP w.r.t. t.
        x_t_mf, v_t_mf = interpolant(latent, t_mf, path_type=self.path_type, eps=z, reverse=reverse, mask=mask)
        v_tgt_mf = compute_guidance(
            model_t, x_t_mf, t_mf, y, v_t_mf, drop_ids, **self.guidance_kwargs
        ).detach()

        unwrapped_model = unwrapped_model.module if hasattr(unwrapped_model, "module") else unwrapped_model
        model_fn = partial(unwrapped_model, y=y, return_logvar=False)
        primals = (x_t_mf, t_mf, r_mf)
        tangents = (v_tgt_mf, torch.ones_like(t_mf), torch.zeros_like(r_mf))
        u, du_dt = torch.func.jvp(model_fn, primals, tangents)
        u_tgt = v_tgt_mf + (r_mf - t_mf) * du_dt
        u_tgt = u_tgt.detach()
        mf_loss_log, mf_loss = mse_loss(u, u_tgt)
        flow_loss = (fm_loss + mf_loss) * 0.5
        fm_loss_weight = kwargs.get('FM_loss_weight', 1.0)

        loss = (1-train_adapter_only) * (fm_loss_weight * flow_loss) + (data_loss_weight * data_consistency_loss) + (kl_loss_weight * kl_loss)
        if not train_adapter_only:
            loss = adaptive(loss)

        loss_dict = {
            "fm_loss_log": fm_loss_log,
            "fm_loss": fm_loss,
            "mf_loss_log": mf_loss_log,
            "mf_loss": mf_loss,
            "data_consistency_loss": data_consistency_loss,
            "kl_loss": kl_loss,
            "flow_loss": flow_loss,
            "total_loss": loss,
            "mu_mean": mu.abs().mean(),
            "sigma_mean": logvar.exp().sqrt().mean(),
        }
        return loss, loss_dict

