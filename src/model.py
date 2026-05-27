import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        scale = torch.exp(torch.arange(half_dim, device=device) * -scale)
        scale = time[:, None] * scale[None, :]
        return torch.cat((scale.sin(), scale.cos()), dim=-1)


def normalize_numeric_features(raw_numeric_features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    features = raw_numeric_features.to(dtype=torch.float32)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = torch.sign(features) * torch.log1p(features.abs())
    features = features.clamp(min=-20.0, max=20.0)
    return features.clamp(min=-20.0 + eps, max=20.0 - eps)


class NumericEncoder(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = cond_dim if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(self, numeric_features: torch.Tensor) -> torch.Tensor:
        return self.net(numeric_features)


class TableEncoder(nn.Module):
    def __init__(
        self,
        hf_model_name: str,
        cond_dim: int,
        numeric_feature_dim: int = 6,
        use_col_context_attn: bool = True,
        col_attn_heads: int = 4,
        col_attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hf_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden_size, cond_dim)
        self.numeric_encoder = NumericEncoder(numeric_feature_dim, cond_dim)
        self.fusion_gate = nn.Linear(cond_dim * 2, cond_dim)
        self.use_col_context_attn = use_col_context_attn
        heads = col_attn_heads if col_attn_heads > 0 and cond_dim % col_attn_heads == 0 else 1
        self.col_attn = nn.MultiheadAttention(
            embed_dim=cond_dim,
            num_heads=heads,
            dropout=col_attn_dropout,
            batch_first=True,
        )
        self.col_norm = nn.LayerNorm(cond_dim)
        self.col_score = nn.Linear(cond_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cls_indexes: torch.Tensor,
        numeric_features: Optional[torch.Tensor] = None,
        numeric_feature_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.proj(hidden)

        attention = attention_mask.unsqueeze(-1).float()
        global_tokens = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp_min(1.0)

        batch_size, max_cols = cls_indexes.shape
        hidden_dim = hidden.size(-1)
        gather_index = cls_indexes.clamp_min(0).unsqueeze(-1).expand(batch_size, max_cols, hidden_dim)
        column_text = torch.gather(hidden, dim=1, index=gather_index)
        valid_columns = cls_indexes >= 0
        valid_mask = valid_columns.unsqueeze(-1).float()
        column_text = column_text * valid_mask

        columns = column_text
        if numeric_features is not None:
            raw_numeric_valid = torch.isfinite(numeric_features).any(dim=-1)
            numeric_features = normalize_numeric_features(numeric_features).to(device=hidden.device, dtype=hidden.dtype)
            column_numeric = self.numeric_encoder(numeric_features)
            if numeric_feature_mask is None:
                numeric_valid = raw_numeric_valid.to(device=hidden.device)
            else:
                numeric_valid = numeric_feature_mask.to(device=hidden.device).bool()
            numeric_valid = numeric_valid & valid_columns
            gate = torch.sigmoid(self.fusion_gate(torch.cat([column_text, column_numeric], dim=-1)))
            fused_columns = gate * column_text + (1.0 - gate) * column_numeric
            columns = torch.where(numeric_valid.unsqueeze(-1), fused_columns, column_text)
            columns = columns * valid_mask

        if not self.use_col_context_attn:
            return global_tokens, columns

        key_padding_mask = ~valid_columns
        attended_columns, _ = self.col_attn(columns, columns, columns, key_padding_mask=key_padding_mask, need_weights=False)
        columns = self.col_norm(columns + attended_columns) * valid_mask
        scores = self.col_score(columns).squeeze(-1)
        scores = scores.masked_fill(~valid_columns, -1e4)
        attn_weights = torch.softmax(scores, dim=1) * valid_columns.float()
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        context = (columns * attn_weights.unsqueeze(-1)).sum(dim=1)
        has_valid = valid_columns.any(dim=1, keepdim=True)
        global_context = torch.where(has_valid, context, global_tokens)
        return global_context, columns


class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        y_dim: int,
        cond_dim: int,
        time_dim: int = 256,
        hidden: int = 512,
        diffuser_type: str = "mlp1",
    ):
        super().__init__()
        self.time_emb = SinusoidalPositionEmbeddings(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        input_dim = y_dim + cond_dim * 2 + hidden
        if diffuser_type == "mlp1":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, y_dim),
            )
        elif diffuser_type == "mlp2":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden * 2),
                nn.GELU(),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, y_dim),
            )
        else:
            raise ValueError(f"unknown diffuser_type: {diffuser_type}")
        self.none_g = nn.Embedding(1, cond_dim)
        self.none_c = nn.Embedding(1, cond_dim)

    def forward(self, e_t: torch.Tensor, t: torch.Tensor, g: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_mlp(self.time_emb(t))
        denoiser_input = torch.cat([e_t, g, c, time_hidden], dim=-1)
        return F.normalize(self.net(denoiser_input), dim=-1)

    def forward_uncon(self, e_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = e_t.size(0)
        indices = torch.zeros(batch_size, device=e_t.device, dtype=torch.long)
        return self.forward(e_t, t, self.none_g(indices), self.none_c(indices))


class TableConditionalDiffusion(nn.Module):
    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cum = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cum = alphas_cum / alphas_cum[0]
        betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
        return betas.clamp(1e-5, 0.999)

    def __init__(
        self,
        hf_model_name: str,
        num_labels: int,
        y_dim: int = 768,
        cond_dim: int = 768,
        numeric_feature_dim: int = 6,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        time_dim: int = 768,
        diffuser_type: str = "mlp1",
        hidden: int = 1024,
        cond_drop_prob: float = 0.2,
        table_noise_shared: bool = True,
        beta_schedule: str = "cosine",
        use_col_context_attn: bool = True,
        col_attn_heads: int = 4,
        col_attn_dropout: float = 0.1,
        col_weight: float = 1.0,
        gem_temp: float = 0.1,
        lambda_recon: float = 0.1,
        mu_ssm: float = 10.0,
        ssm_temp: float = 0.1,
    ):
        super().__init__()
        if cond_dim != y_dim:
            raise ValueError(f"cond_dim ({cond_dim}) must equal y_dim ({y_dim})")
        self.T = T
        self.y_dim = y_dim
        self.cond_drop_prob = cond_drop_prob
        self.table_noise_shared = table_noise_shared
        self.col_weight = col_weight
        self.gem_temp = gem_temp
        self.lambda_recon = lambda_recon
        self.mu_ssm = mu_ssm
        self.ssm_temp = ssm_temp

        self.encoder = TableEncoder(
            hf_model_name=hf_model_name,
            cond_dim=cond_dim,
            numeric_feature_dim=numeric_feature_dim,
            use_col_context_attn=use_col_context_attn,
            col_attn_heads=col_attn_heads,
            col_attn_dropout=col_attn_dropout,
        )
        self.denoiser = ConditionalDenoiser(y_dim, cond_dim, time_dim, hidden, diffuser_type)
        self.label_emb = nn.Embedding(num_labels, y_dim)

        if beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(T)
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        else:
            raise ValueError(f"unsupported beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cum", alphas_cum)
        self.register_buffer("sqrt_alphas_cum", torch.sqrt(alphas_cum))
        self.register_buffer("sqrt_one_minus_alphas_cum", torch.sqrt(1.0 - alphas_cum))

    def _norm_label_emb(self) -> torch.Tensor:
        return F.normalize(self.label_emb.weight, dim=-1)

    def _build_x0(self, label_ids: torch.Tensor) -> torch.Tensor:
        x0 = torch.empty(label_ids.size(0), self.y_dim, device=label_ids.device)
        labeled = label_ids >= 0
        if labeled.any():
            x0[labeled] = F.normalize(self.label_emb(label_ids[labeled]), dim=-1)
        if (~labeled).any():
            x0[~labeled] = float("nan")
        return x0

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0)
        alpha = self.sqrt_alphas_cum[t].unsqueeze(-1)
        sigma = self.sqrt_one_minus_alphas_cum[t].unsqueeze(-1)
        return F.normalize(alpha * x0 + sigma * eps, dim=-1)

    def _make_x_t(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        labeled = ~torch.isnan(x0).any(dim=-1)
        if eps is None:
            eps = torch.randn_like(x0)
        x_t = torch.empty_like(x0)
        if labeled.any():
            x_t[labeled] = self.q_sample(x0[labeled], t[labeled], eps[labeled])
        if (~labeled).any():
            x_t[~labeled] = F.normalize(torch.randn_like(x0[~labeled]), dim=-1)
        return x_t

    def _apply_cond_dropout(self, g: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not self.training) or self.cond_drop_prob <= 0:
            return g, c
        drop_mask = torch.rand(g.size(0), device=g.device) < self.cond_drop_prob
        if drop_mask.any():
            g = g.clone()
            c = c.clone()
            indices = torch.zeros(drop_mask.sum(), device=g.device, dtype=torch.long)
            g[drop_mask] = self.denoiser.none_g(indices)
            c[drop_mask] = self.denoiser.none_c(indices)
        return g, c

    def forward_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cls_indexes: torch.Tensor,
        col_table_ids: torch.Tensor,
        col_pos_in_table: torch.Tensor,
        col_label_ids: torch.Tensor,
        numeric_features: Optional[torch.Tensor] = None,
        numeric_feature_mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ):
        global_context, column_context = self.encoder(
            input_ids,
            attention_mask,
            cls_indexes,
            numeric_features=numeric_features,
            numeric_feature_mask=numeric_feature_mask,
        )
        g = global_context[col_table_ids]
        c = column_context[col_table_ids, col_pos_in_table]

        label_embeddings = self._norm_label_emb()
        labeled_mask = col_label_ids >= 0
        if labeled_mask.any():
            labels = col_label_ids[labeled_mask]
            column_repr = F.normalize(c[labeled_mask], dim=-1)
            logits_col = (column_repr @ label_embeddings.t()) / self.gem_temp
            l_col = F.cross_entropy(logits_col, labels)
        else:
            l_col = g.sum() * 0.0

        g, c = self._apply_cond_dropout(g, c)
        x0 = self._build_x0(col_label_ids)
        num_columns = x0.size(0)

        if self.table_noise_shared and num_columns > 0:
            num_tables = int(col_table_ids.max().item()) + 1
            t_table = torch.randint(0, self.T, (num_tables,), device=x0.device)
            eps_table = torch.randn(num_tables, self.y_dim, device=x0.device, dtype=x0.dtype)
            t = t_table[col_table_ids]
            eps = eps_table[col_table_ids]
        else:
            t = torch.randint(0, self.T, (num_columns,), device=x0.device)
            eps = None

        x_t = self._make_x_t(x0, t, eps)
        if not labeled_mask.any():
            zero = g.sum() * 0.0
            if return_components:
                return zero, {"L_col": 0.0, "L_recon": 0.0, "L_ssm": 0.0}
            return zero

        x0_hat = self.denoiser(x_t, t, g, c)
        x0_labeled = x0[labeled_mask]
        x0_hat_labeled = x0_hat[labeled_mask]
        l_recon = F.mse_loss(x0_hat_labeled, x0_labeled)
        logits_ssm = (F.normalize(x0_hat_labeled, dim=-1) @ label_embeddings.t()) / self.ssm_temp
        l_ssm = F.cross_entropy(logits_ssm, col_label_ids[labeled_mask])
        total = self.col_weight * l_col + self.lambda_recon * l_recon + self.mu_ssm * l_ssm

        if return_components:
            return total, {
                "L_col": l_col.item(),
                "L_recon": l_recon.item(),
                "L_ssm": l_ssm.item(),
            }
        return total

    @torch.no_grad()
    def _ddim_step(
        self,
        e_t: torch.Tensor,
        t_cur: torch.Tensor,
        t_prev: Optional[torch.Tensor],
        g: torch.Tensor,
        c: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        x0_cond = self.denoiser(e_t, t_cur, g, c)
        x0_uncond = self.denoiser.forward_uncon(e_t, t_cur)
        x0_guided = F.normalize(x0_uncond + guidance_scale * (x0_cond - x0_uncond), dim=-1)
        if t_prev is None:
            return x0_guided

        alpha_t = self.alphas_cum[t_cur].unsqueeze(-1)
        alpha_prev = self.alphas_cum[t_prev].unsqueeze(-1)
        tangent = e_t - torch.sqrt(alpha_t) * x0_guided
        tangent = tangent - (tangent * x0_guided).sum(dim=-1, keepdim=True) * x0_guided
        tangent_norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        near_zero = tangent_norm < 1e-6
        eps_direction = tangent / tangent_norm
        eps_direction = torch.where(near_zero.expand_as(eps_direction), x0_guided, eps_direction)
        e_prev = torch.sqrt(alpha_prev) * x0_guided + torch.sqrt(1.0 - alpha_prev) * eps_direction
        return F.normalize(e_prev, dim=-1)

    @torch.no_grad()
    def predict_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cls_indexes: torch.Tensor,
        col_table_ids: torch.Tensor,
        col_pos_in_table: torch.Tensor,
        numeric_features: Optional[torch.Tensor] = None,
        numeric_feature_mask: Optional[torch.Tensor] = None,
        topk: int = 1,
        fast_steps: int = 50,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        mc: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device
        global_context, column_context = self.encoder(
            input_ids,
            attention_mask,
            cls_indexes,
            numeric_features=numeric_features,
            numeric_feature_mask=numeric_feature_mask,
        )
        g = global_context[col_table_ids]
        c = column_context[col_table_ids, col_pos_in_table]
        num_columns = g.size(0)
        if num_columns == 0:
            return (
                torch.empty(0, topk, device=device, dtype=torch.long),
                torch.empty(0, topk, device=device),
            )

        fast_steps = max(1, min(self.T, int(fast_steps)))
        t_indices = torch.linspace(self.T - 1, 0, fast_steps, device=device).long()
        label_embeddings = self._norm_label_emb()
        mc = max(1, int(mc))
        scores_accumulator = None

        for sample_index in range(mc):
            step_seed = None if seed is None else int(seed) + sample_index
            if step_seed is not None:
                cuda_devices = [device.index] if device.type == "cuda" else []
                rng_context = torch.random.fork_rng(devices=cuda_devices)
                rng_context.__enter__()
                torch.manual_seed(step_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(step_seed)

            e = F.normalize(
                    torch.randn(
                        num_columns,
                        self.y_dim,
                        device=device,
                        dtype=label_embeddings.dtype,
                    ),
                    dim=-1,
                )
            for index, t_cur_value in enumerate(t_indices.tolist()):
                t_cur = torch.full((num_columns,), t_cur_value, device=device, dtype=torch.long)
                if index + 1 < len(t_indices):
                    t_prev_value = t_indices[index + 1].item()
                    t_prev = torch.full((num_columns,), t_prev_value, device=device, dtype=torch.long)
                else:
                    t_prev = None
                e = self._ddim_step(e, t_cur, t_prev, g, c, guidance_scale)

            if step_seed is not None:
                rng_context.__exit__(None, None, None)

            scores = e @ label_embeddings.t()
            scores_accumulator = scores if scores_accumulator is None else scores_accumulator + scores

        scores_average = scores_accumulator / mc
        top_values, top_indices = torch.topk(scores_average, k=topk, dim=-1)
        return top_indices, top_values


