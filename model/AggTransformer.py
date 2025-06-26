import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from TrajDIT_bridge.utils.tools import mean_pooling


class AggAttn(nn.Module):
    def __init__(self, d_model, num_heads, depth):
        super(AggAttn, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # depth means current layer index
        self.lambda_init = self.lambda_init_fn(self.depth)
        self.lambda_q = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_v = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))


    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def forward(self, q, k, v, attention_mask=None, src_key_padding_mask=None):
        B, L_q, _ = q.size()
        _, L_k, _ = k.size()

        # Linear projection
        q = self.q_proj(q)  # [B, L_q, D]
        k = self.k_proj(k)  # [B, L_k, D]
        v = self.v_proj(v)  # [B, L_k, D]

        # Split into heads
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, D_h]
        k = k.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_k, D_h]
        v = v.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_k, D_h]

        lambda_self = torch.exp(torch.sum(self.lambda_q * self.lambda_k, dim=-1).float()).type_as(q)
        lambda_cross = torch.exp(torch.sum(self.lambda_k * self.lambda_v, dim=-1).float()).type_as(q)

        # Scaled Dot-Product Attention
        scores_self = (k @ v.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L_q, L_k]
        scores_cross = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L_q, L_k]

        scores = scores_cross * lambda_cross + scores_self * lambda_self
        # scores = scores_cross + scores_self

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            scores += attention_mask

        if src_key_padding_mask is not None:
            mask = (src_key_padding_mask.unsqueeze(1) & src_key_padding_mask.unsqueeze(2)).unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax
        attn_weights = torch.nan_to_num(scores, nan=0.0)  # 替换 NaN 为 0
        attn_weights = F.softmax(attn_weights, dim=-1)  # [B, H, L_q, L_k]
        attn_output = attn_weights @ v  # [B, H, L_q, D_h]

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)  # [B, L_q, D]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class AggAttnLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, depth):
        super(AggAttnLayer, self).__init__()
        self.aggattn = AggAttn(d_model, num_heads, depth)
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(0.5)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, q, kv, attention_mask=None, src_key_padding_mask=None):
        q_, kv_ = q, kv
        q, _ = self.aggattn(q, kv_, kv_, attention_mask, src_key_padding_mask)
        kv, _ = self.aggattn(kv, q_, q_, attention_mask, src_key_padding_mask)

        # Feed-forward block
        q1 = self.linear2(self.dropout(F.relu(self.linear1(q))))
        kv1 = self.linear2(self.dropout(F.relu(self.linear1(kv))))
        q = self.norm1(q + self.dropout(q1))
        kv = self.norm1(kv + self.dropout(kv1))
        return q, kv


class AggAttnEncoder(nn.Module):
    def __init__(self, args,  d_model, num_heads, num_layers, hidden_dim):
        super(AggAttnEncoder, self).__init__()
        self.args = args

        self.proj = nn.Linear(2, d_model)

        self.seq_enc = nn.LSTM(input_size=2, hidden_size=d_model, num_layers=num_layers, batch_first=True)

        self.aggattnlayer = nn.ModuleList([AggAttnLayer(d_model, num_heads, hidden_dim, depth + 1) for depth in range(num_layers)])


    def forward(self, q, kv, attention_mask=None, src_key_padding_mask=None):
        if self.args.measure == "discret_frechet":
            q, kv = self.seq_enc(q)[0], self.seq_enc(kv)[0]
        else:
            q, kv = self.proj(q), self.proj(kv)


        for layer in self.aggattnlayer:
            q, kv = layer(q, kv, attention_mask, src_key_padding_mask)

        out = self.args.epsilon * q + (1 - self.args.epsilon) * kv

        if src_key_padding_mask != None:
            out = mean_pooling(out, src_key_padding_mask)
        else:
            out = F.adaptive_avg_pool1d(out.permute(0, 2, 1), 1).squeeze(-1)

        return F.normalize(out, p=1, dim=-1)


    def ListNetLoss(self, y_true, y_pred):

        prob_pred = F.softmax(y_pred, dim=1)
        prob_true = F.softmax(y_true, dim=1)

        loss = -torch.sum(prob_true * torch.log(prob_pred + 1e-10), dim=1)

        return loss.mean()

    def MSELoss(self, truth_simi, pred_simi):
        return F.mse_loss(truth_simi, pred_simi)

    def rd_listnet_loss(self, y_true, y_pred, epsilon=1e-8):
        """
        Rank-Decay ListNet loss: smooth top-aware variant.
        """
        true_probs = F.softmax(y_true, dim=1)
        pred_probs = F.softmax(y_pred, dim=1)

        B, K = y_true.shape
        sorted_idx = torch.argsort(y_true, dim=1, descending=True)

        weights = torch.ones_like(true_probs)
        decay = 1.0 / torch.log2(torch.arange(2, K + 2, device=y_true.device).float())  # shape (K,)

        for b in range(B):
            weights[b, sorted_idx[b]] = decay

        loss = -torch.sum(weights * true_probs * torch.log(pred_probs + epsilon), dim=1)
        return loss.mean()
