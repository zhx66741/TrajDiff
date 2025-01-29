import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLN(nn.Module):
    def __init__(self, input_dim, eps=1e-6):
        super(AdaLN, self).__init__()
        self.gamma = nn.Parameter(torch.ones(input_dim))  # 可学习的缩放参数
        self.beta = nn.Parameter(torch.zeros(input_dim))  # 可学习的偏移参数
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class SAMLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.1):
        super(SAMLayer, self).__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, embed_size)
        )
        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.norm1 = AdaLN(embed_size)
        self.norm2 = AdaLN(embed_size)
        self.norm3 = AdaLN(embed_size)

    def forward(self, q, kv):
        q_, kv_ = q, kv

        _, kv_score = self.multihead_attn(kv, kv, kv)
        kv_score = kv_score.repeat(self.num_heads, 1, 1)
        q, _ = self.multihead_attn(q, kv, kv, attn_mask=kv_score)

        _, q_score = self.multihead_attn(q_, q_, q_)
        q_score = q_score.repeat(self.num_heads, 1, 1)
        kv_, _ = self.multihead_attn(kv_, q_, q_, attn_mask=q_score)

        return q, kv_



class FP_SAMEncoder(nn.Module):
    def __init__(self,args,init_dim=2, embed_size=256, num_heads=8, ff_hidden_size=2048, nlayer=1, dropout=0.1):
        super(FP_SAMEncoder, self).__init__()
        self.args = args
        self.init_dim = init_dim
        self.embed_size = embed_size
        # self.proj = nn.Sequential(nn.Linear(init_dim, embed_size),
        #                           nn.ReLU(),
        #                           nn.Linear(embed_size, embed_size))
        self.seq_enc = nn.LSTM(input_size=init_dim,hidden_size=embed_size,num_layers=nlayer)
        atten_layer = nn.TransformerEncoderLayer(d_model=embed_size,nhead=num_heads)
        self.T = nn.TransformerEncoder(atten_layer,num_layers=nlayer)

        self.block = nn.ModuleList([SAMLayer(embed_size=embed_size, num_heads=num_heads, ff_hidden_size=ff_hidden_size, dropout=dropout) for i in range(nlayer)])
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, kv):
        q, kv = self.seq_enc(q)[0],self.seq_enc(kv)[0]
        q, kv = self.T(q),self.T(kv)

        for block in self.block:
            q, kv = block(q, kv)

        out = q + self.args.mu * kv
        out = F.adaptive_avg_pool1d(out.permute(1, 2, 0), 1).squeeze(-1)
        return F.normalize(out,dim=1)

    def ListNetLoss(self, true_order, pred_order, epsilon=1e-8):

        true_order = true_order.float()
        pred_order = pred_order.float()

        true_order = torch.clamp(true_order, min=epsilon)
        pred_order = torch.clamp(pred_order, min=epsilon)

        true_probs = F.softmax(true_order, dim=1)
        pred_probs = F.softmax(pred_order, dim=1)

        true_probs = torch.clamp(true_probs, min=epsilon)
        pred_probs = torch.clamp(pred_probs, min=epsilon)

        loss = F.kl_div(pred_probs.log(), true_probs, reduction='batchmean')
        return loss

    def listmle_loss(self,y_true, y_pred):

        prob_pred = F.softmax(y_pred, dim=1)
        prob_true = F.softmax(y_true, dim=1)

        loss = -torch.sum(prob_true * torch.log(prob_pred + 1e-10), dim=1)

        return loss.mean()

    def MSELoss(self, truth_simi, pred_simi):
        return F.mse_loss(truth_simi,pred_simi)


class MLP(nn.Module):
    def __init__(self,nin,nhidden):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(nin,nhidden),
            nn.ReLU(),
            nn.Linear(nhidden,nin)
        )
    def forward(self,x):
        return F.normalize(self.enc(x),dim=1)
