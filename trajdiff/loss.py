from torch.nn import Module
from torch import nn
import torch
import torch.nn.functional as F

class Loss(Module):
    def __init__(self, measure="euc", gamma1=0.01, gamma2=0.01, gamma3=0.01):
        super(Loss, self).__init__()
        self.measure = measure
        self.mse = nn.MSELoss()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3

    def loss_compute(self, traj_emb, truth_simi):
        if self.measure == "euc":
            pred_simi_ = torch.exp(-torch.cdist(traj_emb, traj_emb, 1))
            pred_simi = pred_simi_
            pred_simi_ = pred_simi_[torch.triu(torch.ones(pred_simi_.shape), diagonal=1) == 1]
            truth_simi_ = torch.tensor(truth_simi[torch.triu(torch.ones(truth_simi.shape), diagonal=1) == 1])
            mse_loss = self.mse(pred_simi_, truth_simi_)
            listnet_loss = self.ListNetLoss(truth_simi, pred_simi)
            rd_listnet_loss = self.rd_listnet_loss(truth_simi, pred_simi)
            return self.gamma3 * mse_loss + self.gamma2 * listnet_loss + self.gamma1 * rd_listnet_loss

        elif self.measure == "chebyshev" or self.measure == "cheb":
            pred_simi_ = torch.exp(-self.chebyshev_distance_matrix(traj_emb))
            pred_simi = pred_simi_
            pred_simi_ = pred_simi_[torch.triu(torch.ones(pred_simi_.shape), diagonal=1) == 1]
            truth_simi_ = torch.tensor(truth_simi[torch.triu(torch.ones(truth_simi.shape), diagonal=1) == 1])
            mse_loss = self.mse(pred_simi_, truth_simi_)
            listnet_loss = self.ListNetLoss(truth_simi, pred_simi)
            rd_listnet_loss = self.rd_listnet_loss(truth_simi, pred_simi)
            return self.gamma3 * mse_loss + self.gamma2 * listnet_loss + self.gamma1 * rd_listnet_loss
        else:
            raise ValueError("measure must be euc or chebyshev")



    def chebyshev_distance(self, v1, v2):
        diffs = torch.abs(v1 - v2)
        max_diffs = torch.max(diffs, dim=-1).values
        return max_diffs

    def chebyshev_distance_matrix(self, X):
        # expand the dimension of X to shape (batch_size, 1, hidden_dim)
        X_expanded = X.unsqueeze(1)
        # expand the dimension of X to shape (1, batch_size, hidden_dim)
        X_tiled = X.unsqueeze(0)
        # calculate the absolute difference between X_expanded and X_tiled along the hidden_dim dimension
        diffs = torch.abs(X_expanded - X_tiled)
        # calculate the Chebyshev distance matrix on hidden_dim dimension
        distance_matrix = torch.max(diffs, dim=2).values
        return distance_matrix

    def ListNetLoss(self, y_true, y_pred):

        prob_pred = F.softmax(y_pred, dim=1)
        prob_true = F.softmax(y_true, dim=1)

        loss = -torch.sum(prob_true * torch.log(prob_pred + 1e-10), dim=1)

        return loss.mean()


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
