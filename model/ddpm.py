import torch
import torch.nn as nn
import torch.nn.functional as F

# Diffusion model
class DDPM(nn.Module):
    def __init__(self, denoiser, timesteps=1000,device='cpu'):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)  # Linear beta schedule
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(device)

    def forward(self, traj_p,traj_xy):
        """
        Forward diffusion process: Add noise to x_0 step by step.
        x_0: Tensor of shape (batch_size, seq_len, 2)
        """
        seq_len, batch_size, feature_dim = traj_p.size()
        t = torch.randint(0, self.timesteps, (batch_size,), device=traj_p.device).long()
        noise = torch.randn_like(traj_p)

        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1).unsqueeze(0)
        x_t = torch.sqrt(alpha_bar_t) * traj_p + torch.sqrt(1 - alpha_bar_t) * noise

        traj_p = self.denoiser.seq_enc(traj_p)[0]
        traj_p = self.denoiser.T(traj_p)
        pred_truth = self.denoiser(x_t,traj_xy)

        return F.mse_loss(pred_truth,traj_p)

