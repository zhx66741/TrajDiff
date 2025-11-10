import torch

class DDBM:
    def __init__(
            self,
            beta_min=0.002,
            beta_d=1.0,
            device="cpu",
            args=None,
    ):
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.sigma_max = 1
        self.sigma_min = 0.002
        self.device = device
        self.args = args
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def get_alpha_sigma(self, t):
        # beta(t) = beta_min + beta_d * t
        beta_int = self.beta_min * t + 0.5 * self.beta_d * t ** 2
        alpha = torch.exp(-0.5 * beta_int)  # α
        sigma2 = 1 - alpha ** 2  # σ
        snr = alpha ** 2 / (sigma2 + 1e-7)
        return alpha.to(self.device), sigma2.to(self.device), snr.to(self.device)

    def bridge_mean(self, x0, xT, alpha_t, alpha_T, snr_t, snr_T):
        w = snr_T / (snr_t + 1e-7)
        mu_t = w * (alpha_t / alpha_T) * xT + (1 - w) * alpha_t * x0
        return mu_t

    def bridge_std(self, sigma2_t, snr_t, snr_T):
        w = snr_T / (snr_t + 1e-7)
        std = torch.sqrt(sigma2_t * (1 - w + 1e-7))
        return std


    def training_bridge_losses(self, model, x_start_, x_end_):
        bs = x_start_[0].shape[0]
        t = torch.rand(x_start_[0].shape[0]).to(self.device) * (self.sigma_max - self.sigma_min) + self.sigma_min

        def bridge_sample(x0, xT, t):
            alpha_t, sigma2_t, snr_t = self.get_alpha_sigma(t)
            alpha_T, sigma2_T, snr_T = self.get_alpha_sigma(torch.ones_like(t))

            alpha_t = alpha_t.view(bs, 1, 1)
            alpha_T = alpha_T.view(bs, 1, 1)
            snr_t = snr_t.view(bs, 1, 1)
            snr_T = snr_T.view(bs, 1, 1)
            sigma2_t = sigma2_t.view(bs, 1, 1)

            mu_t = self.bridge_mean(x0, xT, alpha_t, alpha_T, snr_t, snr_T)
            std_t = self.bridge_std(sigma2_t, snr_t, snr_T)
            noise = torch.randn_like(x0)
            x_t = mu_t + std_t * noise

            return x_t, mu_t, noise

        x_t, mu_t, noise = bridge_sample(x_start_[0], x_end_[0], t)
        _, x_t_grid, _ = bridge_sample(x_start_[1], x_end_[1], t)

        model_output = model(x_t.float(), x_t_grid.float())
        if self.args.target_measure == "sspd":
            mu_t = model.proj(mu_t.float())
        else:
            mu_t = model.proj(mu_t.float())[0]
        loss = self.criterion(model_output, mu_t.permute(1, 0, 2))
        return loss

