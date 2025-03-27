import torch

class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # \bar{\alpha_t}

    def forward_process(self, x0, t):
        """
        在时间步 t 计算加噪后的样本 xt。
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise
    
    def reverse_process(self, x_t, t, model):
        """
        反向去噪过程
        """
        noise_pred = model(x_t, t)
        alpha_t = self.alpha[t][:, None, None, None]
        alpha_bar_t = self.alpha_bar[t][:, None, None, None]
        beta_t = self.beta[t][:, None, None, None]

        if t[0] == 0:
            return x_t

        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred)
        z = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        return mean + torch.sqrt(beta_t) * z
