//Denoising Diffusion Probabilistic Models helper functions

import torch
import math


def get_beta_schedule(schedule_type, timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Generate beta schedule for diffusion process
    
    Args:
        schedule_type: Type of schedule ('linear', 'quadratic', 'cosine')
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        torch.Tensor: Beta schedule
    """
    if schedule_type == 'linear':
        return torch.linspace(beta_start, beta_end, timesteps)
    
    elif schedule_type == 'quadratic':
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    
    elif schedule_type == 'cosine':
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def extract(a, t, x_shape):
    """
    Extract coefficients at specified timesteps and reshape to [batch_size, 1, 1, 1]
    
    Args:
        a: Coefficient tensor
        t: Timestep indices
        x_shape: Shape of the data tensor
    
    Returns:
        torch.Tensor: Extracted and reshaped coefficients
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionUtils:
    """
    Utility class for diffusion operations
    """
    
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02, device='cuda'):
        """
        Initialize diffusion utilities
        
        Args:
            timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            device: Device to use ('cuda' or 'cpu')
        """
        self.timesteps = timesteps
        self.device = device
        
        # Beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        # Alpha values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]).to(device), 
            self.alphas_cumprod[:-1]
        ])
        
        # Square root values for reparameterization
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process (add noise)
        
        Args:
            x_start: Original images
            t: Timesteps
            noise: Optional noise tensor
        
        Returns:
            Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x, t, t_index):
        """
        Single reverse diffusion step
        
        Args:
            model: Denoising model
            x: Noisy image at timestep t
            t: Current timestep
            t_index: Index of current timestep
        
        Returns:
            Denoised image at timestep t-1
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """
        Full reverse diffusion process (generate images)
        
        Args:
            model: Denoising model
            shape: Shape of images to generate
        
        Returns:
            Generated images
        """
        device = next(model.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(
                model, 
                img, 
                torch.full((b,), i, device=device, dtype=torch.long),
                i
            )
            imgs.append(img.cpu())
        
        return imgs
    
    def get_loss(self, model, x_start, t, noise=None):
        """
        Calculate diffusion loss
        
        Args:
            model: Denoising model
            x_start: Original images
            t: Timesteps
            noise: Optional noise tensor
        
        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t)
        
        loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        return loss


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Linear beta schedule
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
