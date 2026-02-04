"""
Other utility functions for visualization and data handling
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from PIL import Image


def show_tensor_image(image):
    """
    Display a tensor as an image
    
    Args:
        image: Tensor image (C, H, W) or (H, W)
    """
    if len(image.shape) == 4:
        image = image[0]  # Take first image from batch
    
    if len(image.shape) == 3:
        if image.shape[0] == 1:  # Grayscale
            image = image.squeeze()
        else:  # RGB
            image = image.permute(1, 2, 0)
    
    # Convert to numpy and denormalize
    image = image.cpu().detach().numpy()
    
    # Clip values to valid range
    image = np.clip(image, 0, 1)
    
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')


def show_images(images, title="", num_images=10):
    """
    Display a grid of images
    
    Args:
        images: Tensor of images (B, C, H, W)
        title: Title for the plot
        num_images: Number of images to display
    """
    if len(images) > num_images:
        images = images[:num_images]
    
    fig, axes = plt.subplots(1, len(images), figsize=(15, 2))
    
    if len(images) == 1:
        axes = [axes]
    
    for idx, image in enumerate(images):
        if len(image.shape) == 4:
            image = image[0]
        
        if image.shape[0] == 1:  # Grayscale
            image = image.squeeze()
            axes[idx].imshow(image.cpu().detach().numpy(), cmap='gray')
        else:  # RGB
            image = image.permute(1, 2, 0)
            axes[idx].imshow(image.cpu().detach().numpy())
        
        axes[idx].axis('off')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()


def to_image(tensor):
    """
    Convert tensor to PIL Image
    
    Args:
        tensor: Image tensor
    
    Returns:
        PIL.Image
    """
    # Handle batch dimension
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy
    image = tensor.cpu().detach().numpy()
    
    # Transpose if needed (C, H, W) -> (H, W, C)
    if image.shape[0] in [1, 3]:
        if image.shape[0] == 1:
            image = image.squeeze(0)
        else:
            image = np.transpose(image, (1, 2, 0))
    
    # Denormalize and convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    if len(image.shape) == 2:  # Grayscale
        return Image.fromarray(image, mode='L')
    else:  # RGB
        return Image.fromarray(image, mode='RGB')


def save_images(images, path, nrow=8):
    """
    Save a grid of images
    
    Args:
        images: Tensor of images (B, C, H, W)
        path: Path to save the image
        nrow: Number of images per row
    """
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(0, 1))
    img = to_image(grid)
    img.save(path)
    print(f"Saved images to {path}")


def plot_loss(losses, title="Training Loss"):
    """
    Plot training loss curve
    
    Args:
        losses: List of loss values
        title: Plot title
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.show()


def visualize_diffusion_process(images, timesteps, nrows=2, ncols=5):
    """
    Visualize the diffusion process at different timesteps
    
    Args:
        images: List of images at different timesteps
        timesteps: List of corresponding timesteps
        nrows: Number of rows in the grid
        ncols: Number of columns in the grid
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx, (img, t) in enumerate(zip(images, timesteps)):
        if idx >= len(axes):
            break
        
        if len(img.shape) == 4:
            img = img[0]
        
        if img.shape[0] == 1:
            img = img.squeeze()
            axes[idx].imshow(img.cpu().detach().numpy(), cmap='gray')
        else:
            img = img.permute(1, 2, 0)
            axes[idx].imshow(img.cpu().detach().numpy())
        
        axes[idx].set_title(f't={t}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def denormalize(tensor, mean=0.5, std=0.5):
    """
    Denormalize a tensor
    
    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized tensor
    """
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
    
    return tensor * std + mean


def normalize(tensor, mean=0.5, std=0.5):
    """
    Normalize a tensor
    
    Args:
        tensor: Tensor to normalize
        mean: Target mean
        std: Target std
    
    Returns:
        Normalized tensor
    """
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
    
    return (tensor - mean) / std


def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU)
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    """
    Exponential Moving Average for model parameters
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
