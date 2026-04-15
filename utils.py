import random
import torch
import numpy as np


def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_mse(img1, img2, mask=None):
    """Compute MSE over the full tensor or only the masked region."""
    squared_error = (img1 - img2) ** 2
    if mask is None:
        return squared_error.mean()

    mask = mask.to(device=squared_error.device, dtype=squared_error.dtype)
    weighted_error = squared_error * mask
    active = mask.sum()
    if active.item() == 0:
        return torch.zeros((), device=squared_error.device, dtype=squared_error.dtype)
    return weighted_error.sum() / active


def calculate_psnr(img1, img2, data_range, mask=None):
    """Compute PSNR with an explicit data range and optional region mask."""
    mse = calculate_mse(img1, img2, mask=mask)
    if mse.item() == 0:
        return float('inf')

    peak = torch.as_tensor(data_range, dtype=mse.dtype, device=mse.device)
    psnr = 20 * torch.log10(peak / torch.sqrt(mse))
    return psnr.item()
