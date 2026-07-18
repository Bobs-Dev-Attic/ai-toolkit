import gc
import os

import torch


def value_map(inputs, min_in, max_in, min_out, max_out):
    return (inputs - min_in) * (max_out - min_out) / (max_in - min_in) + min_out


def flush(garbage_collect=True):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # if is mps, also clear the mps cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if garbage_collect:
        gc.collect()


def get_vram_info(device=None):
    """Return (free_gb, total_gb, used_gb) for a CUDA device, or None if no CUDA.

    Uses torch.cuda.mem_get_info, which reports device-wide memory (including
    memory held by other processes), so it reflects what is actually available
    for training rather than just this process's allocations.
    """
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    gb = 1024 ** 3
    return free / gb, total / gb, (total - free) / gb


def vram_status_string(label="", device=None):
    """Format a human-readable VRAM usage/availability string for notifications."""
    info = get_vram_info(device)
    prefix = f"{label} - " if label else ""
    if info is None:
        return f"{prefix}VRAM: no CUDA device available"
    free_gb, total_gb, used_gb = info
    return (
        f"{prefix}VRAM: {used_gb:.2f} GB used / {total_gb:.2f} GB total "
        f"({free_gb:.2f} GB free)"
    )


def get_mean_std(tensor):
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    elif len(tensor.shape) != 4:
        raise Exception("Expected tensor of shape (batch_size, channels, width, height)")
    mean, variance = torch.mean(
        tensor, dim=[2, 3], keepdim=True
    ), torch.var(
        tensor, dim=[2, 3],
        keepdim=True
    )
    std = torch.sqrt(variance + 1e-5)
    return mean, std


def adain(content_features, style_features):
    # Assumes that the content and style features are of shape (batch_size, channels, width, height)

    dims = [2, 3]
    if len(content_features.shape) == 3:
        # content_features = content_features.unsqueeze(0)
        # style_features = style_features.unsqueeze(0)
        dims = [1]

    # Step 1: Calculate mean and variance of content features
    content_mean, content_var = torch.mean(content_features, dim=dims, keepdim=True), torch.var(content_features,
                                                                                                  dim=dims,
                                                                                                  keepdim=True)
    # Step 2: Calculate mean and variance of style features
    style_mean, style_var = torch.mean(style_features, dim=dims, keepdim=True), torch.var(style_features, dim=dims,
                                                                                            keepdim=True)

    # Step 3: Normalize content features
    content_std = torch.sqrt(content_var + 1e-5)
    normalized_content = (content_features - content_mean) / content_std

    # Step 4: Scale and shift normalized content with style's statistics
    style_std = torch.sqrt(style_var + 1e-5)
    stylized_content = normalized_content * style_std + style_mean

    return stylized_content

def get_quick_signature_string(file_path):
    try:
        file_stats = os.stat(file_path)
        # Combine size and mtime into a single string
        return f"{file_stats.st_size}:{int(file_stats.st_mtime)}"
    except Exception as e:
        print(f"Error accessing file {file_path}: {e}")
        return None