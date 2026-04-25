import torch


def fuse_image_score(s_i: torch.Tensor, s_q: torch.Tensor, s_p: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """InCTRLv2 image-level score fusion."""
    return (1.0 - alpha) * ((s_i + s_q) / 2.0) + alpha * s_p


def fuse_pixel_maps(dasl_map: torch.Tensor, oasl_map: torch.Tensor, beta: float = 0.75) -> torch.Tensor:
    """InCTRLv2 pixel-level DASL/OASL map fusion."""
    return (1.0 - beta) * dasl_map + beta * oasl_map
