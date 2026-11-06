import torch

def normalize(u: torch.Tensor, step_size: float = 1.0, relative: bool = True) -> torch.Tensor:
    if relative:
        return step_size * u / (torch.norm(u) + 1e-8)
    else:
        return step_size * u