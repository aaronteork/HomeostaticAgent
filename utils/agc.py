import torch


# ------------------ Adaptive Gradient Clipping ------------------------------
# Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/agc.py
# Original paper and official JAX impl (paper authors): https://github.com/deepmind/deepmind-research/tree/master/nfnets
@torch.no_grad()
def agc(parameters, clip=0.3, pmin=1e-3):
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue

        gradient_norm = torch.linalg.vector_norm(gradient.detach().float())
        parameter_norm = torch.linalg.vector_norm(parameter.detach().float())

        upper = clip * parameter_norm.clamp_min(pmin)
        scale = (upper / gradient_norm.clamp_min(1e-6)).clamp_max(1.0)

        gradient.mul_(scale.to(gradient.dtype))
