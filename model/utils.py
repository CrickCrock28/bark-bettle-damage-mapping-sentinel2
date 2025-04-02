from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def build_optimizer(config, model_params):
    """Build optimizer based on configuration."""
    optimizer_type = config.training["optimizer"]["type"]
    optimizer_params = config.training["optimizer"]["params"]
    if optimizer_type == "Adam":
        return Adam(model_params, **optimizer_params)
    elif optimizer_type == "SGD":
        return SGD(model_params, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def build_scheduler(config, optimizer, train_loader_length, current_epoch):
    """Build scheduler based on configuration."""
    scheduler_type = config.training["scheduler"]["type"]
    T_max_factor = config.training["scheduler"]["T_max_factor"]
    if scheduler_type == "CosineAnnealingLR":
        T_max = int(T_max_factor * (config.training["epochs"] - current_epoch) * train_loader_length)
        return CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
