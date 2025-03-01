import torch
import numpy as np
import random


def set_global_seed(seed: int) -> None:
    """
    Set global seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_generator(seed: int) -> torch.Generator:
    """Для Dataloader."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id) -> None:
    """Для каждого worker в Dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
