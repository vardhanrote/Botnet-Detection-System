import random

import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seeds so experiments are
    more reproducible.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Random seed set to {seed}")


def check_device():
    """
    Check whether GPU is available.
    """

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            "GPU available:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = torch.device("cpu")

        print(
            "GPU not available. Using CPU."
        )

    return device