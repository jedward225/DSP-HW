import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_seed_from_config(config: dict) -> Optional[int]:
    if not isinstance(config, dict):
        return None
    seed = config.get('seed', None)
    if seed is not None:
        return int(seed)
    evaluation = config.get('evaluation', {})
    if isinstance(evaluation, dict) and evaluation.get('seed', None) is not None:
        return int(evaluation['seed'])
    return None
