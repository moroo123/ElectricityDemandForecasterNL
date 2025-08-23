import torch
import numpy as np
import itertools
import copy
import datetime
import os


def expand_config(cfg, sections=("model", "train")):
    """Expand the configuration by creating all combinations of specified sections.

    Args:
        cfg (dict): The original configuration dictionary.
        sections (tuple, optional): The sections to expand. Defaults to ("model", "train").

    Returns:
        list: A list of expanded configuration dictionaries.
    """
    # Build per-section variants (cartesian product of list/non-list fields within each section)
    per_section_variants = []
    for sec in sections:
        sec_dict = cfg.get(sec, {})
        # For each key, create a list of (key, value) options
        keyed_options = [
            [(k, v_i) for v_i in v] if isinstance(v, list) else [(k, v)]
            for k, v in sec_dict.items()
        ]
        # All combinations for this section
        variants = [{k: v for k, v in combo}
                    for combo in itertools.product(*keyed_options)]
        per_section_variants.append((sec, variants))

    # Cross-product across the chosen sections
    expanded = []
    for section_combo in itertools.product(*(v for _, v in per_section_variants)):
        new_cfg = copy.deepcopy(cfg)
        for (sec_name, _), sec_value in zip(per_section_variants, section_combo):
            new_cfg[sec_name] = sec_value
        expanded.append(new_cfg)
    return expanded


def set_seed(seed=42):
    """Set the random seed for reproducibility.

    Args:
        seed (int, optional): The seed value. Defaults to 42.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_run_dir(base_dir='runs'):
    """Create a new directory for a training run.

    Args:
        base_dir (str, optional): The base directory for run folders. Defaults to 'runs'.

    Returns:
        str: The path to the created run directory.
    """
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join(base_dir, ts)
    os.makedirs(path, exist_ok=True)
    return path
