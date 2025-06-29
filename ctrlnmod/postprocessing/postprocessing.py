import os
from datetime import datetime
import csv
import re
from typing import Dict, Any, List, Optional
from tensorboard.backend.event_processing import event_accumulator


def save_hyperparams_to_csv(
    hyperparams: Dict[str, Any],
    version: str,
    csv_path: str = 'hyperparameters.csv'
) -> None:
    """
    Save or update hyperparameters for a specific version in a CSV file.

    Args:
        hyperparams: Dictionary of hyperparameters to save.
        version: Unique identifier for the version (used as a key).
        csv_path: Path to the CSV file (default: 'hyperparameters.csv').
    """
    hyperparams = hyperparams.copy()  # avoid modifying original dict
    hyperparams['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare CSV fieldnames
    fieldnames = ['version'] + [k for k in hyperparams.keys() if k != 'version']

    rows = []
    if os.path.isfile(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Find row with matching version
    update_index = next((i for i, row in enumerate(rows) if row.get('version') == str(version)), None)

    new_row = {'version': version, **hyperparams}

    if update_index is not None:
        rows[update_index] = new_row
    else:
        rows.append(new_row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_info_from_folder_name(config_folder: str) -> Dict[str, Any]:
    """
    Extract model info and hyperparameters from a config folder name string.

    Expected format examples:
        'model_lr_0.001_epochs_50_alpha_0.5'

    Args:
        config_folder: Folder name string containing hyperparameter info.

    Returns:
        Dictionary with extracted keys and converted values.
    """
    keys_to_extract = [
        'init_type', 'lr', 'lin', 'init_lambda_logdet', 'logdet_factor', 'epochs',
        'param', 'alpha', 'lambda_lip'
    ]

    # Extract model name: everything before first underscore
    model_name_match = re.match(r'([^_]+)', config_folder)
    if not model_name_match:
        raise ValueError("Cannot determine model name from folder name.")
    model_name = model_name_match.group(1)

    info = {'model_name': model_name}

    for key in keys_to_extract:
        match = re.search(rf'{key}_([^_]+)', config_folder)
        if match:
            value_str = match.group(1)
            # Convert values according to key
            if key in {'lr', 'init_lambda_logdet', 'logdet_factor', 'alpha', 'lambda_lip'}:
                try:
                    info[key] = float(value_str)
                except ValueError:
                    info[key] = value_str  # fallback to string if conversion fails
            elif key == 'epochs':
                try:
                    info[key] = int(value_str)
                except ValueError:
                    info[key] = value_str
            elif key == 'lin':
                info['lin_req_grad'] = (value_str == 'fre')
            else:
                info[key] = value_str

    return info


def load_tensorboard_scalars(log_dir: str, tags: List[str]) -> Dict[str, List[float]]:
    """
    Load scalar time series data from TensorBoard logs for specified tags.

    Args:
        log_dir: Directory path containing TensorBoard event files.
        tags: List of scalar tags to load.

    Returns:
        Dictionary mapping tag -> list of scalar values.
        If a tag is not found, returns an empty list for that tag.
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    data = {}
    available_tags = ea.Tags().get('scalars', [])
    for tag in tags:
        if tag in available_tags:
            scalars = ea.Scalars(tag)
            data[tag] = [scalar.value for scalar in scalars]
        else:
            data[tag] = []

    return data
