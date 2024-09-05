import os
from datetime import datetime
import csv
from tensorboard.backend.event_processing import event_accumulator
import re
import pandas as pd


def save_hyperparams_to_csv(hyperparams, version, csv_path='hyperparameters.csv'):
    '''
        hyperparms : a dictionary
        version : a unique identifyer that writes on the corresponding row
    '''
    hyperparams['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fieldnames = ['version'] + [key for key in hyperparams.keys() if key != 'version']
    
    rows = []
    if os.path.isfile(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    
    update_index = next((i for i, row in enumerate(rows) if row['version'] == str(version)), None)
    
    if update_index is not None:
        rows[update_index] = {'version': version, **hyperparams}
    else:
        rows.append({'version': version, **hyperparams})
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extraire_infos_dossier(config_folder):
    # Liste de toutes les clés possibles
    all_keys = [
        'init_type', 'lr', 'lin', 'init_lambda_logdet', 'logdet_factor', 'epochs',
        'param', 'alpha', 'lambda_lip'
    ]

    # Extraire le nom du modèle (tout ce qui précède le premier underscore)
    model_name_match = re.match(r'([^_]+)', config_folder)
    if not model_name_match:
        raise ValueError("Impossible de déterminer le nom du modèle")
    
    model_name = model_name_match.group(1)
    info = {'model_name': model_name}

    # Chercher chaque clé dans le nom du dossier
    for key in all_keys:
        match = re.search(rf'{key}_([^_]+)', config_folder)
        if match:
            value = match.group(1)
            # Convertir la valeur si nécessaire
            if key in ['lr', 'init_lambda_logdet', 'logdet_factor', 'alpha', 'lambda_lip']:
                info[key] = float(value)
            elif key == 'epochs':
                info[key] = int(value)
            elif key == 'lin':
                info['lin_req_grad'] = (value == 'fre')
            else:
                info[key] = value

    return info

def load_tensorboard_data(log_dir, tags):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    data = {}
    for tag in tags:
        if tag in ea.Tags()['scalars']:
            scalars = ea.Scalars(tag)
            data[tag] = [x.value for x in scalars]
        else:
            data[tag] = []
    return data