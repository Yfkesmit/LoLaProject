import numpy as np
from datasets import Dataset
from tqdm import tqdm

def sort_on_premise_length(data):
    sorted_indices = np.argsort([len(p) for p in tqdm(data['premise'], desc="Sorting on premise length")])
    sorted_data = {
        'premise': np.array(data['premise'])[sorted_indices],
        'hypothesis': np.array(data['hypothesis'])[sorted_indices],
        'label': np.array(data['label'])[sorted_indices],
    }
    return Dataset.from_dict(sorted_data)

def sort_on_hypothesis_length(data):
    sorted_indices = np.argsort([len(h) for h in tqdm(data['hypothesis'], desc="Sorting on hypothesis length")])
    sorted_data = {
        'premise': np.array(data['premise'])[sorted_indices],
        'hypothesis': np.array(data['hypothesis'])[sorted_indices],
        'label': np.array(data['label'])[sorted_indices],
    }
    return Dataset.from_dict(sorted_data)

def sort_on_both_length(data):
    both_lengths = [len(p) + len(h) for p, h in tqdm(zip(data['premise'], data['hypothesis']), desc="Sorting on both lengths")]
    sorted_indices = np.argsort(both_lengths)
    sorted_data = {
        'premise': np.array(data['premise'])[sorted_indices],
        'hypothesis': np.array(data['hypothesis'])[sorted_indices],
        'label': np.array(data['label'])[sorted_indices],
    }
    return Dataset.from_dict(sorted_data)