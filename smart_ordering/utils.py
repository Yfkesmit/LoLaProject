import string
import numpy as np

from datasets import Dataset

from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

def tokenize(text):
    tokens = [word.strip(string.punctuation) for word in text.split()]
    tokens = [token for token in tokens if token]
    return tokens

def sort_on_triplets(caption_dict, reverse):
    mean_dict = {cid: np.mean([score[0] for score in scores]) for cid, scores in caption_dict.items()}

    sorted_indices = np.argsort(list(mean_dict.values()))[::-1] if reverse else np.argsort(list(mean_dict.values()))

    sorted_data = {
        'premise': [],
        'hypothesis': [],
        'label': [],
        # 'score':[],
    }

    # Sorting within each group of the same mean_tree_height_difference
    for index in tqdm(sorted_indices, desc="Sorting within each group"):
        cid = list(mean_dict.keys())[index]
        sorted_captions = sorted(caption_dict[cid], key=lambda x: x[1])  # Sorting within the list of captions for this caption ID
        sorted_data['premise'].extend([premise for _, premise, _,_, _, _ in sorted_captions])
        sorted_data['hypothesis'].extend([hypothesis for _, _, hypothesis,_, _, _ in sorted_captions])
        sorted_data['label'].extend([label for _, _, _,label, _, _ in sorted_captions])
        # sorted_data['score'].extend([caption_dict[cid]] * len(sorted_captions))

    return Dataset.from_dict(sorted_data)