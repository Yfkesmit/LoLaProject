import numpy as np
from nltk import Tree
from datasets import Dataset
from .metric_functions import tree_height_metric, jaccard_similarity, relative_height_metric
from .utils import tokenize

from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

def sort_on_tree_height_difference(data, reverse=False):
    tree_height_differences = [tree_height_metric(p, h) for p, h in tqdm(zip(data['premise'], data['hypothesis']), desc="Sorting on tree height difference")]
    sorted_indices = np.argsort(tree_height_differences)[::-1] if reverse else np.argsort(tree_height_differences)
    sorted_data = {
        'premise': np.array(data['premise'])[sorted_indices],
        'hypothesis': np.array(data['hypothesis'])[sorted_indices],
        'label': np.array(data['label'])[sorted_indices]
        # 'tree_height_difference': np.array(tree_height_differences)[sorted_indices]
    }
    return Dataset.from_dict(sorted_data)

def sort_on_jaccard_similarity(data, reverse=False):
    premise_tokens = [tokenize(p) for p in data['premise']]
    hypothesis_tokens = [tokenize(h) for h in data['hypothesis']]
    
    jaccard_similarities = [jaccard_similarity(set1, set2) for set1, set2 in tqdm(zip(premise_tokens, hypothesis_tokens), desc="Sorting on Jaccard similarity")]
    
    sorted_indices = np.argsort(jaccard_similarities)[::-1] if reverse else np.argsort(jaccard_similarities)
    
    sorted_data = {
        'premise': np.array(data['premise'])[sorted_indices],
        'hypothesis': np.array(data['hypothesis'])[sorted_indices],
        'label': np.array(data['label'])[sorted_indices],
        # 'jaccard_similarity': np.array(jaccard_similarities)[sorted_indices],
    }
    
    return Dataset.from_dict(sorted_data)

def sort_on_jaccard_similarity_pos(data, reverse=False):
    data_list = data.to_dict()
    # Create a new column for POS tags in your dataset
    data_list['premise_pos_tags'] = [[pos for _, pos in Tree.fromstring(p).pos()] for p in data_list['p_tree']]
    data_list['hypothesis_pos_tags'] = [[pos for _, pos in Tree.fromstring(h).pos()] for h in data_list['h_tree']]    
    jaccard_similarities = [jaccard_similarity(set1, set2) for set1, set2 in tqdm(zip(data_list['premise_pos_tags'], data_list['hypothesis_pos_tags']), desc="Sorting on Jaccard similarity, pos")]
    sorted_indices = np.argsort(jaccard_similarities)[::-1] if reverse else np.argsort(jaccard_similarities)
    sorted_data = {
        'premise': np.array(data['premise'])[sorted_indices],
        'hypothesis': np.array(data['hypothesis'])[sorted_indices],
        'label': np.array(data['label'])[sorted_indices],
        # 'jaccard_similarity': np.array(jaccard_similarities)[sorted_indices],
    }
    return Dataset.from_dict(sorted_data)

def relative_height_by_difference(data, reverse=False):
    relative_heights = [
        abs(relative_height_metric(p_tree, premise) - relative_height_metric(h_tree, hypothesis))
        for p_tree, h_tree, premise, hypothesis in tqdm(zip(data['p_tree'], data['h_tree'], data['premise'], data['hypothesis']), desc="Calculating relative height by difference")]

    sorted_indices = np.argsort(relative_heights)[::-1] if reverse else np.argsort(relative_heights)
    sorted_data = {
        'premise': np.array(data['premise'])[sorted_indices],
        'hypothesis': np.array(data['hypothesis'])[sorted_indices],
        'label': np.array(data['label'])[sorted_indices],
        # 'relative_heights': np.array(relative_heights)[sorted_indices],
    }
    return Dataset.from_dict(sorted_data)

def relative_height_by_sum(data, reverse=False):
    relative_heights = [
        (relative_height_metric(p_tree, premise) + relative_height_metric(h_tree, hypothesis))
        for p_tree, h_tree, premise, hypothesis in tqdm(zip(data['p_tree'], data['h_tree'], data['premise'], data['hypothesis']), desc="Calculating relative height by sum")]

    sorted_indices = np.argsort(relative_heights)[::-1] if reverse else np.argsort(relative_heights)
    sorted_data = {
        'premise': np.array(data['premise'])[sorted_indices],
        'hypothesis': np.array(data['hypothesis'])[sorted_indices],
        'label': np.array(data['label'])[sorted_indices],
        # 'relative_heights': np.array(relative_heights)[sorted_indices],
    }
    return Dataset.from_dict(sorted_data)

def no_sorting(data, reverse=False):
    indices = slice(None, None, -1) if reverse else slice(None, None, None)
    data_dict = {
        'premise': np.array(data['premise'])[indices],
        'hypothesis': np.array(data['hypothesis'])[indices],
        'label': np.array(data['label'])[indices]
    }

    return Dataset.from_dict(data_dict)