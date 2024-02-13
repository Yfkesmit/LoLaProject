import numpy as np

from nltk import Tree
from datasets import Dataset
from collections import defaultdict

from .metric_functions import tree_height_metric, jaccard_similarity, relative_height_metric
from .utils import tokenize, sort_on_triplets

from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

def sort_on_tree_height_difference_mean(data, reverse=False):
    caption_ids = defaultdict(list)
    for premise, hypothesis, label, cid, pid, p_tree, h_tree in tqdm(zip(data['premise'], data['hypothesis'],data['label'], data['cid'], data['pid'],data['p_tree'], data['h_tree']), desc="Collecting tree height differences per captionID"):
        diff = tree_height_metric(p_tree, h_tree)
        caption_ids[cid].append((diff, premise, hypothesis, label, cid, pid))

    res = sort_on_triplets(caption_ids, reverse)
    return res


def sort_on_jaccard_similarity_mean(data, reverse=False):
    premise_tokens = [tokenize(p) for p in data['premise']]
    hypothesis_tokens = [tokenize(h) for h in data['hypothesis']]
    
    caption_ids = defaultdict(list)
    for premise, hypothesis, label, cid, pid, p_tokens, h_tokens in tqdm(zip(data['premise'], data['hypothesis'],data['label'], data['cid'], data['pid'], premise_tokens, hypothesis_tokens), desc="Collecting jaccard similarity per captionID"):
        sim = jaccard_similarity(p_tokens, h_tokens)
        caption_ids[cid].append((sim, premise, hypothesis, label, cid, pid))

    res = sort_on_triplets(caption_ids, reverse)
    return res

def sort_on_jaccard_similarity_pos_mean(data, reverse=False):
    data = data.to_dict()
    # Create a new column for POS tags in your dataset
    data['p_pos'] = [[pos for _, pos in Tree.fromstring(p).pos()] for p in data['p_tree']]
    data['h_pos'] = [[pos for _, pos in Tree.fromstring(h).pos()] for h in data['h_tree']]

    caption_ids = defaultdict(list)
    for p_pos, h_pos, p, h, label, cid, pid in tqdm(zip(data['p_pos'], data['h_pos'], data['premise'], data['hypothesis'],data['label'], data['cid'], data['pid']), desc="Collecting jaccard similarity POS per captionID"):
        sim = jaccard_similarity(p_pos, h_pos)
        caption_ids[cid].append((sim, p, h, label, cid, pid))

    res = sort_on_triplets(caption_ids, reverse)
    return res


def relative_height_by_difference_mean(data, reverse=False):
    caption_ids = defaultdict(list)
    for p_tree, h_tree, p, h, label, cid, pid in tqdm(zip(data['p_tree'], data['h_tree'], data['premise'], data['hypothesis'], data['label'], data['cid'], data['pid']), desc="Calculating relative height by difference"):
        relative_heights = abs(relative_height_metric(p_tree, p) - relative_height_metric(h_tree, h))        
        caption_ids[cid].append((relative_heights, p, h, label, cid, pid))

    res = sort_on_triplets(caption_ids, reverse)
    return res


def relative_height_by_sum_mean(data, reverse=False):
    caption_ids = defaultdict(list)
    for p_tree, h_tree, p, h, label, cid, pid in tqdm(zip(data['p_tree'], data['h_tree'], data['premise'], data['hypothesis'], data['label'], data['cid'], data['pid']), desc="Calculating relative height by sum"):
        relative_heights = (relative_height_metric(p_tree, p) + relative_height_metric(h_tree, h))        
        caption_ids[cid].append((relative_heights, p, h, label, cid, pid))

    res = sort_on_triplets(caption_ids, reverse)
    return res

def no_sorting(data, reverse=False):
    indices = slice(None, None, -1) if reverse else slice(None, None, None)
    data_dict = {
        'premise': np.array(data['premise'])[indices],
        'hypothesis': np.array(data['hypothesis'])[indices],
        'label': np.array(data['label'])[indices]
    }

    return Dataset.from_dict(data_dict)