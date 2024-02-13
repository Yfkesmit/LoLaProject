# Curriculum Learning for NLI
This repository contains the dataset and code used in the paper wrote by Yfke Smit, Sarah Abdalla and Tomás Pires Iken for the UU Logic and Language course in Block 2, Academic Year 2023-2024. The associated paper can be found here (insert link to paper on github)

The code consists of two parts: the creation of the subset used for training, and the ordering functions that are applied on the subset before training. Associated are two notebooks that contain examples of how to work with the code provided in this repository.

Notebooks:


## Subset Creation
The subset was created in the following way:
1. The SNLI dataset was downloaded and reworked to a specific format
2. This formatted dataset was splitted into subsets using the `split_into_subsets` function from [`subset_creation.create_subset.py`](https://github.com/Yfkesmit/LoLaProject/blob/main/subset_creation/create_subset.py)
3. The best subset was chosen, based on performance of the subset with different basic orderings.

## Applying Smart Orderings
With the subset, different orderings before fine-tuning can be applied to see if this improves performance.
The SNLI dataset consists of premise-hypothesis triplets. To order without these triplets in mind, the functions from [`smart_ordering.sorting.py`](https://github.com/Yfkesmit/LoLaProject/blob/main/smart_ordering/sorting.py) can be used. To order with the triplets in mind, import the same functions from [`smart_ordering.sorting_with_triplets.py`](https://github.com/Yfkesmit/LoLaProject/blob/main/smart_ordering/sorting.py)

The following functions are available:
- `sort_on_tree_height_difference`
- `sort_on_jaccard_similarity`
- `sort_on_jaccard_similarity_pos`
- `relative_height_by_difference`
- `relative_height_by_sum`
- `no_sorting`

All functions have the same input variables: `data` and `reverse`. setting `reverse` to `True` reverses the training order.
The `data` should be a `dataset.Dataset` and contain at least the following items:
- `premise`
- `hypothesis`
- `label`
- `p_tree` (premise parsed tree)
- `h_tree` (hypothesis parsed tree)
- `cid` (captionID)
- `pid` (pairID)

