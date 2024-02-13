from collections import Counter
from datasets import Dataset

def split_into_subsets(data, subset_size=50001):
    subsets = []
    
    # Convert lists to numpy arrays for faster indexing
    premise_array = data['premise']
    hypothesis_array = data['hypothesis']
    label_array = data['label']
    p_tree_array = data['p_tree']
    h_tree_array = data['h_tree']
    cid_array = data['cid']
    pid_array = data['pid']

    for i in range(0, len(premise_array), subset_size):
        subset = {
            'premise': premise_array[i:i+subset_size],
            'hypothesis': hypothesis_array[i:i+subset_size],
            'label': label_array[i:i+subset_size],
            'p_tree': p_tree_array[i:i+subset_size],
            'h_tree': h_tree_array[i:i+subset_size],
            'cid': cid_array[i:i+subset_size],
            'pid': pid_array[i:i+subset_size]
        }

        # Count occurrences of each 'cid' in the subset
        cid_counter = Counter(subset['cid'])
        
        # Identify cids occurring at least 3 times
        valid_cids = {cid for cid, count in cid_counter.items() if count % 3 == 0}
        
        # Filter subset to keep only rows with valid cids
        valid_indices = [index for index, cid in enumerate(subset['cid']) if cid in valid_cids]

        subset = {key: [subset[key][index] for index in valid_indices] for key in subset.keys()}
        
        subsets.append(Dataset.from_dict(subset))

    return subsets
