from collections import Counter
import numpy as np
from datasets import Dataset

def split_into_subsets(data, subset_size=50001):
    subsets = []
    
    # Convert lists to numpy arrays for faster indexing
    premise_array = np.array(data['premise'])
    hypothesis_array = np.array(data['hypothesis'])
    label_array = np.array(data['label'])
    p_tree_array = np.array(data['p_tree'])
    h_tree_array = np.array(data['h_tree'])
    cid_array = np.array(data['cid'])

    for i in range(0, len(premise_array), subset_size):
        print(i)
        subset = {
            'premise': premise_array[i:i+subset_size].tolist(),
            'hypothesis': hypothesis_array[i:i+subset_size].tolist(),
            'label': label_array[i:i+subset_size].tolist(),
            'p_tree': p_tree_array[i:i+subset_size].tolist(),
            'h_tree': h_tree_array[i:i+subset_size].tolist(),
            'cid': cid_array[i:i+subset_size].tolist()
        }

        # Count occurrences of each 'cid' in the subset
        cid_counter = Counter(subset['cid'])
        
        # Identify cids occurring that are dividable by three
        valid_cids = {cid for cid, count in cid_counter.items() if count % 3 == 0}
        
        # Filter subset to keep only rows with valid cids
        valid_indices = [index for index, cid in enumerate(subset['cid']) if cid in valid_cids]

        subset = {key: [subset[key][index] for index in valid_indices] for key in subset.keys()}
        
        # Append the adjusted subset
        subsets.append(Dataset.from_dict(subset))

    return subsets