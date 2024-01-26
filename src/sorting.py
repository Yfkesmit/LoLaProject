import spacy
import numpy as np
from datasets import Dataset
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

# Jaccard similarity function
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def sort_on_pos_jaccard_similarity(data):
    # list for [premise, hypothesis, label, Jaccard similarity]
    prob = []

    for i in tqdm(range(len(data['premise'])), desc="Sorting on POS Jaccard Similarity"):
        premise = data['premise'][i]
        hypothesis = data['hypothesis'][i]
        label = data['label'][i]

        # Finding the POS tags for the sentences
        premise_doc = nlp(premise)
        hypothesis_doc = nlp(hypothesis)

        # Extracting the POS tags for premise and hypothesis
        premise_pos_tags = set([token.pos_ for token in premise_doc])
        hypothesis_pos_tags = set([token.pos_ for token in hypothesis_doc])

        similarity = jaccard_similarity(premise_pos_tags, hypothesis_pos_tags)

        prob.append((premise, hypothesis, label, similarity))

    # Sorting the list in decreasing order based on the similarity's real value number
    prob = sorted(prob, key=lambda x: x[3], reverse=True)

    # The ordered list in ascending order based on the Jaccard similarity of POS tags
    ordered_list = [(premise, hypothesis, label) for premise, hypothesis, label, _ in prob]

    sorted_data = {
        'premise': np.array([item[0] for item in ordered_list]),
        'hypothesis': np.array([item[1] for item in ordered_list]),
        'label': np.array([item[2] for item in ordered_list]),
    }

    return Dataset.from_dict(sorted_data)