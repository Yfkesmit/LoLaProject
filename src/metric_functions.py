from nltk import Tree

# Calculates the difference in depth for the trees
def tree_height_metric(premise_tree_str, hypothesis_tree_str):
    tree_premise = Tree.fromstring(premise_tree_str)
    tree_hypothesis = Tree.fromstring(hypothesis_tree_str)

    height_premise = tree_premise.height()
    height_hypothesis = tree_hypothesis.height()

    return abs(height_premise - height_hypothesis)

# Calculates the ratio between tree depth and length of string
def relative_height_metric(t, data):
    tree = Tree.fromstring(t)
    height = tree.height()
    length = len(data.split())
    relative_height = height / length

    return relative_height

# Calculates jaccard similarity between two sets
def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set(set2)))
    union = len(set(set1).union(set(set2)))
    return intersection / union

