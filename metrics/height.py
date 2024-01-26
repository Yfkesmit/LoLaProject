from nltk import Tree

# Calculates the difference in depth for the trees
def tree_height_metric(premise_tree_str, hypothesis_tree_str):
    tree_premise = Tree.fromstring(premise_tree_str)
    tree_hypothesis = Tree.fromstring(hypothesis_tree_str)

    height_premise = tree_premise.height()
    height_hypothesis = tree_hypothesis.height()

    return abs(height_premise - height_hypothesis)

# list for [cid, premise_tree_str, hypothesis_tree_str, tree_height_difference]
sorted_height = []

# Currently done for 5 examples, but should be on whole subset
for i in range(5):
    cid = subset["cid"][i]
    premise_tree_str = subset["p_tree"][i]
    hypothesis_tree_str = subset["h_tree"][i]

    height_difference = tree_height_metric(premise_tree_str, hypothesis_tree_str)

    sorted_height.append((cid, premise_tree_str, hypothesis_tree_str, height_difference))

# Sorting the list in ascending order based on height difference's real value number
# want a list in decreasing order? add reverse=True
sorted_height = sorted(sorted_height, key=lambda x: x[3])
# print(sorted_height)

# List of the sorted 'cid' IF WANTED
sorteid_height_cid = [cid for cid, _, _, _ in sorted_height]
# print(sorteid_height_cid)
