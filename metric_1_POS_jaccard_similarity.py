import spacy
import pickle

with open('/content/drive/MyDrive/SNLI subset/SNLI.pkl', 'rb') as file:
    subset = pickle.load(file)

subset_list = subset.to_dict()

nlp = spacy.load('en_core_web_sm')

# Jaccard similarity function
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# list for [premise, hypothesis, label, Jaccard similarity]
prob = []

# Currently done for 5 examples, but should be on whole subset
for i in range(5):
    premise = subset_list['premise'][i]
    hypothesis = subset_list['hypothesis'][i]
    label = subset_list['label'][i]

    # Finding the POS tags for the sentences
    premise_doc = nlp(premise)
    hypothesis_doc = nlp(hypothesis)

    # Extracting the POS tags for premise and hypothesis
    premise_pos_tags = set([token.pos_ for token in premise_doc])
    hypothesis_pos_tags = set([token.pos_ for token in hypothesis_doc])

    similarity = jaccard_similarity(premise_pos_tags, hypothesis_pos_tags)

    prob.append((premise, hypothesis, label, similarity))

# Sorting the list in decreasing order based on the similarity's real value number
# want a list in ascending order? remove reverse=True
prob = sorted(prob, key=lambda x: x[3], reverse=True)
# print(sorted_tuples)

# The ordered list in ascending order based on the Jaccard similarity of POS tags
ordered_list = [(premise, hypothesis, label) for premise, hypothesis, label, _ in prob]
# print(ordered_list)
