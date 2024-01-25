from datasets import Dataset

def split_into_subsets(data, subset_size=50001):
    subsets = []
    for i in range(0, (len(data['premise'])+1), subset_size):
        print(i)
        subset_premise = data['premise'][i:i+subset_size]
        subset_hypothesis = data['hypothesis'][i:i+subset_size]
        subset_label = data['label'][i:i+subset_size]
        index = 0
        while index < len(subset_premise) - 2:
            if subset_premise[index] == subset_premise[index + 1] == subset_premise[index + 2]:
                index += 3  # Move to the next set of three items
            else:
                # Delete the first two items
                del subset_premise[index:index + 2]
                del subset_hypothesis[index:index + 2]  # Delete the first two items
                del subset_label[index:index + 2]  # Delete the first two items
        # Handle end of subset
        if not (subset_premise[-1] == subset_premise[-2] == subset_premise[-3]):
            if (subset_premise[-1] == subset_premise[-2]):
                del subset_premise[-2:]
                del subset_hypothesis[-2:]
                del subset_label[-2:]
            else:
                del subset_premise[-1]
                del subset_hypothesis[-1]
                del subset_label[-1]

        # Append the adjusted subset
        subset = {
            'premise': subset_premise,
            'hypothesis': subset_hypothesis,
            'label': subset_label
        }
        subsets.append(Dataset.from_dict(subset))

    return subsets