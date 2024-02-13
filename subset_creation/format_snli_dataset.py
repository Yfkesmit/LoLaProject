from datasets import Dataset, DatasetDict
import pandas as pd

def format_snli_train_dataset(train_data):
    # Training set
    train_data = train_data[train_data.gold_label != "-"]
    # this line removes all opther columns besides the three used for training
    train_data = train_data.rename(
        columns={
            "sentence1": "premise",
            "sentence2": "hypothesis",
            "gold_label": "label",
            "captionID": "cid",
            "pairID": "pid",
            "sentence1_parse": "p_tree",
            "sentence2_parse": "h_tree"
        }
    )
    # this line removes all other columns besides the ones used for training
    train_data = train_data[["premise", "hypothesis", "label", "cid", "pid", "p_tree", "h_tree"]]

    train_data.loc[train_data["label"] == "neutral", "label"] = 1
    train_data.loc[train_data["label"] == "contradiction", "label"] = 2
    train_data.loc[train_data["label"] == "entailment", "label"] = 0
    train_data = Dataset.from_dict(train_data)
    return train_data

def format_snli_dataset(path):
    # Training set
    train_data = pd.read_json(f"{path}\snli_1.0_train.jsonl", lines=True)
    train_data = train_data[train_data.gold_label != "-"]
    # this line removes all opther columns besides the three used for training
    train_data = train_data[["sentence1", "sentence2", "gold_label"]]
    train_data = train_data.rename(
        columns={
            "sentence1": "premise",
            "sentence2": "hypothesis",
            "gold_label": "label",
        }
    )
    train_data.loc[train_data["label"] == "neutral", "label"] = 1
    train_data.loc[train_data["label"] == "contradiction", "label"] = 2
    train_data.loc[train_data["label"] == "entailment", "label"] = 0
    train_data = Dataset.from_dict(train_data)

    # Test set
    test_data = pd.read_json(f"{path}\snli_1.0_test.jsonl", lines=True)
    test_data = test_data[test_data.gold_label != "-"]
    # this line removes all opther columns besides the three used for training
    test_data = test_data[["sentence1", "sentence2", "gold_label"]]
    test_data = test_data.rename(
        columns={
            "sentence1": "premise",
            "sentence2": "hypothesis",
            "gold_label": "label",
        }
    )
    test_data.loc[test_data["label"] == "neutral", "label"] = 1
    test_data.loc[test_data["label"] == "contradiction", "label"] = 2
    test_data.loc[test_data["label"] == "entailment", "label"] = 0
    test_data = Dataset.from_dict(test_data)

    # Validation set
    val_data = pd.read_json(f"{path}\snli_1.0_dev.jsonl", lines=True)
    val_data = val_data[val_data.gold_label != "-"]
    # this line removes all opther columns besides the three used for training
    val_data = val_data[["sentence1", "sentence2", "gold_label"]]
    val_data = val_data.rename(
        columns={
            "sentence1": "premise",
            "sentence2": "hypothesis",
            "gold_label": "label",
        }
    )
    val_data.loc[val_data["label"] == "neutral", "label"] = 1
    val_data.loc[val_data["label"] == "contradiction", "label"] = 2
    val_data.loc[val_data["label"] == "entailment", "label"] = 0
    val_data = Dataset.from_dict(val_data)

    return DatasetDict(
        {"test": test_data, "train": train_data, "validation": val_data}
    )