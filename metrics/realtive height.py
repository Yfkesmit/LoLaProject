import pandas as pd
from nltk import Tree
import pickle
import datasets
def relative_height(path,h_rel = True):
    df = pd.DataFrame(pd.read_pickle(path))

    for index,row in df.iterrows():
        df.loc[index,'p_rel'] = Tree.fromstring(df.loc[index]["p_tree"]).height()/len(df.loc[index]["premise"].split())
        df.loc[index,'h_rel'] = Tree.fromstring(df.loc[index]["h_tree"]).height()/len(df.loc[index]["hypothesis"].split())
    if h_rel == True:
        df.sort_values(by=["h_rel"])
    else:
        df.sort_values(by=["p_rel"])

    return datasets.Dataset.from_pandas(df)

def relative_pos_height(path):
    df = pd.DataFrame(pd.read_pickle(path))

    for index,row in df.iterrows():
        pos_height =  abs(Tree.fromstring(df.loc[index]["p_tree"]).height() - Tree.fromstring(df.loc[index]["h_tree"]).height())
        pos_length = (len(df.loc[index]["premise"].split()) - len(df.loc[index]["hypothesis"].split()))
        df.loc[index,'rel_pos_height'] = pos_height / pos_length
        df.sort_values(by=["rel_pos_height"])


    return datasets.Dataset.from_pandas(df)

# with open('relative_h_height.pickle', 'wb') as handle:
#     pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)