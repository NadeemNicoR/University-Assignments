# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 16:32:58 2018

@author: PC_Nt
"""
import pandas as pd
import numpy as np
import xmltodict

from argparse import ArgumentParser
from collections import OrderedDict


parser = ArgumentParser()
parser.add_argument("--data", help="accepts data for training", required=True)
parser.add_argument("--output", help="path to the output xml file", required=True)
parser.add_argument("--header", help="enter True if csv contains headers")
args = parser.parse_args()

data_file = args.data

# Remove headers and first columns if optional parameter --header True
if args.header == "True":
    df = pd.read_csv(args.data)
    column_list = df.keys()
    attributes = column_list[1:-1]
    target_column = column_list[-1]
else:
    # Read csv with column names as numbers
    df = pd.read_csv(args.data, header=None)
    column_list = df.keys()
    attributes = column_list[:-1]
    rename_attributes = ["att{}".format(i) for i in range(len(attributes))]
    rename_values = {key: value for key, value in zip(attributes, rename_attributes)}

    # rename column names with att(N) i.e., att0, att1,... etc.
    df.rename(columns=rename_values, inplace=True)
    attributes = rename_attributes
    target_column = column_list[-1]

unique_class_labels = np.unique(df[target_column])
total_target_labels = len(unique_class_labels)


def entropy(target_col):
    global total_target_labels
    elements, counts = np.unique(target_col, return_counts=True)
    dataset_entropy = -np.sum([(counts[i]/float(np.sum(counts)))*np.log(counts[i]/float(np.sum(counts)))/np.log(total_target_labels)
                               for i in range(len(elements))])
    return dataset_entropy


def info_gain(data, split_attribute_name, target_name="class"):
    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])

    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    # find dataset entropy based on attributes
    weighted_entropy = np.sum([(counts[i] / float(np.sum(counts))) * entropy(
        data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])

    information_gain = total_entropy - weighted_entropy
    return [information_gain, total_entropy]


def ID3(Examples, features, target_attribute_name="class", parent_node_class=None, parent_best_feature=None, parent_value=None):
    # If all target_values have the same value
    if len(np.unique(Examples[target_attribute_name])) <= 1:
        tree = OrderedDict()
        tree["@entropy"] = 0.0
        tree["@feature"] = parent_best_feature
        tree["@value"] = parent_value
        tree["#text"] = np.unique(Examples[target_attribute_name])[0]
        return tree

    # If examples is empty
    elif len(Examples) == 0:
        tree = OrderedDict()
        tree["@entropy"] = 0.0
        tree["@feature"] = parent_best_feature
        tree["@value"] = parent_value
        tree["#text"] = np.unique(Examples[target_attribute_name])[np.argmax(np.unique(Examples[target_attribute_name], return_counts=True)[1])]
        return tree

    elif len(features) == 0:
        tree = OrderedDict()
        tree["@entropy"] = 0.0
        tree["@feature"] = parent_best_feature
        tree["@value"] = parent_value
        tree["#text"] = parent_node_class
        return tree

    else:
        # find the default value for this node
        parent_node_class = np.unique(Examples[target_attribute_name])[
            np.argmax(np.unique(Examples[target_attribute_name], return_counts=True)[1])]

        # Select the feature which best splits the dataset
        item_values = [(info_gain(Examples, feature, target_attribute_name)) for feature in
                       features]
        best_feature_index = np.argmax([item_value[0] for item_value in item_values])
        best_feature = features[best_feature_index]
        total_entropy = item_values[best_feature_index][1]

        tree = OrderedDict()

        # Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]

        # Recursively grow the tree by creating subtrees
        for value in np.unique(Examples[best_feature]):
            sub_data = Examples.where(Examples[best_feature] == value).dropna()

            tree["@entropy"] = total_entropy
            tree["@feature"] = parent_best_feature
            tree["@value"] = parent_value

            # remove the feature and value if they do not exist (root nodes)
            if tree["@feature"] is None:
                tree.pop("@feature")

            if tree["@value"] is None:
                tree.pop("@value")

            subtree = ID3(sub_data, features, target_attribute_name, parent_node_class, best_feature, value)
            if "node" in tree:
                if type(tree["node"]) is OrderedDict:
                    tree["node"] = [tree["node"], subtree]
                else:
                    tree["node"].append(subtree)
            else:
                tree["node"] = subtree
        return tree


tree = ID3(df, attributes, target_column)

wrap = {'tree': tree}

with open(args.output, "w") as out_file:
    out_file.write(xmltodict.unparse(wrap, full_document=True))
