import sys

import numpy as np

from utils import feature_names


def mRMR_applied_datasets(X_train, X_test, dataset_name):
    X_train_fs = X_train[:, top_5_mRMR_features[dataset_name]]
    X_test_fs = X_test[:, top_5_mRMR_features[dataset_name]]
    return X_train_fs, X_test_fs


top_5_mRMR_features = {"PTC_MR": [14, 2, 15, 1, 0],
                       "PTC_MM": [14, 2, 15, 1, 0],
                       "PTC_FM": [14, 2, 16, 15, 8],
                       "PTC_FR": [14, 0, 1, 2, 15],
                       "MUTAG": [8, 0, 10, 5, 9],
                       "Mutagenicity": [0, 13, 16, 15, 14],
                       "ER_MD": [5, 15, 12, 0, 0],
                       "DHFR_MD": [12, 15, 9, 5, 0]
                       }

if __name__ == "__main__":
    feature_dict = feature_names

    # reverse the dictionary for easy lookup
    reverse_feature_dict = {v: k for k, v in feature_dict.items()}

    # data provided for each dataset (replaced mod_zag with mod_zagreb, balban with balaban, n_imp with n_impurity,
    # label_ent with label_entropy and edge_str with edge_strength)
    data = {
        "PTC_MR": ["n_impurity", "narumi", "label_entropy", "estrada", "balaban"],
        "PTC_MM": ["n_impurity", "narumi", "label_entropy", "estrada", "balaban"],
        "PTC_FM": ["n_impurity", "narumi", "edge_strength", "label_entropy", "zagreb"],
        "PTC_FR": ["n_impurity", "balaban", "estrada", "narumi", "label_entropy"],
        "MUTAG": ["zagreb", "balaban", "edges", "randic", "nodes"],
        "Mutagenicity": ["balaban", "hyp_wiener", "edge_strength", "label_entropy", "n_impurity"],
        "ER_MD": ["randic", "label_entropy", "mod_zagreb", "balaban", "balaban"],
        "DHFR_MD": ["mod_zagreb", "label_entropy", "nodes", "randic", "balaban"]
    }

    # generate indices list for each dataset
    for dataset, features in data.items():
        indices = [reverse_feature_dict[feature] for feature in features]
        print(f"\"{dataset}\": {indices},")


def get_best_features():
    """this can be used to pass the saved features and get back the indexes of the features"""
    scores = [0] * 17
    print(scores)
    list = [[14, 2, 15, 1],
            [14, 2, 15, 1],
            [14, 2, 16, 15],
            [14, 0, 2, 15],
            [8, 0, 10, 5],
            [0, 13, 16, 15],
            [5, 15, 12, 9],
            [12, 15, 9, 5]]

    for i, dataset in enumerate(list):
        score = 4
        for j, idx in enumerate(dataset):
            scores[idx] += score - j

    print(scores)
    score = np.array(scores)
    indices = score.argpartition(-4)[-4:]

    # The `highest_scores` array now contains the highest_scores of the 4 largest values in the original array.
    print("highest_scores of 4 max values:", indices)
    indices_sorted = indices[np.argsort(-score[indices])]
    print("Indices of 4 max values, sorted:", indices_sorted)
    for i, descriptor in enumerate(indices_sorted):
        print(f"{i + 1}.", feature_names[descriptor])


def read_feature_output():
    def extract_info(sentence):
        parts = sentence.split(":")
        number = parts[0].split(" ")[1]
        dataset = parts[0].split(" ")[-2]
        reverse_feature_names = {v: k for k, v in feature_names.items()}
        indexes = [reverse_feature_names[feature.strip()] for feature in parts[1].split(",")]
        return number, dataset, indexes

    input_lines = sys.stdin.readlines()

    for line in input_lines:
        number, dataset, indexes = extract_info(line)
        print(f"{number} {dataset}: {indexes}")
