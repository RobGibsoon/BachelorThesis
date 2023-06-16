import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from utils import feature_names, inputs

"""This file contains various methods that were used to analyse the results more efficiently."""


def read_csv_predictions(dn, clf, fs):
    """nullhypothesis is that the two compared classification methods are equal
    if we discard the hypothesis, we conclude that the embedding (E1) has a higher average accuracy than the reference (E2)"""
    if clf == "ann":
        z_scores = []
        for i in range(5):
            data = pd.read_csv(f'../log/predictions/preds_labels_ANN{i}_{dn}_fs{fs}.csv')
            ref = pd.read_csv(f'../log/predictions/predictions_gnn_{dn}_{i + 1}.csv')
            score = calc_z_score(data, ref['predictions'])
            z_scores.append(score)
            if score <= 5:
                print(f'{dn} {fs} ann{i + 1} score: ', score)


    elif clf == "knn":
        data = pd.read_csv(f'../log/predictions/preds_labels_KNeighborsClassifier_{dn}_fs{fs}.csv')
        ref = pd.read_csv(f'../log/predictions/reference_preds_labels_KNeighborsClassifier_{dn}.csv')
        score = calc_z_score(data, ref['preds'])
        if score <= 5:
            print(f'{dn} {fs} knn score: ', score)


    else:
        data = pd.read_csv(f'../log/predictions/preds_labels_SVC_{dn}_fs{fs}.csv')
        ref = pd.read_csv(f'../log/predictions/reference_preds_labels_SVC_{dn}.csv')
        score = calc_z_score(data, ref['preds'])
        if score <= 5:
            print(f'{dn} {fs} svm score: ', score)


def calc_z_score(data, ref_preds):
    E1 = (data['preds'] == data['labels']).mean()
    E2 = (ref_preds == data['labels']).mean()
    E12 = np.mean((data['preds'] == data['labels']) & (ref_preds == data['labels']))
    D_test = data['preds'].size
    mu = E1 - E2
    sigma = (E1 - E12) * (1 - E1 + E2) ** 2 + (E2 - E12) * (1 + E1 - E2) ** 2 + (1 + 2 * E12 - E1 - E2) * (E1 - E2) ** 2
    Z = mu / np.sqrt(sigma / D_test)

    N_Z = 1 - norm.cdf(Z)
    return N_Z * 100


def create_indices_lists_from_names_list():
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
    """inputs needs to have form: The x best features for using clf on dataset were: i_1, i_2, ..."""

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


def calc_stand_devs():
    data_SFS = {
        "MUTAG": [68.42, 84.21, 68.42, 81.58, 68.42],
        "PTC_MM": [60.29, 60.29, 60.29, 60.29, 60.29],
        "PTC_MR": [59.42, 59.42, 59.42, 59.42, 59.42],
        "ER_MD": [55.56, 55.56, 55.56, 54.44, 54.44],
        "PTC_FM": [58.57, 61.43, 60.0, 60.0, 61.43],
        "DHFR_MD": [73.42, 72.15, 72.15, 73.42, 72.15],
        "PTC_FR": [57.75, 57.75, 57.75, 57.75, 57.75],
        "Mutagenicity": [61.63, 61.88, 61.14, 61.51, 61.63],
    }
    data_mRMR = {
        "MUTAG": [68.42, 68.42, 68.42, 68.42, 68.42],
        "Mutagenicity": [64.11, 64.36, 64.11, 64.36, 64.11],
        "DHFR_MD": [73.42, 72.15, 73.42, 73.42, 73.42],
        "PTC_MR": [59.42, 52.17, 60.87, 53.62, 59.42],
        "PTC_FM": [60.0, 60.0, 60.0, 60.0, 60.0],
        "ER_MD": [55.56, 55.56, 53.33, 55.56, 53.33],
        "PTC_MM": [60.29, 60.29, 60.29, 60.29, 60.29],
        "PTC_FR": [57.75, 57.75, 57.75, 57.75, 57.75]
    }
    data_noFS = {
        "MUTAG": [86.84, 84.21, 81.58, 81.58, 78.95],
        "PTC_FM": [60.0, 60.0, 60.0, 60.0, 60.0],
        "PTC_MR": [59.42, 59.42, 60.87, 60.87, 59.42],
        "PTC_FR": [57.75, 57.75, 57.75, 57.75, 57.75],
        "ER_MD": [54.44, 52.22, 52.22, 52.22, 53.33],
        "DHFR_MD": [73.42, 70.89, 70.89, 73.42, 72.15],
        "PTC_MM": [60.29, 60.29, 60.29, 60.29, 60.29],
        "Mutagenicity": [63.24, 63.86, 63.0, 63.24, 63.24]
    }
    data_reference = {
        "PTC_FM": [60.0, 65.71, 60.0, 57.14, 58.57],
        "PTC_MM": [76.47, 76.47, 79.41, 76.47, 75.0],
        "PTC_FR": [60.56, 61.97, 60.56, 61.97, 59.15],
        "ER_MD": [62.22, 60.0, 62.22, 61.11, 63.33],
        "PTC_MR": [62.32, 62.32, 59.42, 60.87, 60.87],
        "DHFR_MD": [67.09, 72.15, 69.62, 68.35, 74.68],
        "MUTAG": [81.58, 81.58, 81.58, 84.21, 84.21],
        "Mutagenicity": [75.62, 76.49, 76.73, 75.99, 75.87]
    }

    sfs = pd.DataFrame(data_SFS)
    mrmr = pd.DataFrame(data_mRMR)
    no_fs = pd.DataFrame(data_noFS)
    ref = pd.DataFrame(data_reference)

    # Then you can calculate the standard deviation for each dataset like this:
    sfs_std = sfs.std()
    mrmr_std = mrmr.std()
    no_fs_std = no_fs.std()
    ref_std = ref.std()

    print('sfs \n ', np.round(sfs_std, 2), '\n')
    print('mrmr \n ', np.round(mrmr_std, 2), '\n')
    print('no_fs \n ', np.round(no_fs_std, 2), '\n')
    print('ref \n ', np.round(ref_std, 2), '\n')


if __name__ == "__main__":
    for combination in inputs.values():
        ref = combination[3]
        fs = combination[2]
        clf = combination[1]
        dn = combination[0]
        if ref or clf == "svm" or (clf == "knn" and dn == "Mutagenicity"):
            continue
        read_csv_predictions(dn, clf, fs)
