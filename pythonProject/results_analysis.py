import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from utils import feature_names, top_5_mRMR_features

"""This file contains various methods that were used to analyse the results more efficiently."""


def read_csv_predictions(dn, clf, fs):
    """nullhypothesis is that the two compared classification methods are equal
    if we discard the hypothesis, we conclude that the embedding (E1) has a higher average accuracy than the reference (E2)"""
    # todo Check whether mRMR or SFS selection was used. If SFS -> fs can be left alone, if mRMR -> uncomment next line
    true_false = fs
    # fs = 'mrmr'
    if clf == "ann":
        z_scores = []
        for i in range(5):
            if dn == "Mutagenicity":
                print("here")
            data = pd.read_csv(f'../log/predictions/preds_labels_ANN{i}_{dn}_fs{fs}.csv')
            reference = pd.read_csv(f'../log/predictions/predictions_gnn_{dn}_{i + 1}.csv')
            score = calc_z_score(dn, clf, data, reference['preds'])
            if score <= 5:
                z_scores.append(score)
        if len(z_scores) > 2:
            print(f'{dn} ann {true_false}: {len(z_scores)}/5 was significantly relevant')
    elif clf == "knn":
        data = pd.read_csv(f'../log/predictions/preds_labels_KNeighborsClassifier_{dn}_fs{fs}.csv')
        reference = pd.read_csv(f'../log/predictions/reference_preds_labels_KNeighborsClassifier_{dn}.csv')
        if not (data['labels'].equals(reference['labels'])):
            print(dn, true_false, 'knn does not have equals labels')
        score = calc_z_score(dn, clf, data, reference['preds'])
        if score <= 5:
            print(f'{dn} {true_false} knn score: ', score)
    else:
        data = pd.read_csv(f'../log/predictions/preds_labels_SVC_{dn}_fs{fs}.csv')
        reference = pd.read_csv(f'../log/predictions/reference_preds_labels_SVC_{dn}.csv')
        if not (data['labels'].equals(reference['labels'])):
            print(dn, true_false, 'svm does not have equals labels')
        score = calc_z_score(dn, clf, data, reference['preds'])
        if score <= 5:
            print(f'{dn} {true_false} svm score: ', score)


def calc_z_score(dn, clf, data, ref_preds):
    E1 = (data['preds'] == data['labels']).mean()
    E2 = (ref_preds == data['labels']).mean()
    E12 = np.mean((data['preds'] == data['labels']) & (ref_preds == data['labels']))
    # to see whether accuracies are significantly worse than references uncomment next 3 lines
    # temp = E2
    # E2 = E1
    # E1 = temp
    D_test = data['preds'].size
    mu = E1 - E2
    sigma = (E1 - E12) * (1 - E1 + E2) ** 2 + (E2 - E12) * (1 + E1 - E2) ** 2 + (1 + 2 * E12 - E1 - E2) * (E1 - E2) ** 2
    Z = mu / np.sqrt(sigma / D_test)

    N_Z = 1 - norm.cdf(Z)
    return N_Z * 100


# reverse the dictionary for easy lookup
reverse_feature_dict = {v: k for k, v in feature_names.items()}


def generate_indices_list(classifier):
    clf_list = []
    for dataset, features in classifier.items():
        indices = [reverse_feature_dict[feature] for feature in features]
        clf_list.append(indices)
        print(f"\"{dataset}\": {indices},")

    return clf_list


def create_indices_lists_from_names_list():
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
    SVM = {
        "MUTAG": ["estrada", "narumi", "padmakar-ivan", "polarity-nr", "randic", "szeged", "zagreb", "nodes", "edges",
                  "label_entropy"],
        "Mutagenicity": ["balaban", "estrada", "polarity-nr", "szeged", "nodes", "schultz", "hyp_wiener", "n_impurity",
                         "label_entropy", "edge_strength"],
        "ER_MD": ["balaban", "narumi", "padmakar-ivan", "polarity-nr", "randic", "szeged", "zagreb", "n_impurity",
                  "label_entropy", "edge_strength"],
        "DHFR_MD": ["balaban", "estrada", "narumi", "padmakar-ivan", "randic", "szeged", "wiener", "zagreb",
                    "n_impurity", "label_entropy"],
        "PTC_MR": ["balaban", "estrada", "narumi", "padmakar-ivan", "polarity-nr", "randic", "szeged", "wiener",
                   "edges", "label_entropy"],
        "PTC_MM": ["balaban", "estrada", "narumi", "padmakar-ivan", "polarity-nr", "randic", "szeged", "wiener",
                   "zagreb", "nodes"],
        "PTC_FM": ["balaban", "padmakar-ivan", "polarity-nr", "randic", "zagreb", "nodes", "edges", "n_impurity",
                   "label_entropy", "edge_strength"],
        "PTC_FR": ["balaban", "estrada", "narumi", "polarity-nr", "randic", "wiener", "zagreb", "nodes", "n_impurity",
                   "edge_strength"]
    }
    KNN = {
        "MUTAG": ["balaban", "estrada", "narumi", "padmakar-ivan", "polarity-nr", "szeged", "edges", "hyp_wiener",
                  "n_impurity", "edge_strength"],
        "Mutagenicity": ["balaban", "narumi", "padmakar-ivan", "polarity-nr", "szeged", "zagreb", "nodes", "n_impurity",
                         "label_entropy", "edge_strength"],
        "ER_MD": ["balaban", "narumi", "padmakar-ivan", "polarity-nr", "randic", "szeged", "zagreb", "edges",
                  "n_impurity", "edge_strength"],
        "DHFR_MD": ["balaban", "estrada", "narumi", "padmakar-ivan", "polarity-nr", "randic", "szeged", "wiener",
                    "n_impurity", "edge_strength"],
        "PTC_MR": ["estrada", "padmakar-ivan", "polarity-nr", "randic", "wiener", "zagreb", "nodes", "mod_zagreb",
                   "hyp_wiener", "edge_strength"],
        "PTC_MM": ["estrada", "padmakar-ivan", "polarity-nr", "randic", "szeged", "zagreb", "nodes", "schultz",
                   "hyp_wiener", "n_impurity"],
        "PTC_FM": ["balaban", "narumi", "polarity-nr", "randic", "wiener", "zagreb", "nodes", "edges", "hyp_wiener",
                   "edge_strength"],
        "PTC_FR": ["estrada", "narumi", "padmakar-ivan", "polarity-nr", "randic", "szeged", "zagreb", "nodes",
                   "schultz", "hyp_wiener"]
    }
    ANN = {
        "MUTAG": ["zagreb", "balaban", "schultz", "polarity-nr", "edge_strength", "nodes", "hyp_wiener", "n_impurity",
                  "mod_zagreb", "szeged"],
        "Mutagenicity": ["edge_strength", "label_entropy", "randic", "narumi", "wiener", "schultz", "polarity-nr",
                         "n_impurity", "hyp_wiener", "mod_zagreb"],
        "ER_MD": ["wiener", "balaban", "nodes", "edges", "szeged", "randic", "hyp_wiener", "n_impurity", "polarity-nr",
                  "label_entropy"],
        "DHFR_MD": ["balaban", "estrada", "narumi", "padmakar-ivan", "polarity-nr", "randic", "szeged", "wiener",
                    "n_impurity", "zagreb"],
        "PTC_MR": ["n_impurity", "mod_zagreb", "edge_strength", "wiener", "hyp_wiener", "zagreb", "schultz", "estrada",
                   "padmakar-ivan", "polarity-nr"],
        "PTC_MM": ["narumi", "nodes", "wiener", "n_impurity", "balaban", "estrada", "padmakar-ivan", "polarity-nr",
                   "hyp_wiener", "randic"],
        "PTC_FM": ["schultz", "balaban", "n_impurity", "polarity-nr", "estrada", "narumi", "szeged", "label_entropy",
                   "edge_strength", "randic"],
        "PTC_FR": ["balaban", "estrada", "narumi", "padmakar-ivan", "polarity-nr", "randic", "szeged", "wiener",
                   "zagreb", "nodes"]
    }
    mRMR_features = {}
    for key in top_5_mRMR_features:
        string_names = []
        for value in top_5_mRMR_features[key]:
            string_names.append(feature_names[value])
        mRMR_features[key] = string_names

    # calculate count of feature appearance
    print("Feature Counts")
    print("\n_________K-NN__________ ")
    get_feature_count(KNN)
    print("\n_________SVM__________ ")
    get_feature_count(SVM)
    print("\n_________ANN__________ ")
    get_feature_count(ANN)
    print("\n_________mRMR__________ ")
    get_feature_count(mRMR_features)

    # calculate it with points: RANK
    print("\n\n\n\nRANK mode")
    print("\n_________K-NN__________ ")
    get_feature_points(KNN, 'RANK')
    print("\n_________SVM__________ ")
    get_feature_points(SVM, 'RANK')
    print("\n_________ANN__________ ")
    get_feature_points(ANN, 'RANK')
    print("\n_________mRMR__________ ")
    get_feature_points(mRMR_features, 'RANK')

    # calculate it with points: HALVES
    print("\n\n\n\nHALVES mode")
    print("\n_________K-NN__________ ")
    get_feature_points(KNN, 'HALVES')
    print("\n_________SVC__________ ")
    get_feature_points(SVM, 'HALVES')
    print("\n_________ANN__________ ")
    get_feature_points(ANN, 'HALVES')
    print("\n_________mRMR__________ ")
    get_feature_points(mRMR_features, 'HALVES')


def get_feature_points(dict, mode):
    """this method gives the features points based on how often they appear and which spot they place on
    You can pick between two modes: HALVES, RANK"""

    feature_points = {}

    for key in dict:
        for idx, feature in enumerate(dict[key]):
            if feature not in feature_points:
                feature_points[feature] = 0

            if mode == 'HALVES':
                half_point = len(dict[key]) // 2
                if idx < half_point:
                    feature_points[feature] += 2
                else:
                    feature_points[feature] += 1

            elif mode == 'RANK':
                if idx == 0:
                    feature_points[feature] += 4
                elif idx == 1:
                    feature_points[feature] += 3
                elif idx == 2:
                    feature_points[feature] += 2
                else:
                    feature_points[feature] += 1
    # sort features in dictionary by their counts
    sorted_feature_count = sorted(feature_points.items(), key=lambda item: item[1], reverse=True)
    print(sorted_feature_count)
    return feature_points


def get_feature_count(dict):
    # a dictionary to store the feature count
    feature_count = {}

    # loop through the ANN dictionary
    for key in dict:
        # for each feature in the list
        for feature in dict[key]:
            # if the feature is already in the feature_count dictionary, increment its count
            if feature in feature_count:
                feature_count[feature] += 1
            # else, add the feature to the dictionary with a count of 1
            else:
                feature_count[feature] = 1
    print(feature_count)
    sorted_feature_count = sorted(feature_count.items(), key=lambda item: item[1], reverse=True)
    print(sorted_feature_count)
    return feature_count


def get_best_features(list):
    """this can be used to pass the saved features and get back the indexes of the features"""
    scores = [0] * 17
    print(scores)
    # list = [[14, 2, 15, 1],
    #         [14, 2, 15, 1],
    #         [14, 2, 16, 15],
    #         [14, 0, 2, 15],
    #         [8, 0, 10, 5],
    #         [0, 13, 16, 15],
    #         [5, 15, 12, 9],
    #         [12, 15, 9, 5]]

    for i, dataset in enumerate(list):
        score = 16
        for j, idx in enumerate(dataset):
            scores[idx] += score - j

    print(scores)
    score = np.array(scores)
    print("score per feature:")
    for i, elt in enumerate(score):
        print(f"{feature_names[i]}: {elt}")
    indices = score.argpartition(-5)[-5:]

    # The `highest_scores` array now contains the highest_scores of the 4 largest values in the original array.
    print("highest_scores of 4 max values:", indices)
    indices_sorted = indices[np.argsort(-score[indices])]
    print("Indices of 4 max values, sorted:", indices_sorted)
    for i, descriptor in enumerate(indices_sorted):
        print(f"{i + 1}.", feature_names[descriptor])

    print("This is done by adding, maybe a different approach would be better")


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
        "MUTAG": [71.05, 76.32, 68.42, 78.95, 68.42],
        "PTC_MM": [64.71, 64.71, 64.71, 64.71, 64.71],
        "PTC_MR": [60.87, 60.87, 60.87, 59.42, 59.42],
        "ER_MD": [58.89, 58.89, 58.89, 58.89, 58.89],
        "PTC_FM": [60.0, 58.57, 60.0, 60.0, 60.0],
        "DHFR_MD": [68.35, 68.35, 68.35, 69.62, 68.35],
        "PTC_FR": [56.34, 56.34, 56.34, 56.34, 56.34],
        "Mutagenicity": [61.63, 62.0, 60.64, 61.01, 60.89],
    }
    data_mRMR = {
        "MUTAG": [68.42, 68.42, 68.42, 68.42, 68.42],
        "Mutagenicity": [63.99, 64.23, 63.86, 63.86, 63.99],
        "DHFR_MD": [68.35, 68.35, 68.35, 68.35, 68.35],
        "PTC_MR": [62.32, 59.42, 62.32, 59.42, 63.77],
        "PTC_FM": [58.57, 60.0, 60.0, 60.0, 60.0],
        "ER_MD": [61.11, 62.22, 58.89, 58.89, 64.44],
        "PTC_MM": [64.71, 64.71, 64.71, 64.71, 64.71],
        "PTC_FR": [56.34, 56.34, 56.34, 56.34, 56.34]
    }
    data_noFS = {
        "MUTAG": [76.32, 84.21, 86.84, 84.21, 86.84],
        "ER_MD": [55.56, 55.56, 60.0, 56.67, 58.89],
        "DHFR_MD": [68.35, 69.62, 69.62, 68.35, 68.35],
        "Mutagenicity": [63.0, 63.74, 63.12, 63.37, 63.49],
        "PTC_FM": [58.57, 60.0, 60.0, 60.0, 60.0],
        "PTC_MR": [66.67, 65.22, 62.32, 60.87, 65.22],
        "PTC_MM": [64.71, 64.71, 64.71, 64.71, 64.71],
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

    print('sfs \n', np.round(sfs_std, 2), '\n')
    print('mrmr \n', np.round(mrmr_std, 2), '\n')
    print('no_fs \n', np.round(no_fs_std, 2), '\n')
    print('ref \n', np.round(ref_std, 2), '\n')


if __name__ == "__main__":
    create_indices_lists_from_names_list()
    # get_best_features()
    # calc_stand_devs()

    """for combination in inputs.values():
        is_ref = combination[3]
        is_fs = combination[2]
        clf = combination[1]
        dn = combination[0]
        if is_ref or (dn == "Mutagenicity"):
            continue
        read_csv_predictions(dn, clf, is_fs)"""
