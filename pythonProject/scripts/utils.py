import csv


def get_csv_idx_split(dn, idx_type):
    file = open(f"log/index_splits/{dn}_{idx_type}_split.csv", "r")
    idx_split = list(csv.reader(file, delimiter=','))
    parsed_idx_split = [int(elt) for elt in idx_split[0]]
    return parsed_idx_split