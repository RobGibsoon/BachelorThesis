import sys

from utils import feature_names

if __name__ == "__main__":
    """this can be used to pass the saved features and get back the indexes of the features"""


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
