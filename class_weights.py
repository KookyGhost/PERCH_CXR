import numpy as np
import pandas as pd

def get_sample_counts(csv_dir):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int
    """
    df = pd.read_csv(csv_dir)
    df = df.iloc[1:,:].fillna(0)
    df = df.replace(-1, 0)
    total_count = df.shape[0]
    class_names = df.iloc[:, 1:].columns.to_list()
    labels = df[class_names].to_numpy()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts


def get_class_weights(total_counts, class_positive_counts, multiply):

    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights
