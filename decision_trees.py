import numpy as np
from typing import Tuple  # Only used for type annotations


def compute_entropy(y: np.ndarray) -> float:
    """
    Computes the entropy of a distribution for the given (possibly more than
    two) classes in array y.

    The entropy is calculated using the formula:
        H(S) = -sum(p_i * log2(p_i)),
    where p_i is the proportion of the i-th class in the dataset, and the sum
    is over all possible classes.

    Parameters:
        * y (1-dimensional np.ndarray): An array of integers where each element
            represents the class of an example.

    Returns:
        * entropy (float): The entropy of the distribution of labels in y.
    """
    # Count number of occurrences of each class within y:
    counts = np.bincount(y)

    # Compute respective proportions for each class:
    total = len(y)
    proportions = counts / total

    # Return 0 entropy if there are no elements to classify:
    if total == 0:
        return 0.0

    # Filter out zero proportions to avoid log2(0):
    nonzero_proportions = proportions[proportions > 0]

    # Compute the entropy:
    entropy = -np.dot(nonzero_proportions, np.log2(nonzero_proportions))
    return entropy


def split_dataset(X: np.ndarray, node_indices: np.ndarray, feature: int,
                  threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the dataset on a specified feature and threshold, returning the
    indices of the rows that go to the left and right child nodes.

    Parameters:
        * X ((m, n) np.ndarray): The dataset to split, where each row is an
          instance and each column is a feature.
        * node_indices (1d np.ndarray): The indices of the rows in X that are
          being considered for the split.
        * feature (int): The index of the feature to split on.
        * threshold (float): The threshold value for the split.

    Returns:
        * left_indices (1d np.ndarray): Indices of rows going to left child.
        * right_indices (1d np.ndarray): Indices of rows going to right child.
    """
    # Determine which rows go to left and right child nodes based on threshold:
    left_mask = X[node_indices, feature] <= threshold
    right_mask = ~left_mask  # The opposite of the left mask

    # Apply the masks to node_indices to get the original indices of the rows:
    left_indices = node_indices[left_mask]
    right_indices = node_indices[right_mask]

    return left_indices, right_indices


def compute_information_gain(X: np.ndarray,
                             y: np.ndarray,
                             node_indices: np.ndarray,
                             left_indices: np.ndarray,
                             right_indices: np.ndarray
                             ) -> float:
    """
    Computes the information gain of a split, based on the entropy before and
    after the split.

    Information gain is calculated as the difference between the entropy of the
    parent node and the weighted sum of the entropies of the two child nodes
    (left and right).

    Parameters:
        * X (2d np.ndarray): The feature matrix where each row is an example
          and each column is a feature.
        * y (1d np.ndarray): The label vector where each element corresponds to
          the label of the example.
        * node_indices (1d np.ndarray): Indices of the rows in X and y that are
          being considered for the split.
        * left_indices (1d np.ndarray): Indices of the rows in X and y that go
          to the left child node after the split.
        * right_indices (1d np.ndarray): Indices of the rows in X and y that go
          to the right child node after the split.

    Returns:
        * info_gain (float): The information gain resulting from the split.
    """

    # Extract labels for the node and its children:
    y_node = y[node_indices]
    y_left = y[left_indices]
    y_right = y[right_indices]

    # Compute entropy for the node and its children:
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)

    # Calculate proportion of instances in each child relative to parent node:
    left_weight = len(left_indices) / len(node_indices)
    right_weight = len(right_indices) / len(node_indices)

    # Compute information gain:
    info_gain = (node_entropy
                 - left_weight * left_entropy
                 - right_weight * right_entropy)

    return info_gain


def get_best_split(X: np.ndarray,
                   y: np.ndarray,
                   node_indices: np.ndarray
                   ) -> (int, float):
    """
    Finds the best feature and threshold to split the node on, maximizing
    information gain.

    Iterates over all features and possible thresholds (unique values in the
    labels) to find the combination that results in the highest information
    gain.

    Parameters:
        * X (2d np.ndarray): The feature matrix where each row is an example
          and each column is a feature.
        * y (1d np.ndarray): The label vector where each element corresponds to
          the label of the example.
        * node_indices (1d np.ndarray): Indices of the rows in X and y that are
          being considered for the split.

    Returns:
        best_feature, best_threshold (tuple[int, float]): The index of
        the best feature and the best threshold value to split on.
    """
    num_features = X.shape[1]
    best_threshold = None
    best_feature = -1
    best_info_gain = 0

    for feature in range(num_features):
        # Extract the values of the specified feature for the given node:
        feature_values = X[node_indices, feature]
        unique_values = np.unique(feature_values)
        sorted_unique_values = np.sort(unique_values)
        # Calculate midpoints between consecutive sorted unique values:
        midpoints = (sorted_unique_values[:-1] + sorted_unique_values[1:]) / 2

        for threshold in midpoints:
            # Split the dataset based on the current feature and threshold:
            left_indices, right_indices = split_dataset(X, node_indices,
                                                        feature, threshold)
            # Compute the information gain from the current split:
            info_gain = compute_information_gain(X, y, node_indices,
                                                 left_indices, right_indices)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


# if __name__ == "__main__":
    # run_binary_decision_trees()
