import numpy as np


def compute_entropy(y, n):
    """
    Computes the entropy of a distribution for the given classes in array y.

    The entropy is calculated using the formula:
        H(S) = -sum(p_i * log2(p_i)),
    where p_i is the proportion of the i-th class in the dataset, and the sum
    is over all n classes.

    Parameters:
        * y (1-dimensional np.ndarray): An array of integers where each element
            represents the class of an example. These classes are assumed to
            range from 0 to n-1.
        * n (int): The number of different classes. This is used to ensure the
            bincount array has a length of at least n, even if some classes
            do not appear in y.
    Returns:
        * float: The entropy of the distribution of class labels in y.
    """
    # Count number of occurrences of each class within y:
    counts = np.bincount(y, minlength=n)

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


def split_dataset(X, node_indices, feature):
    pass
    # TODO

# if __name__ == "__main__":
    # run_binary_decision_trees()
