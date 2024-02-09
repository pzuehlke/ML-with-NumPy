import numpy as np


def find_closest_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Computes the index of the closest centroid for each example in X.

    Parameters:
        * X ((m, n) array): m input examples having n features each.
        * centroids ((K, n) array): K n-dimensional centroids of the clusters.
    Returns:
        * An integer array of shape (m, ) containing the index of the closest
          centroid for each example.
    """
    K = centroids.shape[0]      # Set K = number of clusters
    # Tile X to create an (m, K, n) array:
    X = np.tile(X[:, np.newaxis, :], (1, K, 1))
    # Compute the differences of each example to each centroid by broadcasting:
    diffs = X - centroids
    # Compute the squares of the differences, sum over the n dimensions:
    squared_distances = np.sum(diffs**2, axis=2)  # (m, K) array
    # For each example, take the index yielding the smallest distance:
    closest_centroids = np.argmin(squared_distances, axis=1)  # (m, ) array
    return closest_centroids


def recompute_centroids(X: np.ndarray, closest_to: np.ndarray, K: int
                        ) -> np.ndarray:
    """
    Returns the new centroids by computing the means of the data points closest
    to each of the original centroids.
    Parameters:
        * X ((m, n) array): m input examples having n features each.
        * closest_to ((m, ) array): Contains the index of the closest centroid
          for each example in X. Concretely, closest_to[i] contains the index
          of the centroid closest to example i.
        * K (int): number of centroids.
    Returns:
        * new_centroids ((K, n) array): New centroids computed
    """
    m, n = X.shape
    new_centroids = np.zeros((K, n))
    for k in range(K):
        indices = np.where(closest_to == k)[0]
        if len(indices) > 0:
            new_centroids[k] = np.mean(X[indices], axis=0)
        else:
            print(f"Warning: Cluster {k} is empty! Centroid unchanged.")
    return new_centroids


def initialize_centroids(X: np.ndarray, K: int) -> np.ndarray:
    """
    Initializes K centroids by randomly selecting K distinct examples from the
    dataset X.

    Parameters:
        * X ((m, n) array): The dataset from which centroids are to be
          initialized, consisting of m examples with n features each.
        * K (int): The number of centroids to initialize.

    Returns:
        * (K, n) array: The K n-dimensional centroids.
    """
    # Randomly reorder the indices of examples:
    rand_indices = np.random.permutation(X.shape[0])
    # Then take the first K examples as the centroids:
    centroids = X[rand_indices[:K]]
    return centroids


def run_kMeans(X: np.ndarray, K: int, max_iters: int = 10
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Executes the K-Means clustering algorithm on the dataset X.
    Parameters:
        * X ((m, n) array): The input dataset, where each row represents a
          single example.
        * K (int): The number of centroids (clusters) to use.
        * max_iters (int, optional): The maximum number of iterations to
          perform. Defaults to 10.
    Returns:
        * centroids ((K, n) array): The final centroids after convergence or
          reaching the maximum iterations.
        * closest_to ((m, ) array): The index of the closest centroid for each
          example in X.
    """
    m, n = X.shape
    centroids = initialize_centroids(X, K)
    closest_to = np.zeros(m, dtype=int)

    for j in range(max_iters):
        old_closest_to = closest_to.copy()
        closest_to = find_closest_centroids(X, centroids)
        # Check for convergence:
        if np.array_equal(closest_to, old_closest_to):
            print(f"Convergence reached at iteration {j + 1}.")
            break
        centroids = recompute_centroids(X, closest_to, K)
        print(f"K-means iteration {j + 1} out of {max_iters}")
    return centroids, closest_to


m = 1000  # insert your own value here
n = 2     # insert your own value here
X = np.zeros((m, n))  # insert your own value here
K = 4  # insert your own value here

if __name__ == "__main__":
    run_kMeans(X, K, max_iters=10)
