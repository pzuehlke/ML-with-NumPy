# Type:
#   * Supervised learning
#   * Prediction
#   * Classification
# -----------------------
# Description:
# Given a dataset such that each example consists of features (x_1, ..., x_n)
# and a label y, the algorithm predicts the label of a new datum X by looking
# at the K closest examples and counting how many among these K fall into each
# bin (class); the predicted label y_hat is the one whose bin has the most
# elements.
#
# Performance is measured by the error rate on a test set (the fraction of the
# data in the test set that are incorrectly classified).
#
# In our implementation K is treated as a hyperparameter which is tuned
# using a cross-validation (development) set. The user provides a range of
# values for K, from K_min to K_max, then the algorithm selects that value of K
# for which the error rate on the cross-validation set is smallest. To force a
# specific value of K to be chosen, take K_min = K_max.

import numpy as np


class KNearestNeighbors:
    """
    K Nearest Neighbors classifier that selects the optimal K value using
    cross-validation.
    """

    def __init__(self,
                 K_min: int, K_max: int,
                 X_train: np.ndarray, Y_train: np.ndarray,
                 X_cv: np.ndarray, Y_cv: np.ndarray):
        """
        Initializes the classifier with training and cross-validation datasets,
        and the range for K.

        Parameters:
            * K_min: Minimum value of K to consider.
            * K_max: Maximum value of K to consider.
            * X_train (2D (m, n) array): Training features.
            * Y_train (2D (m, 1) array): Training labels.
            * X_cv (2D array): Cross-validation features.
            * Y_cv (2D array): Cross-validation labels.
        """
        self.K_min = K_min
        self.K_max = K_max
        self.K = 0  # Will be set during training
        self.X_train = X_train  # Training examples
        self.Y_train = Y_train.reshape((-1, 1))  # Associated training labels
        self.X_cv = X_cv  # Cross-validation data
        self.Y_cv = Y_cv.reshape((-1, 1))  # Associated cross-validation labels

    def _label(self, X: np.ndarray, k: int) -> int:
        """
        Predicts the label for a single sample X based on the k nearest
        neighbors.

        Parameters:
            * X (2D (1, n) array): The input sample.
            * k: The number of nearest neighbors to consider.

        Returns:
            * The predicted label y_hat.
        """
        diffs = self.X_train - X  # (m, n)
        squared_dists = np.sum(diffs**2, axis=1)  # (m, )
        idx = np.argsort(squared_dists)[:k]  # k indices of smallest dists
        labels, counts = np.unique(self.Y_train[idx], return_counts=True)
        y_hat = labels[np.argmax(counts)]
        return y_hat

    def _evaluate_loss(self, X: np.ndarray, y: int, k: int) -> int:
        """
        Evaluates the loss for a single sample and label, using the k nearest
        neighbors.

        Parameters:
            * X: The input sample.
            * y: The true label for the sample.
            * k: The number of nearest neighbors to consider.

        Returns:
            * 0 if prediction is correct, 1 otherwise (useful to count errors).
        """
        return int(y != self._label(X, k))

    def train(self) -> None:
        """
        Trains the classifier by selecting the best hyperparameter K based on
        the lowest error rate on the cross-validation set.
        """
        best_K = self.K_min  # Initialize best K
        best_er = 1.0  # Initialize error rate
        total = len(self.X_cv)

        for k in range(self.K_min, self.K_max + 1):
            nr_errors = (sum(self._evaluate_loss(X, y, k)
                             for X, y in zip(self.X_cv, self.Y_cv)))
            error_rate = nr_errors / total
            if error_rate < best_er:
                best_K = k
                best_er = error_rate

        self.K = best_K

    def compute_error_rate(self, X_test: np.ndarray, Y_test: np.ndarray
                           ) -> float:
        """
        Computes the error rate of the classifier on a given test dataset.

        Parameters:
            * X_test (2D (m, n) array): Test features, where 'm' is the number
              of test samples and 'n' is the number of features.
            * Y_test (2D (m, 1) array): True labels for the test samples.

        Returns:
            * error_rate (float): The proportion of incorrect predictions out
              of all test samples, ranging from 0 to 1.

        """
        Y_test.reshape(-1, 1)  # Ensure Y_test is a 2D column vector.
        total = len(Y_test)
        nr_errors = sum(int(self.predict(X) != y)
                        for X, y in zip(X_test, Y_test))
        return nr_errors / total

    def predict(self, X: np.ndarray) -> int:
        """
        Predicts the label for a single sample X using the optimal K found
        during training.

        Parameters:
            * X: The input sample.

        Returns:
            * The predicted label y_hat.
        """
            return self._label(X, self.K)
