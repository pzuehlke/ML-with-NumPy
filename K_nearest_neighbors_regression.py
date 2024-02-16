# Type:
#   * Supervised learning
#   * Prediction
#   * Regression
# -----------------------
# Description:
# Given a dataset such that each example is comprised of n features
# X = (x_1, ..., x_n) and a numerical non-discrete output y = f(X), where f is
# unknown, the algorithm tries to estimate the value of f on a new datum by
# looking at the K closest examples and taking an average of the respective K
# estimates.
#
# Performance is measured by the mean squared error on a test set.
#
# In our implementation the algorithm decides whether to use the arithmetic
# mean or a weighted average (by the corresponding distances) by tuning on a
# cross-validation (development) set. Similarly, K is treated as a
# hyperparameter which is also tuned: the user provides a range of values for
# K, from K_min to K_max, and the algorithm selects that value of K for which
# the MSE on the cross-validation set is smallest. To force a specific value of
# K to be chosen, take K_min = K_max.

import numpy as np


class KNearestNeighborsRegression:
    """
    K Nearest Neighbors regressor that selects the optimal K value and the
    weighting method using cross-validation.
    """

    def __init__(self,
                 K_min: int, K_max: int,
                 X_train: np.ndarray, Y_train: np.ndarray,
                 X_cv: np.ndarray, Y_cv: np.ndarray):
        """
        Initializes the regressor with training and cross-validation datasets,
        and the range for K.

        Parameters:
            * K_min: Minimum value of K to consider.
            * K_max: Maximum value of K to consider.
            * X_train (2D (m, n) array): Training features.
            * Y_train (1D (m,) array): Training target values.
            * X_cv (2D array): Cross-validation features.
            * Y_cv (1D array): Cross-validation target values.
        """
        # Error handling for K_min and K_max:
        if not isinstance(K_min, int) or not isinstance(K_max, int):
            raise ValueError("K_min and K_max must be integers")
        if K_min <= 0 or K_max <= 0:
            raise ValueError("K_min and K_max must be positive")
        if K_max > K_min:
        raise ValueError("K_min must be less than or equal to K_max")

        self.K_min = K_min
        self.K_max = K_max
        self.K = 0  # Will be set during training
        self.weighted = False  # Will be reset during training
        self.X_train = X_train
        self.Y_train = Y_train.reshape((-1, 1))  # Ensure Y_train is column vec
        self.X_cv = X_cv
        self.Y_cv = Y_cv.reshape((-1, 1))  # Ensure Y_cv is a column vector

    def _estimate(self, X: np.ndarray, k: int, weighted: bool = False
                  ) -> float:
        """
        Predicts the target value for a single sample X based on the k nearest
        neighbors, using either a simple or weighted average of the neighbors'
        values.

        Parameters:
            * X (2D (1, n) array): The input sample.
            * k: The number of nearest neighbors to consider.
            * weighted (bool): If True, use a weighted average based on the
              inverse of the distances; otherwise, use a simple average.

        Returns:
            * The predicted target value y_hat.
        """
        # OPTIMIZE: Modify the code so that when training, all distances
        # between A and B are precomputed, where A is in the training set and B
        # is in the cross-validation set. Currently there is an implicit loop
        # along each element of the CV set.
        diffs = self.X_train - X  # (m, n) array
        squared_dists = np.sum(diffs**2, axis=1)  # (m, ) array
        idx = np.argsort(squared_dists)[:k]  # k indices of smallest dists

        if weighted:  # Calculate weighted average using distances as weights.
            # Add epsilon to avoid division by zero:
            weights = 1 / np.sqrt((squared_dists[idx] + 1e-6))
            y_hat = (np.dot(weights, self.Y_train[idx].flatten())
                     / weights.sum())
        else:  # Calculate simple average.
            y_hat = np.mean(self.Y_train[idx])
        return y_hat

    def _evaluate_loss(self, X: np.ndarray, y: float,
                       k: int, weighted: bool = False) -> float:
        """
        Evaluates the squared loss for a single sample and its target value,
        using the k nearest neighbors.

        Parameters:
            * X: The input sample.
            * y: The true target value for the sample.
            * k: The number of nearest neighbors to consider.

        Returns:
            * The squared difference between the predicted and true target
              values.
        """
        y_hat = self._estimate(X, k, weighted)
        return (y_hat - y)**2

    def train(self) -> None:
        """
        Trains the regressor by selecting the best hyperparameters K and
        weighting method based on the lowest mean squared error on the
        cross-validation set.
        """
        best_K = self.K_min
        best_error = np.inf
        best_weighted = False

        for k in range(self.K_min, self.K_max + 1):
            errors = [self._evaluate_loss(X, y, k, weighted=False)
                      for X, y in zip(self.X_cv, self.Y_cv)]
            weighted_errors = [self._evaluate_loss(X, y, k, weighted=True)
                               for X, y in zip(self.X_cv, self.Y_cv)]
            mean_error = np.mean(errors)
            weighted_mean_error = np.mean(weighted_errors)

            if mean_error < best_error:
                best_K = k
                best_error = mean_error
                best_weighted = False

            if weighted_mean_error < best_error:
                best_K = k
                best_error = weighted_mean_error
                best_weighted = True

        self.K = best_K
        self.weighted = best_weighted

    def compute_MSE(self, X_test: np.ndarray, Y_test: np.ndarray) -> float:
        """
        Computes the mean squared error of the regressor on a given test
        dataset.

        Parameters:
            * X_test (2D (m, n) array): Test features, where 'm' is the number
            of test samples and 'n' is the number of features.
            * Y_test (1D (m,) array): True target values for the test samples.

        Returns:
            * MSE (float): The mean squared error of the predictions.
        """
        Y_test = Y_test.reshape(-1, 1)  # Ensure Y_test is a 2D column vector
        errors = [(self.predict(X) - y)**2 for X, y in zip(X_test, Y_test)]
        MSE = np.mean(errors)
        return MSE

    def predict(self, X: np.ndarray) -> float:
        """
        Predicts the target value for a single sample X using the optimal K and
        weighting method found during training.

        Parameters:
            * X: The input sample.

        Returns:
            * The predicted target value y_hat.
        """
        # Ensure X is a 2D array with a single sample if it's not already:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._estimate(X, self.K, self.weighted)
