# Type:
#   * Supervised learning
#   * Prediction
#   * Classification
# -----------------------
# Description:
# TODO: Write description
# TODO: Write docstrings.


import numpy as np


class SimpleLogisticRegression:
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000,
                 threshold: float = 0.5, verbose: bool = False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.threshold = threshold
        self.verbose = verbose  # Controls whether progress is printed
        self.alpha = 0.0  # Intercept
        self.beta = 0.0   # Slope

    def _sigmoid(self, Z):
        """Compute the sigmoid of Z."""
        return 1 / (1 + np.exp(-Z))

    def _cross_entropy_loss(self, Y, P):
        """Compute the cross-entropy loss."""
        return -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))

    def _partial_alpha(self, Y, P):
        """Compute the partial derivative of the loss with respect to alpha."""
        return -np.mean(Y - P)

    def _partial_beta(self, X, Y, P):
        """Compute the partial derivative of the loss with respect to beta."""
        return -np.mean((Y - P) * X)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """Fit the logistic regression model to the training data."""
        for iteration in range(self.iterations):
            Z = self.alpha + self.beta * X_train
            P = self._sigmoid(Z)
            self.alpha -= (self.learning_rate
                           * self._partial_alpha(Y_train, P))
            self.beta -= (self.learning_rate
                          * self._partial_beta(X_train, Y_train, P))

            if self.verbose and iteration % 100 == 0:
                loss = self._cross_entropy_loss(Y_train, P)
                print(f"Iteration {iteration}: Loss = {loss}")

    def predict_probs(self, X):
        """Compute the probability estimates for X."""
        return self._sigmoid(self.alpha + self.beta * X)

    def predict_labels(self, X):
        """Predict class labels for samples in X."""
        P = self.predict_probs(X)
        return (P >= self.threshold).astype(int)
