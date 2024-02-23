# Type:
#   * Supervised learning
#   * Prediction
#   * Regression
# -----------------------
# Description:
# Given datasets X and Y consisting of single numerical observations
# x_1, ..., x_m and y_1, ..., y_m, respectively, this algorithm learns the best
# linear model Y = alpha + beta * X according to the least squares criterion.

# More precisely, the algorithm estimates the parameters alpha (intercept) and
# beta (slope) by minimizing the Residual Sum of Squares (RSS) between the
# observed values Y and the predicted values Y_hat over the training data.
# (The notation beta/alpha is meant to remind one of the beta of an equity.)
#
# Performance metrics include Residual Standard Error (RSE), the correlation
# (Pearson) coefficient r and the R^2 statistic.
#
# This implementation offers training (learn), prediction (predict), and
# evaluation of regression diagnostics (regression_diagnostics).


import numpy as np


class SimpleLinearRegression:
    """
    Simple Linear Regression model that fits a line to the input data using
    the least squares method.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Initializes the regressor with training data.

        Parameters:
            * X (2D (m, 1) array): Training features.
            * Y (2D (m, 1) array): Training target values.
        """
        self.X = X
        self.Y = Y
        self.m = X.shape[0]  # Number of training examples
        self.alpha = 0  # Intercept of least squares line
        self.beta = 0  # Slope of least squares line
        # Initialize placeholders for statistics:
        self.x_bar = 0  # Mean of X
        self.y_bar = 0  # Mean of Y
        self.var_x = 0  # Sample variance of X
        self.covariance = 0  # Covariance of X and Y
        self.Y_hat = None  # Predicted Y values
        self.RSS = 0  # Residual Sum of Squares
        self.RSE = 0  # Residual Standard Error
        self.std_error_beta = 0  # Standard Error of the beta estimate

    def learn(self):
        """
        Learns the parameters of the linear regression model by computing the
        slope and intercept using the least squares method.
        """
        self.x_bar = np.mean(self.X)
        self.y_bar = np.mean(self.Y)
        self.var_x = np.var(self.X, ddof=1)  # Sample variance
        # Compute covariance between X and Y:
        self.covariance = np.cov(self.X.T, self.Y.T)[0, 1]
        self.beta = self.covariance / self.var_x
        self.alpha = self.y_bar - self.beta * self.x_bar
        # Predict the Y values:
        self.Y_hat = self.predict(self.X)
        # Compute statistics:
        self._compute_statistics()

    def _compute_statistics(self):
        """
        Computes various statistics to evaluate the regression model.
        """
        residuals = self.Y - self.Y_hat
        self.RSS = np.sum(residuals**2)
        self.RSE = np.sqrt(self.RSS / (self.m - 2))
        self.std_error_beta = (self.RSE
                               / np.sqrt(np.sum((self.X - self.x_bar)**2)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for a given set of features X using the
        learned linear model.

        Parameters:
            * X (2D (m, 1) array): Features for which to predict target values.

        Returns:
            * Y_hat (2D (m, 1) array): Predicted target values.
        """
        return self.beta * X + self.alpha

    def regression_diagnostics(self) -> dict:
        """
        Returns important regression diagnostics and statistics for both
        the independent variable (X) and dependent variable (Y).

        Returns:
            * A dictionary containing RSS, RSE, standard error of beta,
                R^2, mean, variance, and standard deviation for both X and Y.
        """
        R_squared = 1 - (self.RSS / np.sum((self.Y - self.y_bar)**2))
        diagnostics = {
            "RSS": self.RSS,
            "RSE": self.RSE,
            "Standard Error of Beta": self.std_error_beta,
            "R^2": R_squared,
            "X Mean": self.x_bar,
            "X Variance": self.var_x,
            "X Standard Deviation": np.sqrt(self.var_x),
            "Y Mean": self.y_bar,
            "Y Variance": np.var(self.Y, ddof=1),
            "Y Standard Deviation": np.sqrt(np.var(self.Y, ddof=1))
        }
        return diagnostics

    def print_diagnostics(self):
        """
        Pretty prints the regression diagnostics and statistics.
        """
        diagnostics = self.regression_diagnostics()
        print("Regression Diagnostics and Statistics:")
        print("--------------------------------------")
        for key, value in diagnostics.items():
            print(f"{key}: {value:.4f}")
