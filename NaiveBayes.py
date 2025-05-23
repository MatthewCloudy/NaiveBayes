import math
import numpy as np
from scipy.special import gamma


class NaiveBayes:
    def __init__(
        self,
    ):
        self.mean = None
        self.var = None
        self.classes = None
        self.classes_count = None
        self.classes_prob = None

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.classes_count = dict.fromkeys(self.classes, 0)
        self.classes_prob = dict.fromkeys(self.classes, 0)
        self.mean = dict.fromkeys(self.classes, 0.0)
        self.var = dict.fromkeys(self.classes, 0.0)

        class_attributes = {c: [] for c in self.classes}
        for i in range(X_train.shape[0]):
            self.classes_count[y_train[i]] += 1
            class_attributes[y_train[i]].append(X_train[i])
        for c in self.classes:
            self.classes_prob[c] = self.classes_count[c] / X_train.shape[0]
            self.mean[c] = np.mean(np.array(class_attributes[c]), axis=0)
            self.var[c] = np.var(np.array(class_attributes[c]), axis=0)

    def predict(self, X, distribution):
        predictions = []
        for i in range(X.shape[0]):
            prediction_value = {}
            for c in self.classes:
                prediction_value[c] = self.classes_prob[c]
                if distribution == "gaussian":
                    for j in range(X.shape[1]):
                        prediction_value[c] *= (
                            1 / math.sqrt(2 * math.pi * self.var[c][j])
                        ) * (
                            math.exp(
                                -((X[i, j] - self.mean[c][j]) ** 2)
                                / (2 * self.var[c][j])
                            )
                        )
                else:
                    for j in range(X.shape[1]):
                        factor = (self.mean[c][j] * (1 - self.mean[c][j])) / self.var[
                            c
                        ][j] - 1
                        alpha = self.mean[c][j] * factor
                        beta = (1 - self.mean[c][j]) * factor
                        B = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
                        prediction_value[c] *= (
                            X[i, j] ** (alpha - 1) * (1 - X[i, j]) ** (beta - 1)
                        ) / B

            predictions.append(max(prediction_value, key=prediction_value.get))
        return predictions
