from NaiveBayes import NaiveBayes
from ucimlrepo import fetch_ucirepo
import numpy as np

def load_split_data():
    iris = fetch_ucirepo(id=53)

    X = iris.data.features.to_numpy()
    y = iris.data.targets.to_numpy().ravel()

    test_fraction = 0.2
    test_number = int(test_fraction * X.shape[0])
    indices = np.random.permutation(X.shape[0])
    test_indices = indices[:test_number]
    train_indices = indices[test_number:]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, y_train, X_test, y_test

def test_NB():
    X_train, y_train, X_test, y_test = load_split_data()

    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    print(nb.predict(X_test))
    print(y_test)
if __name__ == '__main__':
    test_NB()
