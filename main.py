from NaiveBayes import NaiveBayes
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns


def load_split_data():
    iris = fetch_ucirepo(id=53)

    X = iris.data.features.to_numpy()
    y = iris.data.targets.to_numpy().ravel()

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min)

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


def test_NB(distribution):
    X_train, y_train, X_test, y_test = load_split_data()

    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test, distribution)
    print(y_pred)
    print(y_test)
    print("Dokładność:", accuracy_score(y_test, y_pred))

    print("Raport klasyfikacji:")
    print(classification_report(y_test, y_pred))
    labels = sorted(list(set(y_test)))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Macierz pomyłek:")
    print(cm)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Przewidziana klasa")
    plt.ylabel("Prawdziwa klasa")
    plt.title("Macierz pomyłek")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    test_NB("beta")
