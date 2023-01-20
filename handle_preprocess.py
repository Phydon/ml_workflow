import joblib
import numpy as np


def save_preprocessed(
    train_data,
    train_labels,
    test_data,
    test_labels,
    model
):
    joblib.dump(train_data, "./traindata.joblib", compress=True)
    joblib.dump(train_labels, "./trainlabels.joblib", compress=True)
    joblib.dump(test_data, "./testdata.joblib", compress=True)
    joblib.dump(test_labels, "./testlabels.joblib", compress=True)
    joblib.dump(model, "./model.joblib", compress=True)


def load_preprocessed():
    train_data = joblib.load("./traindata.joblib")
    train_labels = joblib.load("./trainlabels.joblib")
    test_data = joblib.load("./testdata.joblib")
    test_labels = joblib.load("./testlabels.joblib")
    model = joblib.load("./model.joblib")

    return (train_data, train_labels), (test_data, test_labels), model


def test_save_preprocessed():
    train_data = np.random.rand(5, 5)
    train_labels = np.random.rand(5, 5)
    test_data = np.random.rand(5, 5)
    test_labels = np.random.rand(5, 5)

    model = np.dot(train_data, train_labels)

    save_preprocessed(train_data, train_labels, test_data, test_labels, model)


def test_load_preprocessed():
    (train_data, train_labels), (test_data, test_labels), model = load_preprocessed()
    print(train_data)
    print(train_labels)
    print(test_data)
    print(test_labels)
    print(model)
    

if __name__ == "__main__":
    test_save_preprocessed()
    test_load_preprocessed()
    