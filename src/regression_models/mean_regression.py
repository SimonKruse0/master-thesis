import numpy as np

class MeanRegression:
    def __init__(self, extra_name="") -> None:
        self.name = f"empirical mean and std regression{extra_name}"
        self.latex_architecture = r""
        self.model = None
        self.params = f"mean and std"

    def fit(self, X, Y):
        self.mean = np.mean(Y, axis=0)
        self.std = np.std(Y, axis=0)

    def predict(self, X_test):
        N,_ = X_test.shape
        return np.repeat(self.mean, N), np.repeat(self.std, N), None

if __name__ == "__main__":
    pass