from mlflow import pyfunc


class RevClassifier(object):

    def __init__(self):
        self.pyfunc_model = pyfunc.load_pyfunc("mlruns/0/abc/artifacts/model")

    def predict(self, X):
        output = self.pyfunc_model(X)
        return output