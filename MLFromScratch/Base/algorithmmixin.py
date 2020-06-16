class AlgorithmMixin():
    def __init__(self):
        pass

    
    def fit(self, X, y):
        raise NotImplementedError()


    def predict(self, X):
        raise NotImplementedError()


    def score(self, X, y):
        preds = self.predict(X)
        return Score(y, preds)