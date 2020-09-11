import numpy as np


class ScoreMulticlass:
    def __init__(self, labels, preds):
        EPS = 1e-10
        ulabels = np.unique(labels)
        self.F1Score = np.zeros((ulabels.shape))
        self.accuracy = np.zeros((ulabels.shape))
        self.recall = np.zeros((ulabels.shape))
        self.P = np.zeros((ulabels.shape))
        for i, l in enumerate(ulabels):
            P = (labels == l).sum()
            N = (labels != l).sum()

            TP = (preds[labels == l] == l).sum()
            FN = (preds[labels == l] != l).sum()
            FP = (preds[labels != l] == l).sum()
            TN = (preds[labels != l] != l).sum()

            self.P[i] = P
            self.F1Score[i] = 2 * TP / (2 * TP + FP + FN + EPS)
            self.accuracy[i] = (TP + TN) / (P + N + EPS)
            self.recall[i] = TP / (TP + FN + EPS)
        self.support = self.P

    def print_f1(self):
        print("F1 Score: ")
        print(self.F1Score)
        print((self.F1Score * self.P).sum() / self.P.sum())
        print("------------------------------------")


if __name__ == "__main__":
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])

    print(ScoreMulticlass(y_true, y_pred).F1Score)
    print(ScoreMulticlass(y_true, y_pred).P)
