class Score:
    def __init__(self, labels, preds):
        EPS = 1e-10
        P = (labels == 1).sum()
        N = (labels == 0).sum()

        TP = (preds[labels == 1] == 1).sum()
        FN = (preds[labels == 1] == 0).sum()
        FP = (preds[labels == 0] == 1).sum()
        TN = (preds[labels == 0] == 0).sum()

        self.TPR = TP / (TP + FN + EPS)
        self.FPR = FP / (FP + TN + EPS)
        self.TNR = TN / (TN + FP + EPS)
        self.FNR = FN / (FN + TP + EPS)

        self.PPV = TP / (TP + FP + EPS)
        self.NPV = TN / (TN + FN + EPS)

        self.accuracy = (TP + TN) / (P + N + EPS)
        self.F1Score = 2 * TP / (2 * TP + FP + FN + EPS)

        self.sensitivity = self.TPR
        self.recall = self.TPR
        self.specificity = self.TNR
        self.precision = self.PPV
