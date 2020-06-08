class Score:
    def __init__(self, labels, preds):
        P = (labels == 1).sum()
        N = (labels == 0).sum()
        
        TP = (preds[labels == 1] == 1).sum()
        FN = (preds[labels == 1] == 0).sum()
        FP = (preds[labels == 0] == 1).sum()
        TN = (preds[labels == 0] == 0).sum()
        
        self.TPR = TP / (TP + FN)
        self.FPR = FP / (FP + TN)
        self.TNR = TN / (TN + FP)
        self.FNR = FN / (FN + TP)

        self.PPV = TP / (TP + FP)
        self.NPV = TN / (TN + FN)

        self.accuracy = (TP + TN) / (P + N)
        self.F1Score = 2 * TP / (2 * TP + FP + FN)

        self.sensitivity = self.TPR
        self.specificity = self.TNR
        self.precision = self.PPV
        