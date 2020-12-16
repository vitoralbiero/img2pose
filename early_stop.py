class EarlyStop:
    def __init__(self, patience=5, mode="max", threshold=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.mode = mode
        self.threshold = threshold
        self.val_score = float("Inf")
        if mode == "max":
            self.val_score *= -1

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score

        # if val score did not improve, add to early stop counter
        elif (val_score < self.best_score + self.threshold and self.mode == "max") or (
            val_score > self.best_score + self.threshold and self.mode == "min"
        ):
            self.counter += 1
            print(f"Early stop counter: {self.counter} out of {self.patience}")

            # if not improve for patience times, stop training earlier
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = val_score
            self.counter = 0
