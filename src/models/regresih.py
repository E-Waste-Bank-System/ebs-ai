import pandas as pd

class Regresih:
    def __init__(self, enc1, enc2, model, y_scaler):
        self.enc1     = enc1
        self.enc2     = enc2
        self.model    = model
        self.y_scaler = y_scaler

    def predict(self, X_df):
        # 1) encode features exactly as you did manually
        X_new = X_df.copy()
        X_new["Kondisi"]    = self.enc1 .transform(X_new[["Kondisi"]])
        X_new["Nama Item"]  = self.enc2 .transform(X_new["Nama Item"])
        # 2) predict on your already-trained model
        y_scaled = self.model.predict(X_new)
        # 3) inverse-scale to original units