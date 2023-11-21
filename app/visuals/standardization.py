import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Standardizer:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data

    def standardize(self):
        # Assuming that all columns need to be standardized
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns available for standardization.")

        scaler = StandardScaler()
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        return self.data
