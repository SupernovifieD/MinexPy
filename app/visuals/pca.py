import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from standardization import Standardizer  # Make sure to import the Standardizer class

class PCAnalysis:
    def __init__(self, filename):
        self.filename = filename

    def perform_pca(self, element_concentration, standardized_data):
        splitted = element_concentration.split()
        standardized_data = Standardizer(self.data).standardize()  # Assuming standardize() returns a DataFrame

        if not set(splitted).issubset(standardized_data.columns):
            raise ValueError("One or more specified elements are not in the dataset.")

        ele = standardized_data.loc[:, splitted].values

        pca = PCA(n_components=len(splitted))
        principal_components = pca.fit_transform(ele)

        column_names_PCA = [f"PCA{i}" for i in range(1, len(splitted) + 1)]
        loadings = pd.DataFrame(np.round(pca.components_.T, 3), columns=column_names_PCA, index=splitted)

        return loadings, principal_components

    def process_dataframe(self, element_concentration, loadings, principal_components):
        """
        Process and transform the DataFrame based on PCA results.
        This method will add the PCA components to the original data.
        """     
        splitted = element_concentration.split()

        # Create a DataFrame for the principal components
        column_names_PCA = [f"PCA{i}" for i in range(1, len(splitted) + 1)]
        pca_df = pd.DataFrame(principal_components, columns=column_names_PCA)

        # Reset index of original data if necessary
        # This is to ensure that indices match when concatenating
        self.data.reset_index(drop=True, inplace=True)

        # Concatenate the PCA components with the original data
        # Assuming that you want to keep all original columns along with the new PCA columns
        combined_df = pd.concat([self.data, pca_df], axis=1)

        return combined_df


    def analyze(self, element_concentration):
        loadings, principal_components = self.perform_pca(element_concentration)
        processed_df = self.process_dataframe(element_concentration, loadings)

        return loadings, processed_df
