# Import pandas, numpy, and sklearn modules
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Define a class for PCA analysis
class PCA_Analysis:
    
    # Initialize the class with the element concentration string
    def __init__(self, element_concentration):
        self.element_concentration = element_concentration

    # Define a method to generate the loadings dataframe
    def get_loadings_df(self, df):
        
        # Extract the element columns from df
        splitted = self.element_concentration.split()
        ele = df.loc[:, splitted].values

        # Create a list of column names for the loadings dataframe
        column_names_PCA = []
        for i in range(1, len(splitted) + 1):
            column_names_PCA.append(f"LPC{i}")

        # Perform PCA on the element values
        pca = PCA(n_components=len(splitted))
        principalComponents = pca.fit_transform(ele)

        # Create the loadings dataframe and round the values to 3 decimals
        loadings_df = pd.DataFrame(np.round(pca.components_.T, 3), columns=column_names_PCA)

        # Return the loadings dataframe
        return loadings_df

    # Define a method to create the final pc dataframe
    def create_pc_df(self, df):
        
        # Get the loadings dataframe from the previous method
        loadings_df = self.get_loadings_df(df)

        # Extract the element columns from df
        element_cols = self.element_concentration.split()
        elements_df = df[element_cols]

        # Convert the dataframes to numpy arrays
        elements_array = elements_df.to_numpy()
        loadings_array = loadings_df.to_numpy()

        # Multiply the element values with the LPC values
        pc_values = elements_array.dot(loadings_array)

        # Convert the result back to a dataframe and rename the columns
        pc_values = pd.DataFrame(pc_values)

        # Assign the column names to the pc dataframe
        pc_values.columns = element_cols

        # Concatenate the X and Y columns from df with the pc_values
        pc_df = pd.concat([df[["X", "Y"]], pc_values], axis=1)

        # Return the final dataframe
        return pc_df
