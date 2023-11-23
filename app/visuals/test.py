import pandas as pd
from pca import PCA_Analysis
from mapping import Mapping

# Loading the data
data_path = "~/MinexPy/Data.csv"
df = pd.read_csv(data_path)

input_elements = input("\n For what columns should the geochemical processes be done? "
                       "\n Please use 'space' as a separator. \n")

# Zn Pb Ag Cu Mo Cr Ni Co Ba

# Create an instance of PCA_Analysis
pca = PCA_Analysis(input_elements)

# Process the DataFrame with PCA
processed_df = pca.create_pc_df(df)  # Assuming create_pc_df is the method to process the DataFrame

# Pass the processed DataFrame to mapping

m = Mapping(input_elements, processed_df, "X", "Y", "linear", "viridis", "Map of ")
m.create_maps()
