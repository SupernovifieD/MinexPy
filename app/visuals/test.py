import pandas as pd
from mapping import Mapping

df = pd.read_csv("~/Desktop/MinexPy/MinexPy/Data.csv")

input_elements = input("\n For what columns should the geochemical processes be done? "
                       "\n Please use 'space' as a separator. \n")
# Zn Pb Ag Cu Mo Cr Ni Co Ba

m = Mapping(df, "X", "Y", "linear", "viridis", "Map of ")
m.create_maps()

