import pandas as pd
from mapping import Mapping

df = pd.read_csv("~/Desktop/MinexPy/MinexPy/Data.csv")

m = Mapping(df, "X", "Y", "linear", "viridis", "Map of ")
m.create_maps_parallel()

