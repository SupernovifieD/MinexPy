import pandas as pd
from pca import PCAnalysis
from mapping import Mapping

pca = PCAnalysis("~/MinexPy/Data.csv")

input_elements = input("\n For what columns should the geochemical processes be done? "
                       "\n Please use 'space' as a separator. \n")
# Zn Pb Ag Cu Mo Cr Ni Co Ba

_, df, _ = pca.analyze(input_elements)

m = Mapping(input_elements, df, "X", "Y", "linear", "viridis", "Map of ")
m.create_maps()
