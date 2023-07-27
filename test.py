from utils import DataImporter
#from analysis import DescriptiveStatistics, Correlation, StatPlots, input_elements
#from app.visuals import Gridding
from app import input_elements, input_interpol, input_X, input_Y
from app.visuals import Mapping

data_importer = DataImporter('/Users/yasin/Desktop/ProjectX/Data.csv')
cleaned_data = data_importer.get_cleaned_data()

#print(cleaned_data)

columns = input_elements.split()

#a = StatPlots(data=cleaned_data, columns=columns)
#a.plot_stats()

#stats = DescriptiveStatistics(data=cleaned_data ,columns=columns)
#print(stats.basic_stats_df())

#c = Correlation(data=cleaned_data, element_concentration=columns)  # create Correlation object with input_elements parameter
#print(c.correlation_iterated())  # call correlation_iterated method to calculate correlations between all columns in input_elements parameter

#b = Gridding(raw_data=cleaned_data,input_X=input_X, input_Y=input_Y,input_interpol=input_interpol)
#print(b.interpolate_column)

m = Mapping(cleaned_data, "X", "Y", "linear", "viridis", "Map of ")
m.create_maps_parallel()


