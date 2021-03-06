
# This script loads the file pollution.csv and plots each 
# series as a separate subplot, except wind speed dir, which is 
# categorical.


from pandas import read_csv
from matplotlib import pyplot

# load dataset
dataset = read_csv('../../heat_load_weather_calendar.csv', header=0, index_col=0)
values = dataset.values

# specify columns to plot
groups = [0, 1, 2, 3]
i = 1

# plot each column
pyplot.figure(figsize=(10,10))
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()
