
from pandas import read_csv

import subroutines as sub



# configure
n_lag = 24
n_forecast = 24
n_epochs = 10
n_batch = 365
lstmStateful = False
n_neurons = [40]
train_fraction = 0.33
n_days = -1   # -1 will process entire data set
ignoredVariables = [4, 5, 6, 7, 8, 9, 10]  # numbering begins with zero: 0,1,2...etc
predictChange = False
validate = True
cheat = True # values to be predicted are included in input
datafile = '../../heat_load_weather_calendar.csv' # OBS: Value to be forecasted must be in column 0


# load dataset
dataset = read_csv(datafile, header=0, index_col=0)

# prepare data
scaler, train, test, n_variables = sub.prepare_data(dataset, n_lag, n_forecast, train_fraction, n_days, ignoredVariables, predictChange)

print 'Lag:', n_lag
print 'Forecast:', n_forecast
if predictChange:
    print 'Forecast type: Relative'
else:
    print 'Forecast type: Absolute'
if cheat:
    print 'OBS: CHEATING ENABLED!'

# fit model
model = sub.fit_lstm(train, n_forecast, n_batch, n_epochs, n_neurons, lstmStateful, validate, test, cheat)

# make forecast
forecasts = sub.make_forecasts(model, n_batch, test, n_lag, n_forecast, cheat)

# actual values
actual = [row[-n_forecast:] for row in test]

# actual values at t-1
i0 = (n_lag - 1) * n_variables
baseline = [row[i0] for row in test]

# inverse transform
print 'Inverse transform of forecast and test data ...'
forecasts = sub.inverse_transform(forecasts, scaler, n_variables, predictChange, baseline)
actual = sub.inverse_transform(actual, scaler, n_variables, predictChange, baseline)

# evaluate forecast quality
print 'Performance LSTM [Persistence]:'
sub.evaluate_forecasts(actual, forecasts, n_forecast)

# plot forecasts
sub.plot_forecasts(dataset, forecasts, test.shape[0])

