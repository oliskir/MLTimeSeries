
import heat_forecast as hf

inputfile = '../../heat_load_weather_calendar.csv'
n_lag = 24
n_forecast = 24
n_neurons = [40]
n_epochs = 10
n_batch = 365
train_fraction = 0.33
predictChange = False
validate = True
cheat = True
verbosity = 2

hf.run(inputfile, n_lag, n_forecast, n_neurons, n_epochs, n_batch, train_fraction, predictChange, validate, cheat, verbosity)
