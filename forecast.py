
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from numpy import array
import numpy as np
import time


#-----------------------------------------------------------------------
# convert time series into supervised learning problem
#-----------------------------------------------------------------------
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	varname = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']
	print 'Variables: ', varname
	print 'Variable to be forecasted: ', varname[0] 
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('%s(t-%d)' % (varname[j], i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('%s(t)' % varname[j]) for j in range(n_vars)]
		else:
			names += [('%s(t+%d)' % (varname[j], i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

 
#-----------------------------------------------------------------------
# transform series into train and test sets for supervised learning
#-----------------------------------------------------------------------
def prepare_data(series, n_in, n_out, train_frac, n_days):

    # extract raw values
    values = dataset.values
    
    if n_days > 0:
        values = values[:n_days*24, :]
    
    n_var = values.shape[1]
    
    print values.shape[0], 'data points'

    # The 4th column of 'values' (wind direction) is encoded as integers.
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])

    # ensure all data is float
    values = values.astype('float32')

    # normalize features (i.e., restrict values to be between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_in, n_out)

    # rearrange columns so columns we want to predict are at the end
    l = []
    for i in range(n_out):
        j = n_in*n_var+(n_var-1)*i  # pm2.5(t+i)
        l = list(reframed.columns[:j])
        l += list(reframed.columns[j+1:])
        l += list(reframed.columns[[j]])
        reframed = reframed.reindex_axis(l, axis=1)

##    print(reframed.head())

    # split into train and test sets
    values = reframed.values
    n_train_hours = int(values.shape[0] * train_frac)

    train = values[:n_train_hours, :]  # select first n_train_hours entries
    test = values[n_train_hours:, :]   # select remaining entries

    print('Training hours: %i ' % train.shape[0])
    print('Test hours: %i ' % test.shape[0])

    return scaler, train, test, n_var


#-----------------------------------------------------------------------
# fit an LSTM network to training data
#-----------------------------------------------------------------------
def fit_lstm(train, n_in, n_out, n_batch, nb_epoch, n_neurons, lstmStateful):

    # split into input (X) and output (y)
    X, y = train[:, :-n_out], train[:, -n_out:]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # stacked layers?
    returnSeq = False
    if len(n_neurons) >= 2:
        returnSeq = True

    # design network
    model = Sequential()
    
    # 1st LSTM layer:
    model.add(LSTM(n_neurons[0], return_sequences=returnSeq, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=lstmStateful))
    # more LSTM layers:
    for k in range(1,len(n_neurons)):
        if k < len(n_neurons)-1:
            model.add(LSTM(n_neurons[k], return_sequences=True))
        else:
            model.add(LSTM(n_neurons[k], return_sequences=False))        

    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    print('LSTM Layers: %i' % len(n_neurons))
    print 'Neurons:', n_neurons
    print('Batch size: %i' % n_batch)
    print('Training LSTM model ...')

    # fit network
    start_time = 0
    for i in range(nb_epoch):
        if i == 1: 
            start_time = time.time()
        j = i + 1
        print("Epoch {0}/{1}".format(j, nb_epoch))
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()

    t = (time.time() - start_time) / nb_epoch
    print("%.1f seconds/epoch" % t)
    print 'Training completed'

    return model
	

#-----------------------------------------------------------------------
# make one forecast with an LSTM
#-----------------------------------------------------------------------
def make_single_forecast(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]


#-----------------------------------------------------------------------
# make forecasts for entire test set
#-----------------------------------------------------------------------
def make_forecasts(model, n_batch, test, n_in, n_out):
    print 'Forecasting test data ...'
    forecasts = list()
    for i in range(len(test)):
        # select input
        X = test[i, :-n_out]
        # make forecast
        forecast = make_single_forecast(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)		
    return forecasts


#-----------------------------------------------------------------------
# inverse data transform
#-----------------------------------------------------------------------
def inverse_transform(normalized, scaler, n_var):
    inverted = list()
    for i in range(len(normalized)):
        norm_i = array(normalized[i])
        inv_i = list()
        for j in range(len(norm_i)):
            # Create numpy array with suitable dimensions to perform 
            # inverse transform.
            # Set 0th entry equal to pm2.5 while all other entries 
            # can be set to zero since we do not care about them.
            norm_ij = np.zeros((1, n_var))
            norm_ij[0][0] = norm_i[j]
            # invert scaling
            inv_scale = scaler.inverse_transform(norm_ij)
            inv_scale = inv_scale[0, :]
            inv_i.append(inv_scale[0])

        # store
        inverted.append(inv_i)

    return inverted


#-----------------------------------------------------------------------
# evaluate the RMSE for each forecast time step
#-----------------------------------------------------------------------
def evaluate_forecasts(test, forecasts, n_out):
    for i in range(n_out):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse_lstm = sqrt(mean_squared_error(actual, predicted))

        persistence = [row[i] for row in test]
        for j in range(i+1):
            actual.pop(0)   # remove 1st item
            persistence.pop() # remove last item
        rmse_persist = sqrt(mean_squared_error(actual, persistence))

        print('t+%d RMSE: %f [%f]' % ((i+1), rmse_lstm, rmse_persist))


#-----------------------------------------------------------------------
# plot the forecasts in the context of the original dataset
#-----------------------------------------------------------------------
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(series.values[:, 0], label='data')
    # plot the forecasts in red
    for i in range(len(forecasts[0])):
        if (i > 2) and (i < len(forecasts[0])-1): 
            continue
                    
        lab = 't+' + str(i+1)
        col = 'red'
        if i == 0: 
            col = 'black'
        elif i == 1: 
            col = 'green'
        elif i == 2: 
            col = 'orange'
        
        off_s = len(series) - n_test - len(forecasts[0]) + i + 1
        off_e = off_s + n_test
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [row[i] for row in forecasts]
        yaxis = yaxis[:len(xaxis)]
        pyplot.plot(xaxis, yaxis, label=lab, color=col)

    # show the plot
    pyplot.legend()
    pyplot.show()
						


# configure
n_lag = 6
n_forecast = 24
n_epochs = 1000
n_batch = 500
lstmStateful = False
n_neurons = [50,50,50,50]
train_fraction = 0.33
n_days = -1   # -1 will process entire data set

# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)

# OBS: Value to be forecasted must be in 1st column

# prepare data
scaler, train, test, n_variables = prepare_data(dataset, n_lag, n_forecast, train_fraction, n_days)

print 'Lag:', n_lag
print 'Forecast: ', n_forecast

# fit model
model = fit_lstm(train, n_lag, n_forecast, n_batch, n_epochs, n_neurons, lstmStateful)

# make forecast
forecasts = make_forecasts(model, n_batch, test, n_lag, n_forecast)

# actual values
actual = [row[-n_forecast:] for row in test]

# inverse transform
print 'Inverse transform of forecast and test data ...'
forecasts = inverse_transform(forecasts, scaler, n_variables)
actual = inverse_transform(actual, scaler, n_variables)

# evaluate forecast quality
print 'Performance LSTM [Persistence]:'
evaluate_forecasts(actual, forecasts, n_forecast)

# plot forecasts
plot_forecasts(dataset, forecasts, test.shape[0])

