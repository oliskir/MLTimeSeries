
from pandas import DataFrame
from pandas import concat
from pandas import to_datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from numpy import array
import numpy as np
import time
import csv



#-----------------------------------------------------------------------
# convert time series into supervised learning problem
#-----------------------------------------------------------------------
def series_to_supervised(data, hours, varname, n_in=1, n_out=1, n_lead=0, t0_forecast=-1, dropnan=True):
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
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (varname[j], i)) for j in range(n_vars)]
    # forecast sequence (t+lead, t+lead+1, ... t+lead+n)
    for i in range(n_lead, n_lead + n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % varname[j]) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (varname[j], i)) for j in range(n_vars)]	
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # add hour column
    dfh = DataFrame()
    dfh['hour'] = hours
    agg['hour'] = dfh.shift(1).values
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    # if we want to make 1 daily forecast at a specific hour 
    # (instead of hourly forecasts) we have to select the relevant rows
    t0 = t0_forecast - n_lead - 1
    if (t0 < 0): 
        t0 += 24
    agg0 = agg[agg.hour == t0]
    # drop hour column again
    agg0 = agg0.drop('hour', 1)
        
    return agg0

 
#-----------------------------------------------------------------------
# transform series into train and test sets for supervised learning
#-----------------------------------------------------------------------
def prepare_data_for_training(series, n_in, n_out, n_lead, t0_forecast, n_split, n_days, ignoredVar, predictChange, rangeBuffer, logfile):

    # extract raw values
    values = series.values
    
    if n_days > 0:
        values = values[:n_days*24, :]

    # number of data points
    N = values.shape[0]
    
    # hour column
    hours = values[:, 8]

    # ensure all data is float
    values = values.astype('float32')

    # variable names
    variableNames = list(series.columns.values)
    
    # remove ignored variables
    values = np.delete(values, ignoredVar, 1)
    for j in range(len(ignoredVar)-1, -1, -1):
        i = ignoredVar[j]
        del variableNames[i]
        
    # add rate-of-change variable
    if predictChange:
        variableNames.insert(0, 'Dprod')        
        values = np.c_[ np.zeros(N), values ]
        for i in range(1, N-1):
            values[i][0] = values[i][1] - values[i-1][1]
        values = np.delete(values, N-1, axis=0)
        values = np.delete(values, 0, axis=0)
        N = values.shape[0]

    # number of variables    
    n_var = values.shape[1]

    # normalize features (i.e., restrict values to be between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0.+rangeBuffer, 1.-rangeBuffer))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, hours, variableNames, n_in, n_out, n_lead, t0_forecast)

    # rearrange columns so the columns we want to predict are at the end
    l = []
    for i in range(n_out):
        j = n_in * n_var + (n_var - 1) * i  # prod(t+i)
        l = list(reframed.columns[:j])
        l += list(reframed.columns[j+1:])
        l += list(reframed.columns[[j]])
        reframed = reframed.reindex_axis(l, axis=1)
        
###    print(reframed.head())

    # if predicting change, calculate deviation relative to prod(t).
    # (add 1 and divide by 2 to ensure that deviation is between 0 and 1)
    if predictChange:    
        key_0 = reframed.columns[(n_in - 1) * n_var]
        n = reframed.shape[1]
        for i in range(n-1, n-n_out-1, -1):
            key = reframed.columns[i]
            reframed[key] = reframed[key] - reframed[key_0]
            reframed[key] = reframed[key]+1
            reframed[key] = reframed[key]*0.5

    # split into train and test sets
    values = reframed.values
    ss = int(values.shape[0] / n_split) # sample size

    trains, tests, tests_index = list(), list(), list()
    for i in range(n_split):
        i1 = i*ss
        i2 = (i+1)*ss - 1
        if (i2 > values.shape[0] - 1):
            i2 = values.shape[0] - 1
        test = values[i1:i2, :]
        tests.append(test)
        test_index = reframed.index[i1:i2]
        tests_index.append(test_index)
        indices = [i for i in range(i1,i2)]
        train = np.delete(values,indices,axis=0)
        trains.append(train)
        
    # save config data to file
    line = ' ' + str(N) + ' time steps'
    logfile.write(line + '\n')
    line = ' Variables: ' + ', '.join(variableNames)
    logfile.write(line + '\n')
    line = ' Variable to be forecasted: ' + str(variableNames[0])    
    logfile.write(line + '\n')
    line = ' Training samples: ' + str(train.shape[0])        
    logfile.write(line + '\n')
    line = ' Test samples: ' + str(test.shape[0])        
    logfile.write(line + '\n')

    return scaler, trains, tests, tests_index, n_var


#-----------------------------------------------------------------------
# split into input (X) and output (y) columns
#-----------------------------------------------------------------------
def split_into_Xy(data, n, cheat):
    
    # train
    y = data[:, -n:]
    if cheat:
        X = data 
    else:
        X = data[:, :-n] 
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, y


#-----------------------------------------------------------------------
# fit an LSTM network to training data
#-----------------------------------------------------------------------
def fit_lstm(trains, n_lag, n_out, n_batch, nb_epoch, n_neurons, lstmStateful, validate, tests, cheat, figname, verbosity, n_seed):

    # split into input (X) and output (y)
    train = trains[0]
    test  = tests[0]
    X, y = split_into_Xy(train, n_out, cheat)

	# fix random seed for reproducibility
    if (n_seed > 0): 
        np.random.seed(n_seed)

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

    print('Training LSTM model ...')

    # fit network
###    start_time = 0
###    for i in range(nb_epoch):
###        if i == 1: 
###            start_time = time.time()
###        j = i + 1
###        print("Epoch {0}/{1}".format(j, nb_epoch))
###        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
###        model.reset_states()

    start_time = time.time()
    forecasts = list()

    # loop over splits
    loss, val_loss = list(), list()
    for i in range(len(trains)):

        print 'Split:',i+1

        train = trains[i]
        test  = tests[i]        
        
        # split into input (X) and output (y)
        X, y = split_into_Xy(train, n_out, cheat)
        X_test, y_test = split_into_Xy(test, n_out, cheat)

        # train
        if validate:
            history = model.fit(X, y, epochs=nb_epoch, batch_size=n_batch, validation_data=(X_test, y_test), verbose=verbosity, shuffle=False)
        else:
            history = model.fit(X, y, epochs=nb_epoch, batch_size=n_batch, verbose=verbosity, shuffle=False)

        loss.extend(history.history['loss'])
        if validate:
            val_loss.extend(history.history['val_loss'])
            
        # make forecast
        print 'Forecasting ...'
        f = make_forecasts(model, n_batch, test, n_lag, n_out, cheat)
        forecasts.extend(f)

    # plot history
    ax = pyplot.plot(loss, label='train')
    if validate:
        pyplot.plot(val_loss, label='test')
    pyplot.yscale('log')
    pyplot.legend()
    pyplot.grid(b=True, which='both')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.title('Training history')
    pyplot.savefig(figname, bbox_inches='tight')
    pyplot.clf()

    duration = time.time() - start_time
    timePerEpoch = duration / nb_epoch

    print("%.1f seconds" % duration)
    print("%.1f seconds/epoch" % timePerEpoch)
    print 'Training completed'

    return model, timePerEpoch, forecasts
	

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
def make_forecasts(model, n_batch, test, n_in, n_out, cheat):
    forecasts = list()
    for i in range(len(test)):
        # select input
        if cheat:
            X = test[i, :]
        else:
            X = test[i, :-n_out]
        # make forecast
        forecast = make_single_forecast(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)		
    return forecasts


#-----------------------------------------------------------------------
# inverse data transform
#-----------------------------------------------------------------------
def inverse_transform(normalized, scaler, n_var, predictChange, baseline):

    # scaling parameters for forecasted quantity
    beta = scaler.min_[0]
    alpha = scaler.scale_[0]

    # invert scaling
    inverted = list()
    for i in range(len(normalized)):
        norm_i = array(normalized[i])
        inv_i = list()
        for j in range(len(norm_i)):
            yn = norm_i[j]
            if predictChange:
                yn = 2 * yn - 1
                yn = yn + baseline[i]
            y = (yn - beta) / alpha
            inv_i.append(y)

        # store
        inverted.append(inv_i)

    return inverted


#-----------------------------------------------------------------------
# evaluate the RMSE for LSTM model and simple persistence model
#-----------------------------------------------------------------------
def evaluate_forecasts(actuals, forecasts, baseline, n_out, logfile):

    # collapse into a 1d list
    actual, forecast = list(), list()
    for i in range(n_out):
        actual.extend([row[i] for row in actuals])
        forecast.extend([row[i] for row in forecasts])

    persist = list()
    for i in range(len(baseline)):
        b = baseline[i]
        for i in range(n_out):
            persist.append(b)
        
    rmse_lstm = sqrt(mean_squared_error(actual, forecast))

    rmse_persist = sqrt(mean_squared_error(actual, persist))

    line = ' RMSE(LSTM): %f' % rmse_lstm
    logfile.write(line + '\n')
    line = ' RMSE(persist): %f' % rmse_persist
    logfile.write(line + '\n')
    
    return rmse_lstm
