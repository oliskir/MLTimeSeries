
from pandas import read_csv
import subroutines as sub
import datetime as dt
import os
from sklearn.externals import joblib
from keras.models import model_from_json

import rootoutput as ro


def parse_dates(x):
    return dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
    
def train(inputfile, n_lag, n_forecast, t0_make, t0_forecast, n_neurons, n_epochs, n_batch, n_split, validate, cheat, verbosity, seed, ignoredVariables):

    # configure
    n_lead = 24 - t0_make + 1 + t0_forecast
    lstmStateful = False
    n_days = -1   # set to -1 to process entire data set
    rangeBuffer = 0.05

    # create output directories if necessary
    try: 
        os.makedirs('output_train/model')
    except OSError:
        if not os.path.isdir('output_train/model'):
            raise
        
    # current date and time
    today = dt.datetime.now().strftime("%Y-%m-%d")
    now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # open log file
    logfile = open('output_train/'+now+'.log', 'w+')

    # save configuration data
    nowPretty = dt.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    line = ' ' + nowPretty
    logfile.write(line + '\n')
    line = ' Input: ' + inputfile
    logfile.write(line + '\n')

    # load dataset
    dataset = read_csv(inputfile, header=0, parse_dates=[0], date_parser=parse_dates)
    
    # drop date-time column
    dataset_data = dataset.drop('datetime', 1)    

    # prepare data
    scaler, trains, tests, tests_index, n_variables = sub.prepare_data_for_training(dataset_data, n_lag, n_forecast, n_lead, t0_forecast, n_split, n_days, ignoredVariables, rangeBuffer, logfile)

    # save scaler
    sname = 'output_train/model/'+now+'.scaler'
    joblib.dump(scaler, sname) 

    # save more configuration data
    line = ' Input_length: ' + str(n_lag) + ' hours'
    logfile.write(line + '\n')
    line = ' Make_forecast: ' + str(t0_make)
    logfile.write(line + '\n')
    line = ' Start_forecast: ' + str(t0_forecast)
    logfile.write(line + '\n')
    line = ' Forecast_length: ' + str(n_forecast) + ' hours'
    logfile.write(line + '\n')
    line = ' Cheating: '
    if cheat:
        line += 'ENABLED'
    else:
        line += 'DISABLED'
    logfile.write(line + '\n')
    line = ' Network: [' + ', '.join(str(x) for x in n_neurons) + ']'
    logfile.write(line + '\n')
    line = ' Batch_size: ' + str(n_batch) + ' days'
    logfile.write(line + '\n')
    line = ' Epochs: ' + str(n_epochs)
    logfile.write(line + '\n')
    line = ' Validation: '
    if validate:
        line += 'ENABLED'
    else:
        line += 'DISABLED'

    # fit model
    lossfig = 'output_train/lossHistory_' + now + '.png'
    model, time, forecasts = sub.fit_lstm(trains, n_lag, n_forecast, n_batch, n_epochs, n_neurons, lstmStateful, validate, tests, cheat, lossfig, verbosity, seed)

    # serialize model to JSON
    model_json = model.to_json()
    mname = 'output_train/model/'+now+'.json'
    with open(mname, 'w') as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    wname = 'output_train/model/'+now+'.h5'
    model.save_weights(wname)
    print("Saved model to disk")

###    # print as check that network has been correctly configured    
###    print model.layers
###    print model.inputs
###    print model.outputs

    # actual values
    actual = list()
    for test in tests:
        actual.extend([row[-n_forecast:] for row in test])
        
    # actual values at t-1
    i0 = (n_lag - 1) * n_variables
    baseline = list()
    for test in tests:    
        baseline.extend([row[i0] for row in test])

    # inverse transform
    print 'Inverse transform ...'
    forecasts = sub.inverse_transform(forecasts, scaler)
    actual = sub.inverse_transform(actual, scaler)

    # evaluate forecast quality
    print 'Calculating RMSE ...'
    RMSE, RMSE_persist = sub.evaluate_forecasts(actual, forecasts, baseline, n_forecast)

    line = ' RMSE(LSTM): %f' % RMSE
    logfile.write(line + '\n')
    line = ' RMSE(persist): %f' % RMSE_persist
    logfile.write(line + '\n')
    
    # save data and forecasts to root file
    rname = 'output_train/' + now + '.root'
    ro.save_root_tree(dataset, forecasts, tests, tests_index, n_lead, rname)

    # save more configuration data
    line = ' %.1f seconds' % time
    logfile.write(line + '\n')
    line = ' Loss history plot: ' + lossfig
    logfile.write(line + '\n')
    line = ' Root file: ' + rname
    logfile.write(line + '\n')
    line = ' Model: ' + mname
    logfile.write(line + '\n')
    line = ' Weights: ' + wname
    logfile.write(line + '\n')
    line = ' Scaler: ' + sname
    logfile.write(line + '\n')

    print 'Log file:', logfile.name
    logfile.close()

    print 'RMSE: ',RMSE    
        
    return RMSE


def test(inputfile, logfile, modelfile, weightsfile, scalerfile):

    # read settings from log file
    n_lag, n_forecast, t0_make, t0_forecast, ignore, n_batch = sub.read_log_file(logfile)
    n_lead = 24 - t0_make + 1 + t0_forecast

    # load dataset
    dataset = read_csv(inputfile, header=0, parse_dates=[0], date_parser=parse_dates)

    # drop date-time column
    dataset_data = dataset.drop('datetime', 1)    

    # load scaler
    scaler = joblib.load(scalerfile) 

    # load json and create model
    json_file = open(modelfile, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    # load weights into new model
    model.load_weights(weightsfile)

    print("Loaded model from disk")

    # prepare data
    reframed, names, scaler, n_variables, N = sub.prepare_data(dataset_data, n_lag, n_forecast, n_lead, t0_forecast, -1, ignore, 0, scaler)

    values = reframed.values
    
    # make forecast
    print("Forecasting ...")
    forecasts = sub.make_forecasts(model, n_batch, values, n_lag, n_forecast, False)
    
    # actual values
    actual = [row[-n_forecast:] for row in values]
        
    # actual values at t-1
    i0 = (n_lag - 1) * n_variables
    baseline = [row[i0] for row in values]

    # inverse transform
    print 'Inverse transform ...'
    forecasts = sub.inverse_transform(forecasts, scaler)
    actual = sub.inverse_transform(actual, scaler)

    # evaluate forecast quality
    print 'Calculating RMSE ...'
    RMSE, RMSE_persist = sub.evaluate_forecasts(actual, forecasts, baseline, n_forecast)
    print 'RMSE: %f' % RMSE

    # save to ascii file    
    outfile = open('test.out', 'w+')
    actual_list, forecast_list = list(), list()
    for i in range(n_forecast):
        actual_list.extend([row[i] for row in actual])
        forecast_list.extend([row[i] for row in forecasts])
    for j in range(len(actual)):
        line = ' %.1f, %.1f\n' % (actual_list[j], forecast_list[j])
        outfile.write(line)
    outfile.close()
    print 'Data and prediction saved to test.out'
