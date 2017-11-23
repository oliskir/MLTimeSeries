
from pandas import read_csv
import subroutines as sub
import datetime as dt
import os
from sklearn.externals import joblib
from keras.models import model_from_json

import rootoutput as ro


def parse_dates(x):
    return dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
    
def train(inputfile, n_lag, n_forecast, t0_make, t0_forecast, n_neurons, n_epochs, n_batch, n_split, validate, cheat, verbosity, seed):

    # configure
    n_lead = 24 - t0_make + 1 + t0_forecast
    lstmStateful = False
    n_days = -1   # set to -1 to process entire data set
##    ignoredVariables = [3,8] # sunRad and hour
##    ignoredVariables = [3,5,6,7,8,9,10] 
    ignoredVariables = [3,4,5,6,7,8,9,10]
    rangeBuffer = 0.05

    # create output directories if necessary
    try: 
        os.makedirs('output')
    except OSError:
        if not os.path.isdir('output'):
            raise
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
    joblib.dump(scaler, 'output_train/model/'+now+'.scaler') 

    # save more configuration data
    line = ' Input length: ' + str(n_lag) + ' hours'
    logfile.write(line + '\n')
    line = ' Make forecast: ' + str(t0_make) + ':00'
    logfile.write(line + '\n')
    line = ' Start forecast: ' + str(t0_forecast) + ':00'
    logfile.write(line + '\n')
    line = ' Forecast length: ' + str(n_forecast) + ' hours'
    logfile.write(line + '\n')
    line = ' Cheating: '
    if cheat:
        line += 'ENABLED'
    else:
        line += 'DISABLED'
    logfile.write(line + '\n')
    line = ' Network: [' + ', '.join(str(x) for x in n_neurons) + ']'
    logfile.write(line + '\n')
    line = ' Batch size: ' + str(n_batch) + ' days'
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
    model, timePerEpoch, forecasts = sub.fit_lstm(trains, n_lag, n_forecast, n_batch, n_epochs, n_neurons, lstmStateful, validate, tests, cheat, lossfig, verbosity, seed)

    # serialize model to JSON
    model_json = model.to_json()
    with open('output_train/model/'+now+'.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('output_train/model/'+now+'.h5')
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
    forecasts = sub.inverse_transform(forecasts, scaler, n_variables, baseline)
    actual = sub.inverse_transform(actual, scaler, n_variables, baseline)

    # evaluate forecast quality
    print 'Calculating RMSE ...'
    RMSE = sub.evaluate_forecasts(actual, forecasts, baseline, n_forecast, logfile)
    
    # save data and forecasts to root file
    rname = 'output_train/' + now + '.root'
    ro.save_root_tree(dataset, forecasts, tests, tests_index, n_lead, rname)

    # save more configuration data
    line = ' %.1f seconds/epoch' % timePerEpoch
    logfile.write(line + '\n')
    line = ' Loss history plot: ' + lossfig
    logfile.write(line + '\n')
    line = ' Root file: ' + rname
    logfile.write(line + '\n')

    print 'Log file:', logfile.name
    logfile.close()

    print 'RMSE: ',RMSE    
        
    return RMSE


def test(inputfile, modelfile, weightsfile, scalerfile):

    # load dataset
    dataset = read_csv(inputfile, header=0, parse_dates=[0], date_parser=parse_dates)

    # drop date-time column
    dataset_data = dataset.drop('datetime', 1)    

    # load scaler
    scaler = joblib.load(scalerfile) 

    # load json and create model
    json_file = open(modelfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(weightsfile)

    print("Loaded model from disk")

    # prepare data
#    scaler, trains, tests, tests_index, n_variables = sub.prepare_data_for_testing(dataset_data, n_lag, n_forecast, n_lead, t0_forecast, n_split, n_days, ignoredVariables, predictChange, rangeBuffer, logfile)
    

