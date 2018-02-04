#!/usr/bin/python

from pandas import read_csv
import subroutines as sub
import datetime as dt
import os
from sklearn.externals import joblib
from keras.models import model_from_json
import rootoutput as ro
   
   
def parse_dates(x):
    return dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
        
def train(inputfile, n_lag, n_forecast, t0_make, t0_forecast, n_neurons, n_epochs, n_batch, n_split, scalerType, validate, cheat, verbosity, seed, ignoredVariables):

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
    scaler, trains, tests, tests_index, n_variables = sub.prepare_data_for_training(dataset_data, n_lag, n_forecast, n_lead, t0_forecast, n_split, n_days, ignoredVariables, scalerType, logfile)

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


import sys, getopt


def main(argv):

    # default values
    inputfile = '../../heat_load_data/heat_load_weather_calendar.csv'
    n_lag = 34
    n_forecast = 24
    t0_make = 10 
    t0_forecast = 0
    n_neurons = [100]
    n_epochs = 1000
    n_batch = 365
    n_split = 5
    scalerType = 'Standard'
    validate = False
    cheat = False
    verbosity = 0
    seed = 0
    ignore = []

    # parse command-line args
    try:
        opts, args = getopt.getopt(argv,"hVCi:f:e:b:p:v:d:n:m:s:r:g:x:")
    except getopt.GetoptError:
        print 'train.py -n <neurons-1st-layer-1>[<neurons-2nd-layer>,etc] -i <input-length> -f <forecast-length> -m <forecast-make-hour> -s <forecast-start-hour> -e <epochs> -b <batch-size> -p <split> -x <scaler-type> -v <verbosity> -r <random-number-generator-seed> -d <data-file> -g <ignore-column>[<ignore-another-column>,etc] -V -C'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'train.py -n <neurons-1st-layer-1>[<neurons-2nd-layer>,etc] -i <input-length> -f <forecast-length> -m <forecast-make-hour> -s <forecast-start-hour> -e <epochs> -b <batch-size> -p <split> -x <scaler-type> -v <verbosity> -r <random-number-generator-seed> -d <data-file> -g <ignore-column>[<ignore-another-column>,etc] -V -C'
            sys.exit()
        elif opt in ("-i"):
            n_lag = int(arg)
        elif opt in ("-f"):
            n_forecast = int(arg)
        elif opt in ("-e"):
            n_epochs = int(arg)
        elif opt in ("-b"):
            n_batch = int(arg)
        elif opt in ("-p"):
            n_split = int(arg)
        elif opt in ("-v"):
            verbosity = int(arg)
        elif opt in ("-m"):
            t0_make = int(arg)
        elif opt in ("-s"):
            t0_forecast = int(arg)
        elif opt in ("-d"):
            inputfile = arg
        elif opt in ("-r"):
            seed = int(arg)
        elif opt in ("-x"):
            scalerType = arg
        elif opt in ("-n"):
            layers = arg.split(",")
            del n_neurons[:]
            for l in layers:
                n_neurons.append(int(l))
        elif opt in ("-g"):
            ignore = arg.split(",")
        elif opt in ("-V"):
            validate = True
        elif opt in ("-C"):
            cheat = True

    # run program
    train(inputfile, n_lag, n_forecast, t0_make, t0_forecast, n_neurons, n_epochs, n_batch, n_split, scalerType, validate, cheat, verbosity, seed, ignore)


if __name__ == "__main__":
    main(sys.argv[1:])
   
