 #!/usr/bin/python

from pandas import read_csv
from pandas import read_pickle
import subroutines as sub
import datetime as dt
import os
from sklearn.externals import joblib
from keras.models import model_from_json
import sys
   
   
def parse_dates(x):
    return dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def test(inputfile, logfile, modelfile, weightsfile, scalerfile):

    # read settings from log file
    n_lag, n_forecast, t0_make, t0_forecast, ignore, n_batch = sub.read_log_file(logfile)
    n_lead = 24 - t0_make + 1 + t0_forecast
    
    # load dataset
    if '.csv' in inputfile:
        dataset = read_csv(inputfile, header=0, parse_dates=[0], date_parser=parse_dates)
        dataset_data = dataset.drop('datetime', 1)    
    elif '.pkl' in inputfile:
        dataset_data = read_pickle(inputfile)
    else:
        print 'Unknown file format: ',inputfile
        sys.exit(0)
        
    # ignore lagged variables
    variableNames = list(dataset_data.columns.values)
    for name in variableNames:
        if '_lag' in name:
            ignore.append(name)

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
    

import getopt


def main(argv):

    # default values
    inputfile = '../../heat_load_data/heat_load_weather_calendar.csv'
    now = '2017-11-24_081427'

    # parse command-line args
    try:
        opts, args = getopt.getopt(argv,"hd:m:")
    except getopt.GetoptError:
        print 'test.py -d <data-file> -m <model-datetime>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -d <data-file> -m <model-datetime>'
            sys.exit()
        elif opt in ("-m"):
            now = arg
        elif opt in ("-d"):
            inputfile = arg

    logfile   = 'output_train/' + now + '.log'
    modelfile   = 'output_train/model/' + now + '.json'
    weightsfile = 'output_train/model/' + now + '.h5'
    scalerfile  = 'output_train/model/' + now + '.scaler'

    # run program
    test(inputfile, logfile, modelfile, weightsfile, scalerfile)


if __name__ == "__main__":
    main(sys.argv[1:])
    
