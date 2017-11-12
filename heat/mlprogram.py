
from pandas import read_csv
import subroutines as sub
import datetime as dt
import os
import rootoutput as ro


def parse_dates(x):
    return dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
    
def run(inputfile, n_lag, n_forecast, n_lead, t0_forecast, n_neurons, n_epochs, n_batch, train_fraction, predictChange, validate, cheat, verbosity):

    # configure
    lstmStateful = False
    n_days = -1   # set to -1 to process entire data set
    ignoredVariables = [3, 4, 5, 6, 7, 8, 9, 10]

    # create output directory if necessary
    try: 
        os.makedirs('output')
    except OSError:
        if not os.path.isdir('output'):
            raise
        
    # current date and time
    today = dt.datetime.now().strftime("%Y-%m-%d")
    now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # open log file
    logfile = open('output/'+now+'.log', 'w+')

    # save configuration data
    nowPretty = dt.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    line = '# ' + nowPretty
    logfile.write(line + '\n')
    line = '# Input: ' + inputfile
    logfile.write(line + '\n')

    # load dataset
###    dataset = read_csv(inputfile, header=0, index_col=0, parse_dates=[0], date_parser=parse_dates)
    dataset = read_csv(inputfile, header=0, parse_dates=[0], date_parser=parse_dates)
    
    # drop date-time column
    dataset_data = dataset.drop('datetime', 1)    

    # prepare data
    scaler, train, test, test_index, n_variables = sub.prepare_data(dataset_data, n_lag, n_forecast, n_lead, t0_forecast, train_fraction, n_days, ignoredVariables, predictChange, logfile)

    # save more configuration data
    line = '# Lag: ' + str(n_lag)
    logfile.write(line + '\n')
    line = '# Forecast: ' + str(n_forecast)
    logfile.write(line + '\n')
    line = '# Lead: ' + str(n_lead)
    logfile.write(line + '\n')
    line = '# Forecast start: ' + str(t0_forecast)
    logfile.write(line + '\n')
    line = '# Forecast type: '
    if predictChange:
        line += 'Relative'
    else:
        line += 'Absolute'
    logfile.write(line + '\n')
    line = '# Cheating: '
    if cheat:
        line += 'ENABLED'
    else:
        line += 'DISABLED'
    logfile.write(line + '\n')
    line = '# Network: [' + ', '.join(str(x) for x in n_neurons) + ']'
    logfile.write(line + '\n')
    line = '# Batch size: ' + str(n_batch)
    logfile.write(line + '\n')
    line = '# Epochs: ' + str(n_epochs)
    logfile.write(line + '\n')
    line = '# Validation: '
    if validate:
        line += 'ENABLED'
    else:
        line += 'DISABLED'

    # fit model
    lossfig = 'output/lossHistory_' + now + '.png'
    model, timePerEpoch = sub.fit_lstm(train, n_forecast, n_batch, n_epochs, n_neurons, lstmStateful, validate, test, cheat, lossfig, verbosity)

###    # print as check that network has been correctly configured    
###    print model.layers
###    print model.inputs
###    print model.outputs

    # make forecast
    print 'Forecasting ...'
    forecasts = sub.make_forecasts(model, n_batch, test, n_lag, n_forecast, cheat)

    # actual values
    actual = [row[-n_forecast:] for row in test]

    # actual values at t-1
    i0 = (n_lag - 1) * n_variables
    baseline = [row[i0] for row in test]

    # inverse transform
    print 'Inverse transform ...'
    forecasts = sub.inverse_transform(forecasts, scaler, n_variables, predictChange, baseline)
    actual = sub.inverse_transform(actual, scaler, n_variables, predictChange, baseline)

    # evaluate forecast quality
    print 'Calculating RMSE ...'
    sub.evaluate_forecasts(actual, forecasts, n_forecast, logfile)
    
    # save data and forecasts to root file
    rname = 'output/' + now + '.root'
    ro.save_root_tree(dataset, forecasts, test, test_index, n_lead, rname)

    # save more configuration data
    line = "# %.1f seconds/epoch" % timePerEpoch
    logfile.write(line + '\n')
    line = '# Loss history plot: ' + lossfig
    logfile.write(line + '\n')
###    line = '# LSTM forecast plot: ' + forecastfig
###    logfile.write(line + '\n')
    line = '# Root file: ' + rname
    logfile.write(line + '\n')

    print 'Log file:', logfile.name
    logfile.close()

