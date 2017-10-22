
from pandas import read_csv
import subroutines as sub
import datetime
import os


def run(inputfile, n_lag, n_forecast, n_neurons, n_epochs, n_batch, train_fraction, predictChange, validate, cheat, verbosity):

    # configure
    lstmStateful = False
    n_days = -1   # -1 will process entire data set
    ignoredVariables = [4, 5, 6, 7, 8, 9, 10]  # numbering begins with zero: 0,1,2...etc

    # create output directory if necessary
    try: 
        os.makedirs('output')
    except OSError:
        if not os.path.isdir('output'):
            raise
        
    # current date and time
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # open log file
    logfile = open('output/log_'+now+'.txt', 'w+')

    # save configuration data
    nowPretty = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    line = '# ' + nowPretty
    logfile.write(line + '\n')
    line = '# Input: ' + inputfile
    logfile.write(line + '\n')

    # load dataset
    dataset = read_csv(inputfile, header=0, index_col=0)

    # prepare data
    scaler, train, test, n_variables = sub.prepare_data(dataset, n_lag, n_forecast, train_fraction, n_days, ignoredVariables, predictChange, logfile)

    # save more configuration data
    line = '# Lag: ' + str(n_lag)
    logfile.write(line + '\n')
    line = '# Forecast: ' + str(n_forecast)
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

    # plot forecasts
    forecastfig = 'output/forecast_' + now + '.png'
    sub.plot_forecasts(dataset, forecasts, test.shape[0], forecastfig)


    # save more configuration data
    line = "# %.1f seconds/epoch" % timePerEpoch
    logfile.write(line + '\n')
    line = '# Loss history plot: ' + lossfig
    logfile.write(line + '\n')
    line = '# LSTM forecast plot: ' + forecastfig
    logfile.write(line + '\n')

    print 'Log file:', logfile.name
    logfile.close()

