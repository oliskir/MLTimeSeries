#!/usr/bin/python

import sys, getopt
import mlprogram as ml

def main(argv):

    # default values
    inputfile = '../../heat_load_weather_calendar.csv'
    n_lag = 24
    n_forecast = 24
    n_lead = 14
    t0_forecast = 0  # set to -1 to make hourly forecasts
    n_neurons = [40]
    n_epochs = 2
    n_batch = 365
    train_fraction = 0.33
    predictChange = False
    validate = False
    cheat = False
    verbosity = 2

    # parse command-line args
    try:
        opts, args = getopt.getopt(argv,"hDVCl:f:e:b:t:v:i:n:a:s:")
    except getopt.GetoptError:
        print 'main.py -n <neurons> -l <lag> -f <forecast-length> -a <lead> -s <forecast-start-hour> -e <epochs> -b <batch-size> -t <training-fraction> -v <verbosity> -i <input-file> -D -V -C'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -n <neurons> -l <lag> -f <forecast-length> -e <epochs> -b <batch-size> -t <training-fraction> -v <verbosity> -i <input-file> -D -V -C'
            sys.exit()
        elif opt in ("-l"):
            n_lag = int(arg)
        elif opt in ("-f"):
            n_forecast = int(arg)
        elif opt in ("-e"):
            n_epochs = int(arg)
        elif opt in ("-b"):
            n_batch = int(arg)
        elif opt in ("-t"):
            train_fraction = float(arg)
        elif opt in ("-v"):
            verbosity = int(arg)
        elif opt in ("-a"):
            n_lead = int(arg)
        elif opt in ("-s"):
            t0_forecast = int(arg)
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-n"):
            layers = arg.split(",")
            del n_neurons[:]
            for l in layers:
                n_neurons.append(int(l))
        elif opt in ("-D"):
            predictChange = True
        elif opt in ("-V"):
            validate = True
        elif opt in ("-C"):
            cheat = True

    # run program
    ml.run(inputfile, n_lag, n_forecast, n_lead, t0_forecast, n_neurons, n_epochs, n_batch, train_fraction, predictChange, validate, cheat, verbosity)


if __name__ == "__main__":
    main(sys.argv[1:])
   
   


