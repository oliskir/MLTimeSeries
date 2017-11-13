#!/usr/bin/python

import sys, getopt
import mlprogram as ml

def main(argv):

    # default values
    inputfile = '../../heat_load_weather_calendar.csv'
    n_lag = 34
    n_forecast = 24
    t0_make = 10 
    t0_forecast = 0
    n_neurons = [100]
    n_epochs = 1000
    n_batch = 365
    n_split = 5
    predictChange = False
    validate = False
    cheat = False
    verbosity = 2

    # parse command-line args
    try:
        opts, args = getopt.getopt(argv,"hDVCi:f:e:b:p:v:d:n:m:s:")
    except getopt.GetoptError:
        print 'main.py -n <neurons> -l <input-length> -f <forecast-length> -m <forecast-make-hour> -s <forecast-start-hour> -e <epochs> -b <batch-size> -p <split> -v <verbosity> -d <data-file> -D -V -C'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -n <neurons> -i <input-length> -f <forecast-length> -m <forecast-make-hour> -s <forecast-start-hour> -e <epochs> -b <batch-size> -p <split> -v <verbosity> -d <data-file> -D -V -C'
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
            t0_start = int(arg)
        elif opt in ("-d"):
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
    ml.run(inputfile, n_lag, n_forecast, t0_make, t0_start, n_neurons, n_epochs, n_batch, n_split, predictChange, validate, cheat, verbosity)


if __name__ == "__main__":
    main(sys.argv[1:])
   
   


