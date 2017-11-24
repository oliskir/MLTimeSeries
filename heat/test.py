 #!/usr/bin/python

import sys, getopt
import ml

def main(argv):

    # default values
    inputfile = '../../heat_load_weather_calendar.csv'
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
    ml.test(inputfile, logfile, modelfile, weightsfile, scalerfile)


if __name__ == "__main__":
    main(sys.argv[1:])
   
   


