 #!/usr/bin/python

import sys, getopt
import ml

def main(argv):

    inputfile = '../../heat_load_weather_calendar.csv'

    now = '2017-11-20_165128'

    modelfile   = 'output_train/model/' + now + '.json'
    weightsfile = 'output_train/model/' + now + '.h5'
    scalerfile  = 'output_train/model/' + now + '.scaler'

    # run program
    ml.test(inputfile, modelfile, weightsfile, scalerfile)


if __name__ == "__main__":
    main(sys.argv[1:])
   
   


