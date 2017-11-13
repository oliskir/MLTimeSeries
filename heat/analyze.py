#!/usr/bin/python

import mlprogram as ml

def analyze:

    inputfile = '../../heat_load_weather_calendar.csv'

    logfile = open('analyze.log', 'w+')

    rmse = ml.run(inputfile, 34, 24, 10, 0, [50], 1000, 365, 5, False, False, True, 0)
    x = 'n=[50] %.1f' % rmse
    logfile.write(x)

    rmse = ml.run(inputfile, 34, 24, 10, 0, [75], 1000, 365, 5, False, False, True, 0)
    x = 'n=[75] %.1f' % rmse
    logfile.write(x)

    rmse = ml.run(inputfile, 34, 24, 10, 0, [100], 1000, 365, 5, False, False, True, 0)
    x = 'n=[100] %.1f' % rmse
    logfile.write(x)

    rmse = ml.run(inputfile, 34, 24, 10, 0, [150], 1000, 365, 5, False, False, True, 0)
    x = 'n=[150] %.1f' % rmse
    logfile.write(x)

    rmse = ml.run(inputfile, 34, 24, 10, 0, [200], 1000, 365, 5, False, False, True, 0)
    x = 'n=[200] %.1f' % rmse
    logfile.write(x)
    
    
    logfile.close()

