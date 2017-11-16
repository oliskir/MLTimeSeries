#!/usr/bin/python

import mlprogram as ml


inputfile = '../../heat_load_weather_calendar.csv'

logfile = open('analyze.log', 'a')
logfile.write('\n')

for n in range(10,60,10):
    y = list()
    for i in range(3):
        yy = ml.run(inputfile, 34, 24, 10, 0, [n], 5000, 365, 5, False, False, False, 0)
        y.append(yy)
    rmse = sum(y)/len(y)
    x = 'n=[%i] %.1f' % (n, rmse)
    logfile.write(x + '\n')

logfile.close()

