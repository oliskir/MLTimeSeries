#!/usr/bin/python

import ml


inputfile = '../../heat_load_weather_calendar.csv'

logfile = open('analyze.log', 'a')
logfile.write('\n')

seed = 7

for n in range(10,20,10):
    y = list()
    for i in range(1):
        yy = ml.train(inputfile, 1, 24, 10, 0, [n], 5000, 365, 5, False, False, 0, seed)
        y.append(yy)
    rmse = sum(y)/len(y)
    x = 'n=[%i] %.1f' % (n, rmse)
    logfile.write(x + '\n')

logfile.close()

