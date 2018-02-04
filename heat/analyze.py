#!/usr/bin/python

import train


inputfile = '../../heat_load_data/heat_load_weather_calendar.csv'

seed = 7

ignore = ['Tout','vWind','sunRad','weekend','observance','national_holiday','school_holiday','hour','weekday','month']

ignore.remove('Tout')
ignore.remove('vWind')
ignore.remove('sunRad')
             
e=5000
for n in [10,20,40]:
    logfile = open('analyze.log', 'a')
    logfile.write('\n')
    y = list()
    for i in range(1):
        yy = train.train(inputfile, 158, 24, 10, 1, [n,n], e, 365, 5, 'Standard', False, False, 0, seed, ignore)
        y.append(yy)
    rmse = sum(y)/len(y)
    x = 'n=[%i,%i] %.1f' % (n, rmse)
    logfile.write(x + '\n')
    logfile.close()

