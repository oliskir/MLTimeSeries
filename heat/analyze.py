#!/usr/bin/python

import train


inputfile = '../../heat_load_weather_calendar.csv'

seed = 7

ignore = ['Tout','vWind','sunRad','hour','weekend','observance','national_holiday','school_holiday','weekday','month']

ignore.remove('Tout')
ignore.remove('vWind')
ignore.remove('weekend')
ignore.remove('observance')
ignore.remove('national_holiday')
ignore.remove('school_holiday')
ignore.remove('weekday')
ignore.remove('month')
             
e=50
for n in range(1):
    logfile = open('analyze.log', 'a')
    logfile.write('\n')
    y = list()
    for i in range(1):
        yy = train.train(inputfile, 1, 24, 10, 0, [2,2], e, 365, 5, False, False, 0, seed, ignore)
        y.append(yy)
    rmse = sum(y)/len(y)
    x = 'n=[%i] %.1f' % (n, rmse)
    logfile.write(x + '\n')
    logfile.close()

