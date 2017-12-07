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
             
e=5000
n=5
for input in [1,34,58,178]:
    logfile = open('analyze.log', 'a')
    logfile.write('\n')
    y = list()
    for i in range(1):
        yy = train.train(inputfile, input, 24, 10, 0, [n,n], e, 365, 5, False, False, 0, seed, ignore)
        y.append(yy)
    rmse = sum(y)/len(y)
#    x = 'n=[%i] %.1f' % (n, rmse)
    x = 'input=[%i] %.1f' % (input, rmse)
    logfile.write(x + '\n')
    logfile.close()

