#!/usr/bin/python

import train


inputfile = '../../heat_load_data/heat_load_weather_calendar.csv'

seed = 7

scaler = 'Standard' #'MinMax'

validate = True

ignore = ['Tout','vWind','sunRad','weekend','observance','national_holiday','school_holiday','hour','weekday','month']

ignore.remove('Tout')
ignore.remove('vWind')
ignore.remove('sunRad')

ignore.remove('weekend')
ignore.remove('weekday')
ignore.remove('month')
#ignore.remove('hour')

#ignore.remove('observance')
#ignore.remove('national_holiday')
#ignore.remove('school_holiday')

             
e=15
for n in [110]:
    logfile = open('analyze.log', 'a')
    logfile.write('\n')
    y = list()
    for i in range(1):
        yy = train.train(inputfile, 168, 24, 10, 1, [n], e, 365, 5, scaler, validate, False, 0, seed, ignore)
        y.append(yy)
    rmse = sum(y)/len(y)
    x = 'n=[%i] %.1f' % (n, rmse)
    logfile.write(x + '\n')
    logfile.close()

