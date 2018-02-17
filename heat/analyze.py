#!/usr/bin/python

import train
import test

inputfile = '../../heat_load_data/heat_load_weather_calendar.csv'

seed = 0 #7

scaler = 'Standard' #'MinMax'

validate = True

days = 500 #730 #365

ignore = ['Tout','vWind','sunRad','weekend','observance','national_holiday','school_holiday','hour','weekday','month']

ignore.remove('Tout')
ignore.remove('vWind')
ignore.remove('sunRad')

ignore.remove('weekend')
ignore.remove('weekday')
#ignore.remove('month')
#ignore.remove('hour')

#ignore.remove('observance')
#ignore.remove('national_holiday')
#ignore.remove('school_holiday')

             
e=15
for n in [110]:
    logfile = open('analyze.log', 'a')
    logfile.write('\n')
    y = list()
    for i in range(5):
        yy, lfile, mfile, wfile, sfile = train.train(inputfile, 168, 24, 10, 1, [n], e, days, 5, scaler, validate, False, 0, seed, ignore)
        y.append(yy)
        
        z = test.test("../../heat_load_data/test_data.pkl", lfile, mfile, wfile, sfile)
        zs = '%.1f, ' % z
        logfile.write(zs)
        
    rmse = sum(y)/len(y)
    x = '  [%.1f]' % (rmse)
    logfile.write(x + '\n')
    logfile.close()

