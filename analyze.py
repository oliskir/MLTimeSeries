#!/usr/bin/python

import train
import test

inputfile = '../heat_load_data/heat_load_weather_calendar.csv'

seed = 0 #7

scaler = 'Standard' #'MinMax'

validate = True

days = 500 #730 #365

ignore = ['Tout','vWind','sunRad','weekend','observance','national_holiday','school_holiday','hour','weekday','month']

ignore.remove('Tout')
ignore.remove('vWind')
ignore.remove('sunRad')

#ignore.remove('weekend')
#ignore.remove('weekday')
#ignore.remove('month')
#ignore.remove('hour')

ignore.remove('observance')
ignore.remove('national_holiday')
ignore.remove('school_holiday')

             
n=110
for e in [15]:
    logfile = open('analyze.log', 'a')
    logfile.write('\n')
    y = list()
    z = list()
    for i in range(5):
        yy, lfile, mfile, wfile, sfile = train.train(inputfile, 168, 24, 10, 1, [n], e, days, 5, scaler, validate, False, 0, seed, ignore)
        y.append(yy)
        
        zz = test.test("../heat_load_data/test_data.pkl", lfile, mfile, wfile, sfile)
        z.append(zz)
        zs = '%.1f, ' % zz
        logfile.write(zs)
        
    rmse = sum(y)/len(y)
    rmse_test = sum(z)/len(z)
    x = '  [%.1f]' % (rmse_test)
    logfile.write(x + '\n')
    logfile.close()

