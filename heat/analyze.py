#!/usr/bin/python

import ml


inputfile = '../../heat_load_weather_calendar.csv'

logfile = open('analyze.log', 'a')
logfile.write('\n')

seed = 7

ignore = ['sunRad','hour','weekend','observance','national_holiday','school_holiday','weekday','month']

for j in range(1):
    print 'j: ',j
    if j == 1:
        ignore.remove('weekend')
    elif j == 2:
        ignore.remove('observance')
    elif j == 3:
        ignore.remove('national_holiday')
    elif j == 4:
        ignore.remove('school_holiday')
    elif j == 5:
        ignore.remove('weekday')
    elif j == 6:
        ignore.remove('month')
          
    for n in range(10,11,1):
        y = list()
        for i in range(1):
            yy = ml.train(inputfile, 178, 24, 10, 0, [n], 5000, 365, 5, False, False, 0, seed, ignore)
            y.append(yy)
        rmse = sum(y)/len(y)
        x = 'n=[%i] %.1f' % (n, rmse)
        logfile.write(x + '\n')

logfile.close()

