#!/bin/bash

#python train.py -n 4 -e 1000 -v 0 -V -C
#python train.py -n 16 -e 1000 -v 0 -V -C
#python train.py -n 64 -e 1000 -v 0 -V -C
#python train.py -n 128 -e 1000 -v 0 -V -C

#python train.py -n 100 -i 34 -f 24 -m 10 -s 0 -e 10 -v 0 -V -C

python train.py -n 10 -i 34 -f 24 -m 10 -s 1 -e 1000 -r 7 -v 0 -V -g sunRad,hour,weekend,observance,national_holiday,school_holiday,weekday,month
