#!/bin/bash

#python main.py -n 4 -e 1000 -v 0 -V -C
#python main.py -n 16 -e 1000 -v 0 -V -C
#python main.py -n 64 -e 1000 -v 0 -V -C
#python main.py -n 128 -e 1000 -v 0 -V -C

python main.py -n 16 -e 10000 -l 1 -f 32 -v 0 -V -C
python main.py -n 16,16 -e 10000 -l 1 -f 32 -v 0 -V -C
python main.py -n 16,16,16 -e 10000 -l 1 -f 32 -v 0 -V -C
python main.py -n 16,16,16,16 -e 10000 -l 1 -f 32 -v 0 -V -C
