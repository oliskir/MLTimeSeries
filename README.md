# MLTimeSeries
Machine learning approach to time-series forecasting.

Based on: 
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

## Test model
```terminal
python test.py -d <csv-data-file> -m <model-datetime>
```

## Train model
```terminal
python train.py [options] -d <csv-data-file>
```

## Options
| Option     | Description           | Example  |
| ---------- | --------------------- | -------- |
| -n         | Number of LSTM neurons in each layer of the network (default: 100) | -n 20,30   |
| -i         | Input length: time steps (hours) back in time used for the forecast (default: 34) | -i 38 |
| -f         | Forecast length: time steps (hours) into the future that we wish to forecast (default: 24) | -f 12 |
| -m         | Hour (0-23) that the forecast is made (default: 10) | -m 4 |
| -s         | Start hour (0-23) of forecast window (default: 0) | -s 12 |
| -e         | Training epochs (default: 1000) | -e 10000 |
| -b         | Batch size (days) (default: 365) | -b 100 |
| -p         | Number of splits for cross validation (default: 5) | -p 4 |
| -v         | Verbosity level (default: 0)  | -v 2 |
| -V         | Enable validation  | -V | 
| -C         | Cheat (include values that we wish to forecast as part of the input) | -C |
| -r         | Seed for random number generator | -r 7 |
| -g         | Ignore columns in the data file | -g sunRad,hour |

## Example
```terminal
python train.py -n 16,16 -i 24 -f 24 -m 13 -s 0 -e 10000 -b 365 -p 5 -d data.csv -C -g sunRad,hour
python test.py -d data.csv -m 2017-11-24_081427
```
Two-layer LSTM network with 16 x 16 neurons. Forecast the power production for every hour of the following day starting at midnight. The forecast is made at 1 PM. Use data from the latest 24 hours as input. 1E4 epochs and batch size of 365 (1 year). Split the data into 5 equal size chuncks for cross validation. The data file is called data.csv. Cheating is enabled. The columns 'sunRad' and 'hour' will be ignored.

## Output
The program outputs:

1) x.log: Configuration and RMSE

2) x.root: Data series and forecast

3) x.png: Loss history graph. 

4) x.json: Keras model

5) x.h5: Weights

6) x.scaler: Normalization

where x is the current date and time. Files 1-3 are saved in the subfolder output_train/ while files 4-6 are saved in output_train/model/.

## Prerequisites
- Python 2.7
- numpy
- Keras deep learning library with Theano backend
- ROOT
- rootpy

ROOT and rootpy are only needed to produce the .root output file, and hence are not essential.

## Installation notes
- To run Keras on GPU: 
(1) conda install theano pygpu
(1) conda install -c mila-udem -c mila-udem/label/pre theano pygpu
(2) THEANO_FLAGS=device=cuda,floatX=float32 python my_keras_script.py
