# MLTimeSeries
Machine learning approach to time-series forecasting.

Based on: 
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

## Syntax
```terminal
python main.py -n <neurons1>,<neurons2>,... -l <lag> -f <forecast-length> -a <lead> -s <forecast-start> -e <epochs> -b <batch-size> -t <training-fraction> -v <verbosity> -i <input-file> -D -V -C
```
## Arguments
| Option     | Description           | Example  |
| ---------- | --------------------- | -------- |
| -n         | Number of LSTM neurons in each layer of the network (default: 100) | -n 20,30   |
| -i         | Input length: time steps (hours) back in time used for the forecast (default: 34) | -i 38 |
| -f         | Forecast length: time steps (hours) into the future that we wish to forecast (default: 24) | -f 12 |
| -m         | Hour that forecast is made (default: 10) | -a 4 |
| -s         | Start hour of forecast window (default: 0) | -s 12 |
| -e         | Training epochs (default: 1000) | -e 10000 |
| -b         | Batch size (days) (default: 365) | -b 100 |
| -p         | Number of splits for cross validation (default: 5) | -p 4 |
| -v         | Verbosity level  | -v 2 |
| -d         | Input data file       | -d data.csv |
| -D         | Predict change rather than absolute value | -D | 
| -V         | Validate on test data during training | -V | 
| -C         | Cheat (include values that we wish to forecast as part of the input) | -C |

## Example
```terminal
python main.py -n 16,16 -i 24 -f 24 -m 13 -s 0 -e 10000 -b 365 -p 5 -v 0 -d data.csv -C
```
Two-layer LSTM network with 16 x 16 neurons. Forecast the power production for every hour of the following day starting at midnight. The forecast is made at 1 PM. Use data from the latest 24 hours as input. 1E4 epochs and batch size of 365 (1 year). Split the data into 5 equal size chuncks for cross validation. Lowest possible verbosity level. The data file is called data.csv. Cheating is enabled. 

## Output
The program outputs (1) a .txt file with a summary of the settings and the forecast quality, (2) a .root file with the complete series of data points and forecasted values, and (3) a .png figure showing the loss history. The files are all tagged with the current date and time and saved to the subfolder output/.

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
