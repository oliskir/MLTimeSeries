# MLTimeSeries
Machine learning approach to time-series forecasting.

Based on: 
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

## Syntax
```terminal
python main.py -n <neurons1>,<neurons2>,... -l <lag> -f <forecast> -e <epochs> -b <batch-size> -t <training-fraction> -v <verbosity> -i <input-file> -D -V -C
```
## Arguments
| Option     | Description           | Example  |
| ---------- | --------------------- | -------- |
| -n         | Number of LSTM neurons in each layer of the network  | -n 20,30   |
| -l         | Lag: time steps (hours) back in time used for the forecast | -l 24 |
| -f         | Forecast length: time steps (hours) into the future that we wish to forecast | -f 12 |
| -e         | Training epochs | -e 1000 |
| -b         | Batch size | -b 365 |
| -t         | Fraction of data set used for training | -t 0.3 |
| -v         | Verbosity level  | -v 2 |
| -i         | Input file       | -i data.csv |
| -D         | Predict change rather than absolute value | -D | 
| -V         | Validate on test data during training | -V | 
| -C         | Cheat (include values that we wish to forecast as part of the input) | -C |

## Example
```terminal
python main.py -n 16 -l 24 -f 24 -e 10000 -b 365 -t 0.3 -v 0 -i data.csv -C
```
Two-layer LSTM network with 16 x 16 neurons. Forecast next 24 hours based on previous 24 hours. 1E4 epochs and batch size of 365 (1 year). Use 1/3 of the data set for training. Lowest possible verbosity level. The data file is called data.csv. Cheating is enabled. 

## Output
The program outputs (1) a .txt file with a summary of the settings and the forecast quality, (2) a .root file with the complete series of data points and forecasted values, and (3) a .png figure showing the loss history. The files are all tagged with the current date and time and saved to the subfolder output/.

## Prerequisites
- Python 2.7
- numpy
- Keras deep learning library with Theano backend
- ROOT
- rootpy

ROOT and rootpy are only needed to produce the .root output file, and hence are not essential.
