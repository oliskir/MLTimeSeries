
from ROOT import TTree, TFile
from rootpy.tree import Tree, TreeModel, FloatCol, IntCol, FloatArrayCol
from rootpy.io import root_open
    
#-----------------------------------------------------------------------
# save the data series and the forecasts to a ROOT TTree
#-----------------------------------------------------------------------
def save_root_tree(series, forecasts, n_test, fname):
    # number of time steps that we are forecasting
    mulmax = len(forecasts[0])
    # open ROOT file
    f = root_open(fname, "recreate")
    # define the model
    class Event(TreeModel):
        data = FloatCol()
        forecast = FloatArrayCol(mulmax)
    # create tree
    t = Tree("heat", model=Event)
    # power production time series
    s = series.values[:, 0]
    # start and end hour of each forecast
    start, stop = [], []
    for i in range(mulmax):
        start.append(len(s) - n_test - mulmax + i + 1)
        stop.append(start[i] + n_test - 1)
    # loop over all data  
    for n in range(len(s)):
        t.data = s[n]
        # loop over forecasts            
        for i in range(mulmax):
            if (n >= start[i]) and (n <= stop[i]):
                j = n - start[i]
                t.forecast[i] = forecasts[j][i]
        t.fill()
    t.write()
    f.close()
