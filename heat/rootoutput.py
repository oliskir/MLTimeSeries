
from ROOT import TTree, TFile
from rootpy.tree import Tree, TreeModel, FloatCol, IntCol, FloatArrayCol
from rootpy.io import root_open
    
#-----------------------------------------------------------------------
# save the data series and the forecasts to a ROOT TTree
#-----------------------------------------------------------------------
def save_root_tree(series, forecasts, test, test_index, n_lead, fname):
    # number of test samples
    nsamples = test_index.shape[0]
    # number of time steps that we are forecasting
    fmax = len(forecasts[0])
    # open ROOT file
    f = root_open(fname, "recreate")
    # define the model
    class Event(TreeModel):
        data = FloatCol()
        forecast = FloatCol()  #FloatArrayCol(mulmax)
        year = IntCol()
        month = IntCol()
        day = IntCol()
        hour = IntCol()
    # create tree
    t = Tree("heat", model=Event)
    # power column
    s = series.values[:, 1]
    fr = 0 # row
    fj = 0 # col
    # loop over all data  
    for i in range(len(s)):  # i = data index
        t.data = s[i]
        t.year = series.datetime[i].year
        t.month = series.datetime[i].month
        t.day = series.datetime[i].day
        t.hour = series.datetime[i].hour
        
        fi = -1
        if (fr < nsamples): 
            fi = test_index[fr] + n_lead  # fi = forecast index

        fi1 = -1  
        if (fr < nsamples - 1): 
            fi1 = test_index[fr+1] + n_lead  # fi1 = next forecast index

#        print i, t.hour, fi, fj, fr
#        if (t.day > 4):
#            exit()

        t.forecast = 0
        if (i == fi + fj):
            t.forecast = forecasts[fr][fj] 
            if ((fj < fmax - 1) and (i < fi1 or fi1 == -1)):
                fj += 1
            else:
                fj = 0
                fr += 1
            
        # fill event
        t.fill()
    t.write()
    f.close()
