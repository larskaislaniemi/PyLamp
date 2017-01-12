#!/usr/bin/python3 
import sys

sys.path.append('/usr/lib/python3.5/site-packages')

import numpy as np
from pylamp_const import *
#from bokeh.plotting import figure, output_file, show, save
import vtk
#import gr
from scipy.interpolate import griddata
import os
import glob
import sys, getopt
import matplotlib.pyplot as plt

###
# program to plot pylamp output (npz files) 
###

PLOTTYPES = {
        'TR_TMP': 0,
        'TR_VEL': 1,
        'TR_MAT': 2,
        'TR_ETA': 3,
        'TR_RHO': 4,
        'TR_MRK': 5
}


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["it=","time=","plot=","help"])
    except getopt.GetoptError:
        print("Usage:")
        print("  python3 pylamp_plot.py [options]")
        print("")
        print(" Options:")
        print("    --plot={TR_TMP, TR_VEL, TR_MAT, TR_ETA, TR_RHO, TR_MRK}")
        print("    --it=N    Plot data from timestep N")
        print("    --time=t  Plot data from the first time step that has time > t")
        sys.exit(2)

    it = None
    time = None
    plottype = None

    for opt, arg in opts:
        if opt == "--it":
            it = arg
        elif opt == "--time":
            time = arg
        elif opt == "--plot":
            plottype = arg
        elif opt in ["--help", "-h"]:
            print("Usage:")
            print("  python3 pylamp_plot.py [options]")
            print("")
            print(" Options:")
            print("    --plot={TR_TMP, TR_VEL, TR_MAT, TR_ETA, TR_RHO, TR_MRK}")
            print("    --it=N    Plot data from timestep N")
            print("    --time=t  Plot data from the first time step that has time > t")
            sys.exit(0)


    if plottype is None:
        print("Option --plot is mandatory")
        sys.exit(2)

    if it is not None and time is not None:
        print("Define either --it or --time, not both")

    try:
        plottypenum = PLOTTYPES[plottype]
    except:
        print("Invalid plot type:", plotname)
        sys.exit(3)

    if plottypenum < 100:
        plotsource = 'tracs'
    else:
        plotsource = 'griddata'

    if it is not None:
        if os.path.isfile((plotsource + ".{:06d}.npz").format(int(it))):
            fileslist = [(plotsource + ".{:06d}.npz").format(int(it))]
        else:
            print("Time step", it, "not found")
            sys.exit(5)
    else:
        fileslist = glob.glob("./"+plotsource+".*.npz")

        filefound = False

        for file in fileslist:
            npzdata = np.load(file)
            thistime = npzdata["time"] / (60*60*24*365.25*1e6)
            if thistime >= time:
                filefound = True
                filefoundname = file
                break

        if not filefound:
            print("Cannot find time", time, "Myrs")
            sys.exit(6)
        else:
            fileslist = [filefoundname]

    
    for file in fileslist:
        npzdata = np.load(file)

        ntracs = npzdata["tr_x"].shape[0]

        if plottype == 'TR_TMP':
            pass
        elif plottype == 'TR_VEL':
            pass
        elif plottype == 'TR_ETA':
            pass
        elif plottype == 'TR_RHO':
            plt.figure()
            plt.scatter(npzdata["tr_x"][:,IX], -npzdata["tr_x"][:,IZ], c=npzdata["tr_f"][:,TR_RHO])
            plt.show()
        elif plottype == 'TR_MRK':
            pass


