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
        'TR_MRK': 5,
        'GR_TMP': 100,
        'GR_RHO': 104,
        'GR_PRS': 106
}

PLOTDESCR = {
        'TR_TMP': "Temperature from tracers",
        'TR_VEL': "Velocity from tracers",
        'TR_MAT': "Material number from tracers",
        'TR_ETA': "Effective viscosity from tracers",
        'TR_RHO': "Effective density from tracers",
        'TR_MRK': "Passive markers from tracers",
        'GR_TMP': "Temperature from Eulerian grid",
        'GR_RHO': "Density from Eulerian grid",
        'GR_PRS': "Pressure from Eulerian grid"
}

def usage():
    print("Usage:")
    print("  python3 pylamp_plot.py [options]")
    print("")
    print(" Options:")
    print("    --plot=... Plot type, see possible values below")
    print("    --it=N     Plot data from timestep N")
    print("    --time=t   Plot data from the first time step that has time > t")
    print("               (Either time or it has to be defined)")
    print("    --nofilter Do not automatically limit number of velocity arrows")
    print("")
    print(" Plot types:")
    for key in PLOTDESCR:
        print("    ", key, ":", PLOTDESCR[key])
    return

SECINMYR = 60*60*24*365.25*1e6

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["it=","time=","plot=","nofilter","help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    it = None
    time = None
    plottype = None

    opt_nofilter = False

    no_opts = True

    for opt, arg in opts:
        no_opts = False
        if opt == "--it":
            it = arg
        elif opt == "--time":
            time = arg
        elif opt == "--plot":
            plottype = arg
        elif opt == "--nofilter":
            opt_nofilter = True
        elif opt in ["--help", "-h"]:
            usage()
            sys.exit(0)

    if no_opts:
        usage()
        exit(0)

    if plottype is None:
        print("Option --plot is mandatory")
        sys.exit(2)

    if it is not None and time is not None:
        print("Define either --it or --time, not both")

    try:
        plottypenum = PLOTTYPES[plottype]
    except:
        print("Invalid plot type:", plottype)
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
    elif time is not None:
        fileslist = sorted(glob.glob("./"+plotsource+".*.npz"))

        filefound = False

        for file in fileslist:
            npzdata = np.load(file)
            thistime = npzdata["time"] / (60*60*24*365.25*1e6)
            if thistime >= float(time):
                filefound = True
                filefoundname = file
                break

        if not filefound:
            print("Cannot find time >", time, "Myrs")
            sys.exit(6)
        else:
            fileslist = [filefoundname]
    else:
        print("No time step or time value defined, exiting")
        sys.exit(6)
    
    for file in fileslist:
        npzdata = np.load(file)

        plt.figure()
        plt.title(plottype + " / time = " + str(npzdata["time"]/SECINMYR) + " Myrs")

        if plottype == 'TR_TMP':
            ntracs = npzdata["tr_x"].shape[0]
            plt.scatter(npzdata["tr_x"][:,IX], -npzdata["tr_x"][:,IZ], c=npzdata["tr_f"][:,TR_TMP], linewidths=0)
            C = plt.colorbar()
            C.set_label("Temperature, K")
        elif plottype == 'TR_VEL':
            ntracs = npzdata["tr_x"].shape[0]
            if opt_nofilter:
                viewntrac = ntracs
            else:
                viewntrac = 2000
            stride = int(ntracs / viewntrac)
            idx = np.arange(0, ntracs, stride)
            plt.quiver(npzdata["tr_x"][idx,IX], -npzdata["tr_x"][idx,IZ], \
                       npzdata["tr_v"][idx,IX], -npzdata["tr_v"][idx,IZ]) # C=tr_f...?
        elif plottype == 'TR_ETA':
            ntracs = npzdata["tr_x"].shape[0]
            plt.scatter(npzdata["tr_x"][:,IX], -npzdata["tr_x"][:,IZ], c=np.log10(npzdata["tr_f"][:,TR_ETA]), linewidths=0)
            C = plt.colorbar()
            C.set_label("Viscosity, log10 Pa s")
        elif plottype == 'TR_RHO':
            ntracs = npzdata["tr_x"].shape[0]
            plt.scatter(npzdata["tr_x"][:,IX], -npzdata["tr_x"][:,IZ], c=npzdata["tr_f"][:,TR_RHO], linewidths=0)
            C = plt.colorbar()
            C.set_label("Density, kg/m3")
        elif plottype == 'TR_MRK':
            ntracs = npzdata["tr_x"].shape[0]
            plt.scatter(npzdata["tr_x"][:,IX], -npzdata["tr_x"][:,IZ], c=npzdata["tr_f"][:,TR_MRK], linewidths=0)
        elif plottype == 'TR_MAT':
            ntracs = npzdata["tr_x"].shape[0]
            plt.scatter(npzdata["tr_x"][:,IX], -npzdata["tr_x"][:,IZ], c=npzdata["tr_f"][:,TR_MAT], linewidths=0)
            C = plt.colorbar()
            C.set_label("Material number")

        elif plottype == 'GR_TMP':
            temp = npzdata["temp"]
            plt.imshow(temp, interpolation="none", extent=[np.min(npzdata["gridx"]),np.max(npzdata["gridx"]),np.min(npzdata["gridz"]),np.max(npzdata["gridz"])])
            plt.colorbar()
        elif plottype == 'GR_RHO':
            temp = npzdata["rho"]
            plt.imshow(temp, interpolation="none", extent=[np.min(npzdata["gridx"]),np.max(npzdata["gridx"]),np.min(npzdata["gridz"]),np.max(npzdata["gridz"])])
            plt.colorbar()
        elif plottype == 'GR_PRS':
            temp = npzdata["pres"]
            plt.imshow(temp, interpolation="none")
            plt.colorbar()

        plt.axes().set_aspect('equal', 'datalim')
        plt.show()


