#!/usr/bin/python3 
import numpy as np
from pylamp_const import *
#from bokeh.plotting import figure, output_file, show, save
import vtk
import sys

POSTTYPE_SCREEN_TRAC_TEMP = 1
POSTTYPE_VTK = 2

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("Needs two arguments, type and file")

    posttype = eval(sys.argv[1])
    tracsfilename = sys.argv[2]

    #it = 1352
    #filename = "tracs.{:06}.npz".format(it)
    #posttype = POSTTYPE_VTK

    if posttype == POSTTYPE_SCREEN_TRAC_TEMP:
        raise Exception("Not working")
        
        tracsdatafile = filename
        tracsdata = np.load(tracsdatafile)

        tr_x = tracsdata["tr_x"]
        tr_f = tracsdata["tr_f"]

        #griddatafile = "griddata.{:06d}.npz".format(it)
        # TODO: This
        griddata = np.load(griddatafile)

        gridx = griddata["gridx"]
        gridz = griddata["gridz"]

        stride = 10

        maxtmp = np.max(tr_f[::stride, TR_TMP])

        #colors = ["#000000" for i in range(tr_x[::stride].shape[0])] 
        colors = ["#%02x%02x%02x" % (int(255*(1-t)), int(255*((t-0.5)**2)), int(255*t)) for t in tr_f[::stride, TR_TMP]/maxtmp]

        output_file("fig_tractemp_{:06d}.html".format(it), title="Temperature field in tracers", mode="cdn")
        TOOLS = "resize,crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"
        p = figure(tools=TOOLS, x_range=(0,2000e3), y_range=(0,660e3))
        p.scatter(tr_x[::stride,IX], tr_x[::stride,IZ], fill_color=colors, line_color=None)

        #show(p)
        save(p)
        
    elif posttype == POSTTYPE_VTK:

        tracsdatafile = tracsfilename
        tracsdata = np.load(tracsdatafile)

        tr_x = tracsdata["tr_x"]
        tr_f = tracsdata["tr_f"]

        N = tr_f[:, TR_TMP].shape[0]

        stride = 200

        vpoints = vtk.vtkPoints()
        vvertices = vtk.vtkCellArray()

        for i in range(N):
            id = vpoints.InsertNextPoint(tr_x[i, IX], tr_x[i, IZ], 0*tr_x[i, IX])
            vvertices.InsertNextCell(1)
            vvertices.InsertCellPoint(id)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vpoints)
        polydata.SetVerts(vvertices)

        array = vtk.vtkDoubleArray()
        array.SetNumberOfComponents(1)
        array.SetNumberOfTuples(N)

        for i in range(N):
            array.SetValue(i, tr_f[i, TR_TMP])

        array.SetName("temperature")
        polydata.GetPointData().AddArray(array)

        polydata.Modified()

        if vtk.VTK_MAJOR_VERSION <= 5:
            polydata.Update()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetDataModeToBinary()
        writer.SetCompressorTypeToZLib();
        writer.SetFileName(tracsfilename + ".vtp")
        writer.SetCompressorTypeToZLib()
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)
        writer.Write()

    else:
        raise Exception("Undefined plot type")
