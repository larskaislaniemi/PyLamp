#!/usr/bin/python3 
import numpy as np
from pylamp_const import *
#from bokeh.plotting import figure, output_file, show, save
import vtk
import sys
import gr
from scipy.interpolate import griddata
import os
import glob

###
# program to convert pylamp output (npz files) to plots or vtk files
#
# usage: python3 pylamp_post.py [SCREEN_TRAC_TEMP|VTK] tracsfile griddatafile
#
# currently only the option "VTK" works
###

POSTTYPES = {
        'SCREEN_TRAC_TEMP': 1,
        'VTK':              2
}



if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("Needs at least one argument: type")

    posttype = POSTTYPES[sys.argv[1]]

    if posttype == POSTTYPES['SCREEN_TRAC_TEMP']:

        if len(sys.argv) < 4:
            raise Exception("Needed arguments: type, tracsfile, gridfile")
        
        tracsdatafile = sys.argv[3]
        #griddatafile = sys.argv[2]
        tracsdata = np.load(tracsdatafile)
        #griddata = np.load(griddatafile)

        tr_x = tracsdata["tr_x"]
        tr_f = tracsdata["tr_f"]

        #gridx = griddata["gridx"]
        #gridz = griddata["gridz"]

        stride = 1000

        maxtmp = np.max(tr_f[::stride, TR_TMP])
        mintmp = np.min(tr_f[::stride, TR_TMP])

        gr.setviewport(0.1, 0.95, 0.1, 0.95)
        gr.setwindow(np.min(tr_x[::stride, IX]), np.max(tr_x[::stride, IX]), np.min(tr_x[::stride, IZ]), np.max(tr_x[::stride, IZ]))
        gr.setspace(np.min(tr_f[::stride, TR_TMP]), np.max(tr_f[::stride, TR_TMP]), 0, 90)
        gr.setmarkersize(1)
        gr.setmarkertype(gr.MARKERTYPE_SOLID_CIRCLE)
        gr.setcharheight(0.024)
        gr.settextalign(2, 0)
        gr.settextfontprec(3, 0)

        x = tr_x[::stride, IX]
        y = tr_x[::stride, IZ]
        z = tr_f[::stride, TR_TMP]
        
        grid = np.mgrid[0:1:200j, 0:1:66j]
        Z = griddata((x, y), z, (grid[0], grid[1]), method='cubic')
        X = np.unique(grid[0].flatten())
        Y = np.unique(grid[1].flatten())

        #X, Y, Z = gr.gridit(x, y, z, 200, 200)
        H = np.linspace(mintmp, maxtmp, 20)
        gr.surface(X, Y, Z, 5)
        gr.contour(X, Y, H, Z, 0)
        gr.polymarker(x, y)
        gr.axes(50e3, 50e3, 0, 0, 0, 0, 10e3)

        gr.updatews()
        
    elif posttype == POSTTYPES['VTK']:
        if len(sys.argv) < 3:
            raise Exception("Neede arguments: type, trascfile(s)")

        trfields = [TR_TMP, TR_RHO, TR_ETA, TR_MRK]
        trfieldnames = ["temp", "dens", "visc", "mark"]

        if os.path.isfile(sys.argv[2]):
            fileslist = [sys.argv[2]]
        else:
            fileslist = glob.glob(sys.argv[2])

        fileslist.sort()

        for tracsdatafile in fileslist:
            if os.path.isfile(tracsdatafile + ".vtp"):
                print("skip " + tracsdatafile)
                continue
            else:
                print(tracsdatafile)
            tracsdata = np.load(tracsdatafile)

            tr_x = tracsdata["tr_x"]
            tr_f = tracsdata["tr_f"]

            N = tr_f[:, TR_TMP].shape[0]

            stride = 1

            vpoints = vtk.vtkPoints()
            vvertices = vtk.vtkCellArray()

            for i in range(N):
                id = vpoints.InsertNextPoint(tr_x[i, IX], tr_x[i, IZ], 0*tr_x[i, IX])
                vvertices.InsertNextCell(1)
                vvertices.InsertCellPoint(id)

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vpoints)
            polydata.SetVerts(vvertices)


            for ifield in range(len(trfields)):
                array = vtk.vtkDoubleArray()
                array.SetNumberOfComponents(1)
                array.SetNumberOfTuples(N)

                for i in range(N):
                    array.SetValue(i, tr_f[i, trfields[ifield]])

                array.SetName(trfieldnames[ifield])
                polydata.GetPointData().AddArray(array)

                polydata.Modified()

            if vtk.VTK_MAJOR_VERSION <= 5:
                polydata.Update()

            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetDataModeToBinary()
            writer.SetCompressorTypeToZLib();
            writer.SetFileName(tracsdatafile + ".vtp")
            writer.SetCompressorTypeToZLib()
            if vtk.VTK_MAJOR_VERSION <= 5:
                writer.SetInput(polydata)
            else:
                writer.SetInputData(polydata)
            writer.Write()

    else:
        raise Exception("Undefined plot type")
