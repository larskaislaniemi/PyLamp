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

###
# program to convert pylamp output (npz files) to plots or vtk files
#
# usage: python3 pylamp_post.py [SCREEN_TRAC_TEMP|VTKTRAC|VTKGRID] {required filenames ...}
#
# currently only the option "VTKTRAC" works
###

POSTTYPES = {
        'SCREEN_TRAC_TEMP': 1,
        'VTKTRAC':          2,
        'VTKGRID':          3
}



if __name__ == "__main__":
    if len(sys.argv) < 2:
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
        
    elif posttype == POSTTYPES['VTKTRAC']:
        if len(sys.argv) < 3:
            raise Exception("Needed arguments: type, tracfile(s)")

        trfields = [TR_TMP, TR_RHO, TR_ETA, TR_MRK, TR_MAT, TR__ID]
        trfieldnames = ["temp", "dens", "visc", "mark", "mat", "id"]

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

            tr_v_present = True
            tr_x = tracsdata["tr_x"]
            tr_f = tracsdata["tr_f"]
            try:
                tr_v = tracsdata["tr_v"]
            except KeyError:
                tr_v_present = False

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
                trac_array = vtk.vtkDoubleArray()
                trac_array.SetNumberOfComponents(1)
                trac_array.SetNumberOfTuples(N)

                for i in range(N):
                    trac_array.SetValue(i, tr_f[i, trfields[ifield]])

                trac_array.SetName(trfieldnames[ifield])
                polydata.GetPointData().AddArray(trac_array)

                polydata.Modified()

            # special field, velocity
            if tr_v_present:
                trac_array = vtk.vtkDoubleArray()
                trac_array.SetNumberOfComponents(3)
                trac_array.SetNumberOfTuples(N)

                for i in range(N):
                    trac_array.SetTuple3(i, tr_v[i, IX], tr_v[i, IZ], tr_v[i, IZ]*0.0)

                trac_array.SetName("velo")
                polydata.GetPointData().AddArray(trac_array)
                polydata.Modified()


            if vtk.VTK_MAJOR_VERSION <= 5:
                polydata.Update()

            trac_writer = vtk.vtkXMLPolyDataWriter()
            trac_writer.SetDataModeToBinary()
            trac_writer.SetCompressorTypeToZLib();
            trac_writer.SetFileName(tracsdatafile + ".vtp")
            trac_writer.SetCompressorTypeToZLib()
            if vtk.VTK_MAJOR_VERSION <= 5:
                trac_writer.SetInput(polydata)
            else:
                trac_writer.SetInputData(polydata)
            trac_writer.Write()

    elif posttype == POSTTYPES['VTKGRID']:
        if len(sys.argv) < 3:
            raise Exception("Needed arguments: type, gridfile(s)")

        grfields = ["temp", "velz", "velx", "pres", "rho"]
        grfieldnames = ["temp", "velz", "velx", "pres", "rho"]

        if os.path.isfile(sys.argv[2]):
            fileslist = [sys.argv[2]]
        else:
            fileslist = glob.glob(sys.argv[2])

        fileslist.sort()

        for griddatafile in fileslist:
            if os.path.isfile(griddatafile + ".vtk"):
                print("skip " + griddatafile)
                continue
            else:
                print(griddatafile)
            griddata = np.load(griddatafile)

            grid = [[]] * 2
            grid[IZ] = griddata["gridz"]
            grid[IX] = griddata["gridx"]

            N = np.prod(griddata[grfields[0]].shape)
            Ng = len(grid[IZ]) * len(grid[IX])
            assert N == Ng

            stride = 1

            # VTK coords xyz are coords xzy in PyLamp (vtk z = pylamp y = 0 always, 2D)
            arrCoords = [vtk.vtkDoubleArray() for i in range(3)]
            for i in grid[IX]:
                arrCoords[IX].InsertNextValue(i)
            for i in grid[IZ]:
                arrCoords[IZ].InsertNextValue(i)
            arrCoords[IY].InsertNextValue(0)

            vtkgrid = vtk.vtkRectilinearGrid()
            vtkgrid.SetDimensions(len(grid[IX]), len(grid[IZ]), 1)
            vtkgrid.SetXCoordinates(arrCoords[IX])
            vtkgrid.SetYCoordinates(arrCoords[IZ])
            vtkgrid.SetZCoordinates(arrCoords[IY])

            for ifield in range(len(grfields)):
                try:
                    dummy = griddata[grfields[ifield]]
                except:
                    # variable does not exist in output
                    continue
                grid_array = vtk.vtkDoubleArray()
                grid_array.SetNumberOfComponents(1)
                grid_array.SetNumberOfTuples(N)

                for i in range(len(grid[IZ])):
                    for j in range(len(grid[IX])):
                        grid_array.SetTuple1(i*len(grid[IX]) + j, griddata[grfields[ifield]][i, j])

                grid_array.SetName(grfieldnames[ifield])

                vtkgrid.GetPointData().AddArray(grid_array)

            grid_writer = vtk.vtkRectilinearGridWriter()
            grid_writer.SetFileName(griddatafile + ".vtk")
            if vtk.VTK_MAJOR_VERSION <= 5:
                grid_writer.SetInput(vtkgrid)
            else:
                grid_writer.SetInputData(vtkgrid)
            grid_writer.Write()
            

    else:
        raise Exception("Undefined plot type")
