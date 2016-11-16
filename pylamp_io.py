#!/usr/bin/python3

from pylamp_const import *
import numpy as np

def vtkOutPoints(x, fielddata, fieldnames, filename):

    dims = x.shape[1]
    N = x.shape[0]
    nfields = fielddata.shape[1]

    assert (dims == 2 or dims == 3)

    fd = open(filename, 'w')
    fd.write("# vtk DataFile Version 3.0\n")
    fd.write("PyLamp tracer field\n")
    fd.write("ASCII\n")
    fd.write("DATASET POLYDATA\n")
    fd.write("FIELD FieldData 2\n")
    fd.write("Time_Ma 1 1 double\n")
    fd.write(" 0.0\n")
    fd.write("Step 1 1 int\n")
    fd.write(" 0\n")
    fd.write("POINTS " + str(N) + " double\n")

    for ip in range(N):
        fd.write(str(x[ip,1]) + " ")
        fd.write(str(x[ip,0]) + " ")
        if dims == 3:
            fd.write(str(x[ip,2]) + " ")
        else:
            fd.write("0.0 ")
        fd.write("\n")

    fd.write("POINT_DATA " + str(N) + "\n")

    fd.write("FIELD FieldData " + str(nfields) + "\n")
    for i in range(nfields):
        fd.write(fieldnames[i] + " 1 " + str(N) + " double\n")
        for ip in range(N):
            fd.write(str(fielddata[ip,i]) + " ")
        fd.write("\n")

    fd.close()

    return
