#!/usr/bin/python3

from pylamp_const import *
from scipy.sparse import lil_matrix
import numpy as np
import sys

BC_TYPE_FIXTEMP = 0
BC_TYPE_FIXFLOW = 1

def gidx(idxs, nx, dim):
    # Global index for the matrix in linear system of equations
    #   gidx = ...  +  iz * ny * nx * DIM  +  ix * ny * DIM  +  iy * DIM  +  ieq
    # idxs is a list of integers (length DIM) or 1D numpy arrays (all of same
    # length), or a combination

    if len(idxs) != dim:
        raise Exception("num of idxs != dimensions")
    if dim == 2:
        ret = idxs[IZ] * nx[IX] + idxs[IX] 
    else:
        print("!!! NOT IMPLEMENTED")

    return ret


def numOfZeroRows(a, c):
    if DEBUG > 5:
        a = np.sum(np.abs(a), axis=1)
        b = np.sum(a == 0)
        print (" # of zero rows: ", b, "/", a.shape[0], "/", a.shape[0]-c)
    return

def printmatrix(arr, nx):
    sys.stdout.write('          ')
    for j in range(arr.shape[1]):
        if (j+1) % 10 == 0:
            sys.stdout.write('|')
        elif (j+1) % 2 == 0:
            sys.stdout.write('.')
        else:
            sys.stdout.write(' ')
    sys.stdout.write("\n")

    for i in range(arr.shape[0]):
        ieq = i % 3
        inode = int(i / 3)
        irow = int(inode / nx[1])
        icol = inode % nx[1]
        sys.stdout.write("{:>3}=".format(i))
        sys.stdout.write("{:>1}–".format(irow))
        sys.stdout.write("{:>1}–".format(icol))
        sys.stdout.write("{:>1}".format(ieq))
        sys.stdout.write('|')
        if np.sum(np.abs(arr[i,:])) == 0:
            for j in range(arr.shape[1]):
                sys.stdout.write('!')
        else:
            for j in range(arr.shape[1]):
                if arr[i,j] > 0:
                    sys.stdout.write('+')
                elif arr[i,j] < 0:
                    sys.stdout.write('–')
                else:
                    if i == j:
                        sys.stdout.write('O')
                    else:
                        sys.stdout.write(' ')
        sys.stdout.write('|')
        sys.stdout.write("\n")
    sys.stdout.flush()


def x2t(x, nx):
    # reshape solution from diffusion solver to temperature mesh

    dof = np.prod(nx)

    newtemp = x.reshape(nx)

    return newtemp

def makeDiffusionMatrix(nx, grid, f_kx, f_kz, f_Cp, f_rho, bctype, bcvalue):
    # Form the solution matrix for diffusion eq
    #
    # Currently can do only 2D
    #

    dof = np.prod(nx)
    A   = lil_matrix((dof, dof))
    rhs = np.zeros(dof)

    #### boundaries: ####

    # at z = 0
    i = 0

    if bc[DIM*0 + IZ] == BC_TYPE_FIXTEMP:
        j = np.arange(0, nx[IX])
        A[gidx([i, j], nx, DIM), gidx([i, j], nx, DIM)] = 1
        rhs[gidx([i, j], nx, DIM)] = bcvalue[DIM*0 + IZ]

    elif bc[DIM*0 + IZ] == BC_TYPE_FIXFLOW:
        j = np.arange(0, nx[IX])
        A[gidx([i, j], nx, DIM), gidx([i+1, j], nx, DIM)] = f_kz[i, j] / (grid[IZ][i+1] - grid[IZ][i])
        A[gidx([i, j], nx, DIM), gidx([i, j], nx, DIM)] = -f_kz[i, j] / (grid[IZ][i+1] - grid[IZ][i])
        rhs[gidx([i, j], nx, DIM)] = bcvalue[DIM*0 + IZ]

    # at z = zL
    i = nx[IZ]-1

    if bc[DIM*1 + IZ] == BC_TYPE_FIXTEMP:
        j = np.arange(0, nx[IX])
        A[gidx([i, j], nx, DIM), gidx([i, j], nx, DIM)] = 1
        rhs[gidx([i, j], nx, DIM)] = bcvalue[DIM*0 + IZ]

    elif bc[DIM*1 + IZ] == BC_TYPE_FIXFLOW:
        j = np.arange(0, nx[IX])
        A[gidx([i, j], nx, DIM), gidx([i, j], nx, DIM)] = f_kz[i-1, j] / (grid[IZ][i] - grid[IZ][i-1])
        A[gidx([i, j], nx, DIM), gidx([i-1, j], nx, DIM)] = -f_kz[i-1, j] / (grid[IZ][i] - grid[IZ][i-1])
        rhs[gidx([i, j], nx, DIM)] = bcvalue[DIM*1 + IZ]

    # at x = 0
    j = 0

    if bc[DIM*0 + IX] == BC_TYPE_FIXTEMP:
        i = np.arange(0, nx[IZ])
        A[gidx([i, j], nx, DIM), gidx([i, j], nx, DIM)] = 1
        rhs[gidx([i, j], nx, DIM)] = bcvalue[DIM*0 + IX]

    elif bc[DIM*0 + IX] == BC_TYPE_FIXFLOW:
        i = np.arange(0, nx[IZ])
        A[gidx([i, j], nx, DIM), gidx([i, j+1], nx, DIM)] = f_kx[i, j] / (grid[IX][j+1] - grid[IX][j])
        A[gidx([i, j], nx, DIM), gidx([i, j], nx, DIM)] = -f_kx[i, j] / (grid[IX][j+1] - grid[IX][j])
        rhs[gidx([i, j], nx, DIM)] = bcvalue[DIM*0 + IX]

    # at z = zL
    j = nx[IX]-1

    if bc[DIM*1 + IX] == BC_TYPE_FIXTEMP:
        i = np.arange(0, nx[IZ])
        A[gidx([i, j], nx, DIM), gidx([i, j], nx, DIM)] = 1
        rhs[gidx([i, j], nx, DIM)] = bcvalue[DIM*0 + IX]

    elif bc[DIM*1 + IX] == BC_TYPE_FIXFLOW:
        i = np.arange(0, nx[IZ])
        A[gidx([i, j], nx, DIM), gidx([i, j], nx, DIM)] = f_kx[i, j-1] / (grid[IX][j] - grid[IX][j-1])
        A[gidx([i, j], nx, DIM), gidx([i-1, j], nx, DIM)] = -f_kz[i, j-1] / (grid[IX][j] - grid[IX][j-1])
        rhs[gidx([i, j], nx, DIM)] = bcvalue[DIM*1 + IX]


    ### rest of the points
    
    iset = np.arange(1, nx[IZ]-1)
    jset = np.arange(1, nx[IX]-1)
    ijset = np.meshgrid(iset, jset)
    i = ijset[IZ].flatten()
    j = ijset[IX].flatten()
    
    mat_row = gidx([i, j], nx, DIM)

    precoef = tstep / (f_rho[i, j] * f_Cp[i, j])

    A[mat_row, gidx([i  , j+1], nx, DIM)] = f_kx[i  , j  ] / (grid[IX][j+1] - grid[IX][j]) / (grid[IX][j] - grid[IX][j-1])
    A[mat_row, gidx([i  , j-1], nx, DIM)] = f_kx[i  , j-1] / (grid[IX][j] - grid[IX][j-1]) / (grid[IX][j] - grid[IX][j-1])
    A[mat_row, gidx([i+1, j  ], nx, DIM)] = f_kz[i  , j  ] / (grid[IZ][i+1] - grid[IZ][i]) / (grid[IZ][i] - grid[IZ][i-1])
    A[mat_row, gidx([i-1, j  ], nx, DIM)] = f_kz[i-1, j  ] / (grid[IZ][i] - grid[IX][i-1]) / (grid[IZ][i] - grid[IZ][i-1])
    A[mat_row, gidx([i  , j  ], nx, DIM)] = \
            -f_kx[i  , j  ] / (grid[IX][j+1] - grid[IX][j]) / (grid[IX][j] - grid[IX][j-1]) + \
            -f_kx[i  , j-1] / (grid[IX][j] - grid[IX][j-1]) / (grid[IX][j] - grid[IX][j-1]) + \
            -f_kz[i  , j  ] / (grid[IZ][i+1] - grid[IZ][i]) / (grid[IZ][i] - grid[IZ][i-1]) + \
            -f_kz[i-1, j  ] / (grid[IZ][i] - grid[IZ][i-1]) / (gird[IZ][i] - grid[IZ][i-1])

    rhs[mat_row] = 0
                
    return (A, rhs)
