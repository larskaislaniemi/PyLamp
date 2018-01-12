#!/usr/bin/python3

from pylamp_const import *
from scipy.sparse import lil_matrix
import numpy as np
import sys

###
# subroutines to build the system of linear equations to solve the stokes 
# and continuity equations
#
# variable viscosity
#
# 2D implemented, formulation "3D ready"
###

BC_TYPE_NOSLIP = 1 
BC_TYPE_FREESLIP = 2
BC_TYPE_CYCLIC = 4
BC_TYPE_FLOWTHRU = 128

def gidx(idxs, nx, dim):
    # Global index for the matrix in linear system of equations
    #   gidx = ...  +  iz * ny * nx * (DIM+1)  +  ix * ny * (DIM+1)  +  iy * (DIM+1)  +  ieq
    # idxs is a list of integers (length DIM) or 1D numpy arrays (all of same
    # length), or a combination

    if len(idxs) != dim:
        raise Exception("num of idxs != dimensions")
    if dim == 2:
        ret = idxs[IZ] * nx[IX] * (dim+1) + idxs[IX] * (dim+1)
    else:
        print("!!! NOT IMPLEMENTED")

    return ret

#return np.sum([idxs[i] * np.prod(nx[(i+1):dim]) * (dim+1) for i in range(dim)],dtype=np.int)

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


def x2vp(x, nx):
    # split solution from stokes solver to vel and pres fields
    # and remove the ghost nodes

    dof = np.prod(nx)*(DIM+1)

    newvel = [[]] * DIM
    for d in range(DIM):
        newvel[d] = x[range(d, dof, DIM+1)].reshape(nx)
    newpres = x[range(DIM, dof, DIM+1)].reshape(nx)

    #for d in range(DIM):
    #    newvel[d] = np.delete(newvel[d], nx[d]-1, axis=d)
    #    newpres = np.delete(newpres, nx[d]-1, axis=d)

    return (newvel, newpres)


def makeStokesMatrix(nx, grid, gridmp, f_etas, f_etan, f_rho, bc, bcvals=None, surfstab=False, tstep=None, surfstab_theta=0.5):
    # Form the solution matrix for stokes/cont solving
    #
    # Currently can do only 2D
    #

    dof = np.prod(nx) * (DIM + 1)
    A   = lil_matrix((dof, dof))
    lc  = np.zeros(dof)
    rhs = np.zeros(dof)

    # calc scaling coeffs
    minetas = np.min(f_etas)
    minetan = np.min(f_etan)
    mineta = min(minetas, minetan)
    avgdx = (grid[IX][-1] - grid[IX][0]) / grid[IX].shape[0]
    avgdz = (grid[IZ][-1] - grid[IZ][0]) / grid[IZ].shape[0]
    Kcont = 2*mineta / (avgdx + avgdz)
    Kbond = 4*mineta / (avgdx + avgdz)**2



    #### ghost points: ####

    j = nx[IX]-1
    i = np.arange(0, nx[IZ])

    # force vy and P to zero
    A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont
    rhs[gidx([i, j], nx, DIM) + IZ] = 0
    lc[gidx([i, j], nx, DIM) + IZ] += 1

    A[gidx([i, j], nx, DIM) + IP, gidx([i, j], nx, DIM) + IP] = Kcont
    rhs[gidx([i, j], nx, DIM) + IP] = 0
    lc[gidx([i, j], nx, DIM) + IP] += 1

    j = np.arange(nx[IX])
    i = nx[IZ]-1


    # force vx and P to zero
    A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = Kcont
    rhs[gidx([i, j], nx, DIM) + IX] = 0
    lc[gidx([i, j], nx, DIM) + IX] += 1

    j = np.arange(nx[IX]-1)
    A[gidx([i, j], nx, DIM) + IP, gidx([i, j], nx, DIM) + IP] = Kcont
    rhs[gidx([i, j], nx, DIM) + IP] = 0
    lc[gidx([i, j], nx, DIM) + IP] += 1


    
    #### boundaries: ####


    # at z = 0
    i = 0

    # vx
    j = np.arange(1, nx[IX]-1)
    if bc[DIM*0 + IZ] & BC_TYPE_NOSLIP:
        # vx extrapolated to be zero from two internal nodes
        #coordGrad = (gridmp[IZ][i] - grid[IZ][i]) / (gridmp[IZ][i+1] - gridmp[IZ][i])
        #A[gidx([i, j], nx, DIM) + IX, gidx([i,   j], nx, DIM) + IX] = Kcont * (1.0 + coordGrad)
        #A[gidx([i, j], nx, DIM) + IX, gidx([i+1, j], nx, DIM) + IX] = Kcont * (-coordGrad)
        A[gidx([i, j], nx, DIM) + IX, gidx([i,   j], nx, DIM) + IX] = Kcont * (-1 / (grid[IZ][i+2] - grid[IZ][i]) + (-1) / (grid[IZ][i+1] - grid[IZ][i])) 
        A[gidx([i, j], nx, DIM) + IX, gidx([i+1, j], nx, DIM) + IX] = Kcont * (1 / (grid[IZ][i+2] - grid[IZ][i]))
        lc[gidx([i, j], nx, DIM) + IX] += 1
        rhs[gidx([i, j], nx, DIM) + IX] = 0

    elif bc[DIM*0 + IZ] & BC_TYPE_FREESLIP:
        # vx equals to vx in grid point next to bnd
        A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = Kcont
        A[gidx([i, j], nx, DIM) + IX, gidx([i+1, j], nx, DIM) + IX] = -Kcont
        lc[gidx([i, j], nx, DIM) + IX] += 1
        rhs[gidx([i, j], nx, DIM) + IX] = 0

    elif bc[DIM*0 + IZ] & BC_TYPE_CYCLIC:
        A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = Kcont
        A[gidx([i, j], nx, DIM) + IX, gidx([nx[IZ]-1, j], nx, DIM) + IX] = -Kcont
        lc[gidx([i, j], nx, DIM) + IX] += 1
        rhs[gidx([i, j], nx, DIM) + IX] = 0

    # vz
    j = np.arange(0, nx[IX]-1)
    if bc[DIM*0 + IZ] & BC_TYPE_CYCLIC:
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont
        A[gidx([i, j], nx, DIM) + IZ, gidx([nx[IZ]-1, j], nx, DIM) + IZ] = -Kcont
        lc[gidx([i, j], nx, DIM) + IZ] += 1
        rhs[gidx([i, j], nx, DIM) + IZ] = 0
    else:
        # vz = 0, no flowing through the boundary
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont
        lc[gidx([i, j], nx, DIM) + IZ] += 1
        rhs[gidx([i, j], nx, DIM) + IZ] = 0


    # at z = Lz
    i = nx[IZ]-1
        
    # vx
    j = np.arange(1, nx[IX]-1)
    if bc[DIM*1 + IZ] & BC_TYPE_NOSLIP:
        # vx interpolated to be zero from one internal and one external node
        #A[gidx([i-1, j], nx, DIM) + IX, gidx([i-1, j], nx, DIM) + IX] = Kcont * 0.5
        #A[gidx([i-1, j], nx, DIM) + IX, gidx([i-2, j], nx, DIM) + IX] = Kcont * 0.5
        A[gidx([i-1, j], nx, DIM) + IX, gidx([i-1, j], nx, DIM) + IX] = Kcont * (-1 / (grid[IZ][i-2] - grid[IZ][i]) + (-1) / (grid[IZ][i-1] - grid[IZ][i]))
        A[gidx([i-1, j], nx, DIM) + IX, gidx([i-2, j], nx, DIM) + IX] = Kcont * (1 / (grid[IZ][i-2] - grid[IZ][i]))
        lc[gidx([i-1, j], nx, DIM) + IX] += 1
        rhs[gidx([i-1, j], nx, DIM) + IX] = 0

    elif bc[DIM*1 + IZ] & BC_TYPE_FREESLIP:
        # vx equals to vx in grid point next to bnd
        A[gidx([i-1, j], nx, DIM) + IX, gidx([i-1, j], nx, DIM) + IX] = Kcont
        A[gidx([i-1, j], nx, DIM) + IX, gidx([i-2, j], nx, DIM) + IX] = -Kcont
        lc[gidx([i-1, j], nx, DIM) + IX] += 1
        rhs[gidx([i-1, j], nx, DIM) + IX] = 0

    elif bc[DIM*1 + IZ] & BC_TYPE_CYCLIC:
        A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = Kcont
        A[gidx([i, j], nx, DIM) + IX, gidx([0, j], nx, DIM) + IX] = -Kcont
        lc[gidx([i, j], nx, DIM) + IX] += 1
        rhs[gidx([i, j], nx, DIM) + IX] = 0

    # vz
    j = np.arange(0, nx[IX]-1)
    if bc[DIM*1 + IZ] == BC_TYPE_CYCLIC:
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont
        A[gidx([i, j], nx, DIM) + IZ, gidx([0, j], nx, DIM) + IZ] = -Kcont
        lc[gidx([i, j], nx, DIM) + IZ] += 1
        rhs[gidx([i, j], nx, DIM) + IZ] = 0
    else:
        # vz = 0
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont
        lc[gidx([i, j], nx, DIM) + IZ] += 1
        rhs[gidx([i, j], nx, DIM) + IZ] = 0



    # BND x = 0
    j = 0

    # vz
    i = np.arange(1, nx[IZ]-1)
    if bc[DIM*0 + IX] & BC_TYPE_NOSLIP:
        # vz extrapolated to be zero from two internal nodes
        #coordGrad = (gridmp[IX][j] - grid[IX][j]) / (gridmp[IX][j+1] - gridmp[IX][j])
        #A[gidx([i, j], nx, DIM) + IZ, gidx([i, j  ], nx, DIM) + IZ] = Kcont * (1.0 + coordGrad)
        #A[gidx([i, j], nx, DIM) + IZ, gidx([i, j+1], nx, DIM) + IZ] = Kcont * (-coordGrad)
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont * (-1 / (grid[IX][j+2] - grid[IX][j]) + (-1) / (grid[IX][j+1] - grid[IX][j])) 
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j+1], nx, DIM) + IZ] = Kcont * (1 / (grid[IX][j+2] - grid[IX][j]))
        lc[gidx([i, j], nx, DIM) + IZ] += 1
        rhs[gidx([i, j], nx, DIM) + IZ] = 0

    elif bc[DIM*0 + IX] & BC_TYPE_FREESLIP:
        # vz equals to vz in grid point next to bnd
        i = np.arange(1, nx[IZ]-1)
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j+1], nx, DIM) + IZ] = -Kcont
        lc[gidx([i, j], nx, DIM) + IZ] += 1
        rhs[gidx([i, j], nx, DIM) + IZ] = 0

    elif bc[DIM*0 + IX] & BC_TYPE_CYCLIC:
        i = np.arange(1, nx[IZ]-1)
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, nx[IX]-1], nx, DIM) + IZ] = -Kcont
        lc[gidx([i, j], nx, DIM) + IZ] += 1
        rhs[gidx([i, j], nx, DIM) + IZ] = 0

    # vx
    i = np.arange(0, nx[IZ]-1)    
    if bc[DIM*0 + IX] & BC_TYPE_CYCLIC:
        A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = -Kcont
        A[gidx([i, j], nx, DIM) + IX, gidx([i, nx[IX]-1], nx, DIM) + IX] = Kcont
        lc[gidx([i, j], nx, DIM) + IX] += 1
        rhs[gidx([i, j], nx, DIM) + IX] = 0
    elif bc[DIM*0 + IX] & BC_TYPE_FLOWTHRU:
        # zero normal stress (dvx/dx=0) except where velocity predefined
        if bcvals is not None and bcvals[DIM*0 + IX] is not None:
            zerostressidx = np.isnan(bcvals[DIM*0 + IX]) # in these locations zero stress will be applied
            definedvelidx = np.logical_not(zerostressidx)# in these the given velocity will be applied 
            i_zs = i[zerostressidx[i]]
            i_dv = i[definedvelidx[i]]
            
            A[gidx([i_zs, j], nx, DIM) + IX, gidx([i_zs, j], nx, DIM) + IX] = -Kcont
            A[gidx([i_zs, j], nx, DIM) + IX, gidx([i_zs, j+1], nx, DIM) + IX] = Kcont
            lc[gidx([i_zs, j], nx, DIM) + IX] += 1
            rhs[gidx([i_zs, j], nx, DIM) + IX] = 0

            A[gidx([i_dv, j], nx, DIM) + IX, gidx([i_dv, j], nx, DIM) + IX] = Kcont
            lc[gidx([i_dv, j], nx, DIM) + IX] += 1
            rhs[gidx([i_dv, j], nx, DIM) + IX] = Kcont * (bcvals[DIM*0 + IX])[i_dv]
        else:
            A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = -Kcont
            A[gidx([i, j], nx, DIM) + IX, gidx([i, j+1], nx, DIM) + IX] = Kcont
            lc[gidx([i, j], nx, DIM) + IX] += 1
            rhs[gidx([i, j], nx, DIM) + IX] = 0
    else:
        # vx = 0
        A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = Kcont
        lc[gidx([i, j], nx, DIM) + IX] += 1
        rhs[gidx([i, j], nx, DIM) + IX] = 0


    ### BND x = Lx
    j = nx[IX]-1

    # vz
    i = np.arange(1, nx[IZ]-1)
    if bc[DIM*1 + IX] & BC_TYPE_NOSLIP:
        ## vz interpolated to be zero from one internal and one external node
        #A[gidx([i, j-1], nx, DIM) + IZ, gidx([i, j-1], nx, DIM) + IZ] = Kcont * 0.5
        #A[gidx([i, j-1], nx, DIM) + IZ, gidx([i, j-2], nx, DIM) + IZ] = Kcont * 0.5
        A[gidx([i, j-1], nx, DIM) + IZ, gidx([i, j-1], nx, DIM) + IZ] = Kcont * (-1 / (grid[IX][j-2] - grid[IX][j]) + (-1) / (grid[IX][j-1] - grid[IX][j]))
        A[gidx([i, j-1], nx, DIM) + IZ, gidx([i, j-2], nx, DIM) + IZ] = Kcont * (1 / (grid[IX][j-2] - grid[IX][j]))
        lc[gidx([i, j-1], nx, DIM) + IZ] += 1 
        rhs[gidx([i, j-1], nx, DIM) + IZ] = 0

    elif bc[DIM*1 + IX] & BC_TYPE_FREESLIP:
        # vz equals to vz in grid point next to bnd
        A[gidx([i, j-1], nx, DIM) + IZ, gidx([i, j-1], nx, DIM) + IZ] = Kcont
        A[gidx([i, j-1], nx, DIM) + IZ, gidx([i, j-2], nx, DIM) + IZ] = -Kcont
        lc[gidx([i, j-1], nx, DIM) + IZ] += 1
        rhs[gidx([i, j-1], nx, DIM) + IZ] = 0

    elif bc[DIM*1 + IX] & BC_TYPE_CYCLIC:
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, 0], nx, DIM) + IZ] = -Kcont
        lc[gidx([i, j], nx, DIM) + IZ] += 1
        rhs[gidx([i, j], nx, DIM) + IZ] = 0

    # vx
    i = np.arange(0, nx[IZ]-1)            
    if bc[DIM*1 + IX] & BC_TYPE_CYCLIC:
        A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = -Kcont
        A[gidx([i, j], nx, DIM) + IX, gidx([i, 0], nx, DIM) + IX] = Kcont
        lc[gidx([i, j], nx, DIM) + IX] += 1
        rhs[gidx([i, j], nx, DIM) + IX] = 0
    elif bc[DIM*1 + IX] & BC_TYPE_FLOWTHRU:
        # zero normal stress (dvx/dx=0) except where velocity predefined
        if bcvals is not None and bcvals[DIM*1 + IX] is not None:
            zerostressidx = np.isnan(bcvals[DIM*1 + IX]) # in these locations zero stress will be applied
            definedvelidx = np.logical_not(zerostressidx)# in these the given velocity will be applied 
            i_zs = i[zerostressidx[i]]
            i_dv = i[definedvelidx[i]]

            A[gidx([i_zs, j], nx, DIM) + IX, gidx([i_zs, j-1], nx, DIM) + IX] = -Kcont
            A[gidx([i_zs, j], nx, DIM) + IX, gidx([i_zs, j], nx, DIM) + IX] = Kcont
            lc[gidx([i_zs, j], nx, DIM) + IX] += 1
            rhs[gidx([i_zs, j], nx, DIM) + IX] = 0

            A[gidx([i_dv, j], nx, DIM) + IX, gidx([i_dv, j], nx, DIM) + IX] = Kcont
            lc[gidx([i_dv, j], nx, DIM) + IX] += 1
            rhs[gidx([i_dv, j], nx, DIM) + IX] = Kcont * (bcvals[DIM*1 + IX])[i_dv]
        else:
            A[gidx([i, j], nx, DIM) + IX, gidx([i, j-1], nx, DIM) + IX] = -Kcont
            A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = Kcont
            lc[gidx([i, j], nx, DIM) + IX] += 1
            rhs[gidx([i, j], nx, DIM) + IX] = 0
    else:
        # vx = 0
        A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = Kcont
        lc[gidx([i, j], nx, DIM) + IX] += 1
        rhs[gidx([i, j], nx, DIM) + IX] = 0



    ### continuity at the boundaries,
    #   excluding corners

    arrmask = np.empty(nx)
    arrmask[:,:] = False

    j = np.arange(1, nx[IX]-2)
    for i in [0, nx[IZ]-2]:
        arrmask[i, j] = 2

    i = np.arange(1, nx[IZ]-2)
    for j in [0, nx[IX]-2]:
        arrmask[i, j] = 3

    idxlist = np.where(arrmask)
    i = idxlist[IZ]
    j = idxlist[IX]

    mat_row = gidx([i, j], nx, DIM) + IP
    A[mat_row, gidx([i, j+1], nx, DIM) + IX] =  Kcont / (grid[IX][j+1] - grid[IX][j])
    A[mat_row, gidx([i, j], nx, DIM) + IX] = -Kcont / (grid[IX][j+1] - grid[IX][j])
    A[mat_row, gidx([i+1, j], nx, DIM) + IZ] =  Kcont / (grid[IZ][i+1] - grid[IZ][i])
    A[mat_row, gidx([i, j], nx, DIM) + IZ] = -Kcont / (grid[IZ][i+1] - grid[IZ][i])
    lc[mat_row] += 1
    rhs[mat_row] = 0


    ### corners, horizontal symmetry for pressure
    for i in [0, nx[IZ]-2]:
        j = 0
        A[gidx([i, j], nx, DIM) + IP, gidx([i, j+1], nx, DIM) + IP] =  Kbond
        A[gidx([i, j], nx, DIM) + IP, gidx([i, j  ], nx, DIM) + IP] = -Kbond
        lc[gidx([i, j], nx, DIM) + IP] += 1
        rhs[gidx([i, j], nx, DIM) + IP] = 0

        j = nx[IX]-2
        A[gidx([i, j], nx, DIM) + IP, gidx([i, j-1], nx, DIM) + IP] =  Kbond
        A[gidx([i, j], nx, DIM) + IP, gidx([i, j  ], nx, DIM) + IP] = -Kbond
        lc[gidx([i, j], nx, DIM) + IP] += 1
        rhs[gidx([i, j], nx, DIM) + IP] = 0


    ### rest of the points

    # ::: z-stokes :::

    iset = np.arange(1, nx[IZ]-1)
    jset = np.arange(1, nx[IX]-2)
    ijset = np.meshgrid(iset, jset)
    i = ijset[IZ].flatten()
    j = ijset[IX].flatten()

    ieq = IZ
    mat_row = gidx([i, j], nx, DIM) + ieq

    # vy_j+½_i
    A[mat_row, gidx([i  , j  ], nx, DIM) + IZ] = \
            -4 * f_etan[i,   j] / (grid[IZ][i+1] - grid[IZ][i  ]) / (grid[IZ][i+1] - grid[IZ][i-1]) + \
            -4 * f_etan[i-1, j] / (grid[IZ][i  ] - grid[IZ][i-1]) / (grid[IZ][i+1] - grid[IZ][i-1]) + \
            -2 * f_etas[i, j+1] / (grid[IX][j+2] - grid[IX][j  ]) / (grid[IX][j+1] - grid[IX][j  ]) + \
            -2 * f_etas[i,   j] / (grid[IX][j+1] - grid[IX][j-1]) / (grid[IX][j+1] - grid[IX][j  ])

    # vy_j+½_i+1
    A[mat_row, gidx([i+1, j  ], nx, DIM) + IZ] =  4 * f_etan[i  , j  ] / (grid[IZ][i+1] - grid[IZ][i]) / (grid[IZ][i+1] - grid[IZ][i-1])

    # vy_j+½_i-1
    A[mat_row, gidx([i-1, j  ], nx, DIM) + IZ] =  4 * f_etan[i-1, j  ] / (grid[IZ][i] - grid[IZ][i-1]) / (grid[IZ][i+1] - grid[IZ][i-1]) 

    # vy_j+1+½_i
    A[mat_row, gidx([i  , j+1], nx, DIM) + IZ] =  2 * f_etas[i  , j+1] / (grid[IX][j+2] - grid[IX][j]) / (grid[IX][j+1] - grid[IX][j])

    # vy_j-½_i
    A[mat_row, gidx([i  , j-1], nx, DIM) + IZ] =  2 * f_etas[i  , j  ] / (grid[IX][j+1] - grid[IX][j-1]) / (grid[IX][j+1] - grid[IX][j])

    # vx_j+1_i+½
    A[mat_row, gidx([i  , j+1], nx, DIM) + IX] =  2 * f_etas[i  , j+1] / (grid[IZ][i+1] - grid[IZ][i-1]) / (grid[IX][j+1] - grid[IX][j])

    # vx_j+1_i-½
    A[mat_row, gidx([i-1, j+1], nx, DIM) + IX] = -2 * f_etas[i  , j+1] / (grid[IZ][i+1] - grid[IZ][i-1]) / (grid[IX][j+1] - grid[IX][j])

    # vx_j_i+½
    A[mat_row, gidx([i  , j  ], nx, DIM) + IX] = -2 * f_etas[i  , j  ] / (grid[IZ][i+1] - grid[IZ][i-1]) / (grid[IX][j+1] - grid[IX][j])

    # vx_j_i-½
    A[mat_row, gidx([i-1, j  ], nx, DIM) + IX] =  2 * f_etas[i  , j  ] / (grid[IZ][i+1] - grid[IZ][i-1]) / (grid[IX][j+1] - grid[IX][j])

    # P_j+½_i+½
    A[mat_row, gidx([i  , j  ], nx, DIM) + IP] = -2 * Kcont / (grid[IZ][i+1] - grid[IZ][i-1])

    # P_j+½_i-½
    A[mat_row, gidx([i-1, j  ], nx, DIM) + IP] =  2 * Kcont / (grid[IZ][i+1] - grid[IZ][i-1]) 

    if surfstab:
        if tstep is None:
            raise Exception("surface stabilization needs predetermined tstep")
        A[mat_row, gidx([i, j], nx, DIM) + IX] += surfstab_theta * tstep * G[IZ] * 0.5 * (f_rho[i, j+1] + f_rho[i+1, j+1] - f_rho[i, j-1] - f_rho[i+1, j-1]) / (grid[IX][j+1] - grid[IX][j-1]) 
        A[mat_row, gidx([i, j], nx, DIM) + IZ] += surfstab_theta * tstep * G[IZ] * 0.5 * (f_rho[i+1, j] + f_rho[i+1, j+1] - f_rho[i-1, j] - f_rho[i-1, j+1]) / (grid[IZ][i+1] - grid[IZ][i-1]) 

    lc[mat_row] += 1
    rhs[mat_row] = -0.5 * (f_rho[i, j] + f_rho[i, j+1]) * G[IZ] 



    # ::: x-stokes :::

    iset = np.arange(1, nx[IZ]-2)
    jset = np.arange(1, nx[IX]-1)
    ijset = np.meshgrid(iset, jset)
    i = ijset[IZ].flatten()
    j = ijset[IX].flatten()

    ieq = IX
    mat_row = gidx([i, j], nx, DIM) + ieq

    # vx_i+½_j
    A[mat_row, gidx([i  , j  ], nx, DIM) + IX] = \
    -4 * f_etan[i,   j] / (grid[IX][j+1] - grid[IX][j  ]) / (grid[IX][j+1] - grid[IX][j-1]) + \
    -4 * f_etan[i, j-1] / (grid[IX][j  ] - grid[IX][j-1]) / (grid[IX][j+1] - grid[IX][j-1]) + \
    -2 * f_etas[i+1, j] / (grid[IZ][i+2] - grid[IZ][i  ]) / (grid[IZ][i+1] - grid[IZ][i  ]) + \
    -2 * f_etas[i,   j] / (grid[IZ][i+1] - grid[IZ][i-1]) / (grid[IZ][i+1] - grid[IZ][i  ])  
    # coefficients were -4, -4, -2, -2

    #coefficients below were 4 4 2 -2 2 -2 2 -2 -2 2
    # vx_i+½_j+1
    A[mat_row, gidx([i  , j+1], nx, DIM) + IX] =  4 * f_etan[i  , j  ] / (grid[IX][j+1] - grid[IX][j]) / (grid[IX][j+1] - grid[IX][j-1])

    # vx_i+½_j-1
    A[mat_row, gidx([i  , j-1], nx, DIM) + IX] =  4 * f_etan[i  , j-1] / (grid[IX][j] - grid[IX][j-1]) / (grid[IX][j+1] - grid[IX][j-1])

    # vx_i+1+½_j
    A[mat_row, gidx([i+1, j  ], nx, DIM) + IX] =  2 * f_etas[i+1, j  ] / (grid[IZ][i+2] - grid[IZ][i]) / (grid[IZ][i+1] - grid[IZ][i])

    # vx_i-½_j  
    A[mat_row, gidx([i-1, j  ], nx, DIM) + IX] =  2 * f_etas[i  , j  ] / (grid[IZ][i+1] - grid[IZ][i-1]) / (grid[IZ][i+1] - grid[IZ][i])

    # vy_i+1_j+½
    A[mat_row, gidx([i+1, j  ], nx, DIM) + IZ] =  2 * f_etas[i+1, j  ] / (grid[IX][j+1] - grid[IX][j-1]) / (grid[IZ][i+1] - grid[IZ][i])

    # vy_i+1_j-½
    A[mat_row, gidx([i+1, j-1], nx, DIM) + IZ] = -2 * f_etas[i+1, j  ] / (grid[IX][j+1] - grid[IX][j-1]) / (grid[IZ][i+1] - grid[IZ][i])

    # vy_i_j+½        
    A[mat_row, gidx([i  , j  ], nx, DIM) + IZ] = -2 * f_etas[i  , j  ] / (grid[IX][j+1] - grid[IX][j-1]) / (grid[IZ][i+1] - grid[IZ][i])

    # vy_i_j-½  
    A[mat_row, gidx([i  , j-1], nx, DIM) + IZ] =  2 * f_etas[i  , j  ] / (grid[IX][j+1] - grid[IX][j-1]) / (grid[IZ][i+1] - grid[IZ][i])

    # P_i+½_j+½
    A[mat_row, gidx([i  , j  ], nx, DIM) + IP] = -2 * Kcont / (grid[IX][j+1] - grid[IX][j-1])

    # P_i+½_j-½
    A[mat_row, gidx([i  , j-1], nx, DIM) + IP] =  2 * Kcont / (grid[IX][j+1] - grid[IX][j-1])

    if surfstab:
        if tstep is None:
            raise Exception("surface stabilization needs predetermined tstep")
        A[mat_row, gidx([i, j], nx, DIM) + IX] += surfstab_theta * tstep * G[IX] * 0.5 * (f_rho[i, j+1] + f_rho[i+1, j+1] - f_rho[i, j-1] - f_rho[i+1, j-1]) / (grid[IX][j+1] - grid[IX][j-1]) 
        A[mat_row, gidx([i, j], nx, DIM) + IZ] += surfstab_theta * tstep * G[IX] * 0.5 * (f_rho[i+1, j] + f_rho[i+1, j+1] - f_rho[i-1, j] - f_rho[i-1, j+1]) / (grid[IZ][i+1] - grid[IZ][i-1]) 

    lc[mat_row] += 1
    rhs[mat_row] = -0.5 * (f_rho[i, j] + f_rho[i+1, j]) * G[IX] 



    # ::: continuity :::

    iset = np.arange(1, nx[IZ]-2)
    jset = np.arange(1, nx[IX]-2)
    ijset = np.meshgrid(iset, jset)
    i = ijset[IZ].flatten()
    j = ijset[IX].flatten()

    ieq = IP
    mat_row = gidx([i, j], nx, DIM) + ieq

    # vx_i-½_j
    A[mat_row, gidx([i  , j+1], nx, DIM) + IX] =  Kcont / (grid[IX][j+1] - grid[IX][j])

    # vx_i-½_j-1
    A[mat_row, gidx([i  , j  ], nx, DIM) + IX] = -Kcont / (grid[IX][j+1] - grid[IX][j])

    # vy_i_j-½ 
    A[mat_row, gidx([i+1, j  ], nx, DIM) + IZ] =  Kcont / (grid[IZ][i+1] - grid[IZ][i])

    # vy_i-1_j-½
    A[mat_row, gidx([i  , j  ], nx, DIM) + IZ] = -Kcont / (grid[IZ][i+1] - grid[IZ][i])

    lc[mat_row] += 1
    rhs[mat_row] = 0


    # one pressure point with absolute pressure value
    # define at in-/outflow boundary if one exists,
    # otherwise at i=3,j=2

    bc_alldirichlet = True
    flowbnd_wall = 0
    flowbnd_dir = 0
    for idir in range(DIM):
        for iwall in [0,1]:
            if bc[DIM*iwall + idir] & BC_TYPE_FLOWTHRU:
                bc_alldirichlet = False
                flowbnd_wall = iwall
                flowbnd_dir = idir

    if bc_alldirichlet:
        i = 3
        j = 2
    else:
        if flowbnd_wall == 0 and flowbnd_dir == IX:
            j = 0
            i = int(nx[IZ]/2)
        elif flowbnd_wall == 1 and flowbnd_dir == IX:
            j = nx[IX]-1
            i = int(nx[IZ]/2)
        else:
            raise Exception("flow bnd condition in IZ dir no implemented")

    if bc_alldirichlet:
        mat_row = gidx([i, j], nx, DIM) + IP
        #A[mat_row, :] = 0
        A[mat_row, gidx([i, j  ], nx, DIM) + IP] += Kcont
        lc[mat_row] = 1
        rhs[mat_row] += 0

        ## surf pres to zero
        #j = np.arange(0, nx[IX])
        #mat_row = gidx([0, j], nx, DIM) + IP
        #A[mat_row, gidx([0, j], nx, DIM) + IP] = Kcont
        #rhs[mat_row] = 0


  
    if DEBUG > 5:
        print("================")
        print(">1:", np.sum(lc > 1))
        if np.sum(lc > 1) > 0:
            print(np.where(lc>1))
        print("=0:", np.sum(lc==0))
        print("================")

    return (A, rhs)
