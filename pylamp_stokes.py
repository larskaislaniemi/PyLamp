#!/usr/bin/python3

from pylamp_const import *
import numpy as np
import sys

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

    for d in range(DIM):
        newvel[d] = np.delete(newvel[d], nx[d]-1, axis=d)
        newpres = np.delete(newpres, nx[d]-1, axis=d)

    return (newvel, newpres)


def makeStokesMatrix(nx, grid, f_etas, f_etan, f_rho):
    # Form the solution matrix for stokes/cont solving
    #
    # Currently can do only 2D
    #

    dof = np.prod(nx) * (DIM + 1)
    A   = np.zeros((dof,dof)) #scipy.sparse.lil_matrix((dof, dof))
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

    A[gidx([i, j], nx, DIM) + IP, gidx([i, j], nx, DIM) + IP] = Kcont
    rhs[gidx([i, j], nx, DIM) + IP] = 0

    j = np.arange(nx[IX])
    i = nx[IZ]-1


    # force vx and P to zero
    A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = Kcont
    rhs[gidx([i, j], nx, DIM) + IX] = 0

    j = np.arange(nx[IX]-1)
    A[gidx([i, j], nx, DIM) + IP, gidx([i, j], nx, DIM) + IP] = Kcont
    rhs[gidx([i, j], nx, DIM) + IP] = 0


    
    #### boundaries: ####

    # at z = 0
    i = 0

    # vx extrapolated to be zero from two internal nodes
    j = np.arange(1, nx[IX]-1)

    dx1 = grid[IZ][i+1] - grid[IZ][i  ]
    dx2 = grid[IZ][i+2] - grid[IZ][i+1]
    A[gidx([i, j], nx, DIM) + IX, gidx([i  , j], nx, DIM) + IX] = Kcont * (1 + dx1 / (dx1+dx2))
    A[gidx([i, j], nx, DIM) + IX, gidx([i+1, j], nx, DIM) + IX] = Kcont * dx1 / (dx1+dx2)
    rhs[gidx([i, j], nx, DIM) + IX] = 0

    # vz = 0
    j = np.arange(0, nx[IX]-1)
    A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont
    rhs[gidx([i, j], nx, DIM) + IZ] = 0


    # at z = Lz
    i = nx[IZ]-1

    # vx extrapolated to be zero from two internal nodes
    j = np.arange(1, nx[IX]-1)
    dx1 = grid[IZ][i] - grid[IZ][i-1]
    dx2 = grid[IZ][i-1] - grid[IZ][i-2]
    A[gidx([i-1, j], nx, DIM) + IX, gidx([i-1, j], nx, DIM) + IX] = Kcont * (1 + dx1 / (dx1+dx2))
    A[gidx([i-1, j], nx, DIM) + IX, gidx([i-2, j], nx, DIM) + IX] = Kcont * dx1 / (dx1+dx2)
    rhs[gidx([i-1, j], nx, DIM) + IX] = 0

    # vz = 0
    j = np.arange(0, nx[IX]-1)
    A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont
    rhs[gidx([i, j], nx, DIM) + IZ] = 0
        

    # at x = 0
    j = 0

    # vz extrapolated to be zero from two internal nodes
    i = np.arange(1, nx[IZ]-1)
    dx1 = grid[IX][j+1] - grid[IX][j  ]
    dx2 = grid[IX][j+2] - grid[IX][j+1]
    A[gidx([i, j], nx, DIM) + IZ, gidx([i  , j], nx, DIM) + IZ] = Kcont * (1 + dx1 / (dx1+dx2))
    A[gidx([i, j], nx, DIM) + IZ, gidx([i+1, j], nx, DIM) + IZ] = Kcont * dx1 / (dx1+dx2)
    rhs[gidx([i, j], nx, DIM) + IZ] = 0

    # vx = 0
    i = np.arange(0, nx[IZ]-1)
    A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = Kcont
    rhs[gidx([i, j], nx, DIM) + IX] = 0


    # at x = Lx
    j = nx[IX]-1

    # vz extrapolated to be zero from two internal nodes
    i = np.arange(1, nx[IZ]-1)
    dx1 = grid[IX][j] - grid[IX][j-1]
    dx2 = grid[IX][j-1] - grid[IX][j-2]
    A[gidx([i, j-1], nx, DIM) + IZ, gidx([i, j-1], nx, DIM) + IZ] = Kcont * (1 + dx1 / (dx1+dx2))
    A[gidx([i, j-1], nx, DIM) + IZ, gidx([i, j-2], nx, DIM) + IZ] = Kcont * dx1 / (dx1+dx2)
    rhs[gidx([i, j], nx, DIM) + IZ] = 0

    # vx = 0
    i = np.arange(0, nx[IZ]-1)
    A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = Kcont
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
    rhs[mat_row] = 0


    ### corners, horizontal symmetry for pressure
    for i in [0, nx[IZ]-2]:
        j = 0
        A[gidx([i, j], nx, DIM) + IP, gidx([i, j+1], nx, DIM) + IP] =  Kbond
        A[gidx([i, j], nx, DIM) + IP, gidx([i, j  ], nx, DIM) + IP] = -Kbond
        rhs[gidx([i, j], nx, DIM) + IP] = 0

        j = nx[IX]-2
        A[gidx([i, j], nx, DIM) + IP, gidx([i, j-1], nx, DIM) + IP] =  Kbond
        A[gidx([i, j], nx, DIM) + IP, gidx([i, j  ], nx, DIM) + IP] = -Kbond
        rhs[gidx([i, j], nx, DIM) + IP] = 0



    # rest of the points

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
    A[mat_row, gidx([i  , j-1], nx, DIM) + IZ] = -2 * f_etas[i  , j  ] / (grid[IX][j+1] - grid[IX][j-1]) / (grid[IX][j+1] - grid[IX][j])

    # vx_j+1_i+½
    A[mat_row, gidx([i  , j+1], nx, DIM) + IX] =  2 * f_etas[i  , j+1] / (grid[IZ][i+1] - grid[IZ][i-1]) / (grid[IX][j+1] - grid[IX][j])

    # vx_j+1_i-½
    A[mat_row, gidx([i-1, j+1], nx, DIM) + IX] = -2 * f_etas[i  , j+1] / (grid[IZ][i+1] - grid[IZ][i-1]) / (grid[IX][j+1] - grid[IX][j])
            
    # vx_j_i+½
    A[mat_row, gidx([i  , j  ], nx, DIM) + IX] =  2 * f_etas[i  , j  ] / (grid[IZ][i+1] - grid[IZ][i-1]) / (grid[IX][j+1] - grid[IX][j])

    # vx_j_i-½
    A[mat_row, gidx([i-1, j  ], nx, DIM) + IX] = -2 * f_etas[i  , j  ] / (grid[IZ][i+1] - grid[IZ][i-1]) / (grid[IX][j+1] - grid[IX][j])

    # P_j+½_i+½
    A[mat_row, gidx([i  , j  ], nx, DIM) + IP] = -2 * Kcont / (grid[IZ][i+1] - grid[IZ][i-1])

    # P_j+½_i-½
    A[mat_row, gidx([i-1, j  ], nx, DIM) + IP] =  2 * Kcont / (grid[IZ][i+1] - grid[IZ][i-1]) 

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

    # vx_i+½_j+1
    A[mat_row, gidx([i  , j+1], nx, DIM) + IX] =  4 * f_etan[i  , j  ] / (grid[IX][j+1] - grid[IX][j]) / (grid[IX][j+1] - grid[IX][j-1])

    # vx_i+½_j-1
    A[mat_row, gidx([i  , j-1], nx, DIM) + IX] =  4 * f_etan[i  , j-1] / (grid[IX][j] - grid[IX][j-1]) / (grid[IX][j+1] - grid[IX][j-1])

    # vx_i+1+½_j
    A[mat_row, gidx([i+1, j  ], nx, DIM) + IX] =  2 * f_etas[i+1, j  ] / (grid[IZ][i+2] - grid[IZ][i]) / (grid[IZ][i+1] - grid[IZ][i])

    # vx_i-½_j  
    A[mat_row, gidx([i-1, j  ], nx, DIM) + IX] = -2 * f_etas[i  , j  ] / (grid[IZ][i+1] - grid[IZ][i-1]) / (grid[IZ][i+1] - grid[IZ][i])

    # vy_i+1_j+½
    A[mat_row, gidx([i+1, j  ], nx, DIM) + IZ] =  2 * f_etas[i+1, j  ] / (grid[IX][j+1] - grid[IX][j-1]) / (grid[IZ][i+1] - grid[IZ][i])

    # vy_i+1_j-½
    A[mat_row, gidx([i+1, j-1], nx, DIM) + IZ] = -2 * f_etas[i+1, j  ] / (grid[IX][j+1] - grid[IX][j-1]) / (grid[IZ][i+1] - grid[IZ][i])

    # vy_i_j+½        
    A[mat_row, gidx([i  , j  ], nx, DIM) + IZ] =  2 * f_etas[i  , j  ] / (grid[IX][j+1] - grid[IX][j-1]) / (grid[IZ][i+1] - grid[IZ][i])

    # vy_i_j-½  
    A[mat_row, gidx([i  , j-1], nx, DIM) + IZ] = -2 * f_etas[i  , j  ] / (grid[IX][j+1] - grid[IX][j-1]) / (grid[IZ][i+1] - grid[IZ][i])

    # P_i+½_j+½
    A[mat_row, gidx([i  , j  ], nx, DIM) + IP] = -2 * Kcont / (grid[IX][j+1] - grid[IX][j-1])

    # P_i+½_j-½
    A[mat_row, gidx([i  , j-1], nx, DIM) + IP] =  2 * Kcont / (grid[IX][j+1] - grid[IX][j-1])

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

    rhs[mat_row] = 0

    i == 3
    j == 4
    A[gidx([i, j], nx, DIM) + IP, gidx([i, j], nx, DIM) + IP] += Kcont
    rhs[mat_row] += 0



    return (A, rhs)
