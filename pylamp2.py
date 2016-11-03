#!/usr/bin/python3

#import math
#import time
import numpy as np
import matplotlib.pyplot as plt
import sys
#from mpl_toolkits.mplot3d import axes3d
import scipy.sparse.linalg
#import scipy.sparse.linalg as linalg
#import itertools
#from mpi4py import MPI



def pprint(*arg):
    comm = MPI.COMM_WORLD
    iproc = comm.Get_rank()
    nproc = comm.Get_size()
    
    if iproc == 0:
        print(''.join(map(str,arg)))
    
    return

def gidx(idxs, nx, dim):
    # Global index for the matrix in linear system of equations
    #   gidx = ...  +  iz * ny * nx * (DIM+1)  +  ix * ny * (DIM+1)  +  iy * (DIM+1)  +  ieq
    if len(idxs) != dim:
        raise Exception("num of idxs != dimensions")
    return np.sum([idxs[i] * np.prod(nx[(i+1):dim]) * (dim+1) for i in range(dim)],dtype=np.int)

def numOfZeroRows(a, c):
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


#### MAIN ####

# Configurable options
nx    =   [33,50] #[6,7]         # use order z,x,y
L     =   [660e3, 1000e3]


# Constants
DIM   =   2                # designed values: 2 or 3, implemented: 2 (partly)
G     =   [9.81, 0]
IZ    =   0
IX    =   1
IY    =   2
IP    =   DIM

# Derived options
dx    =   [L[i]/(nx[i]-1) for i in range(DIM)]
dof   =   np.prod(nx) * (DIM + 1)

# Form the grids
grid   =   [np.linspace(0, L[i], nx[i]) for i in range(DIM)] 
mesh   =   np.meshgrid(*grid)
gridmp =   [(grid[i][1:nx[i]] + grid[i][0:(nx[i]-1)]) / 2 for i in range(DIM)]
meshmp =   np.meshgrid(*gridmp)

# Variable fields
f_vel  =   [np.zeros(nx) for i in range(DIM)]  # vx in y-midpoint field
                                               # vy in x-midpoint field
f_etas =   np.zeros(nx)    # viscosity in main grid points
f_T    =   np.zeros(nx)    # temperature in main grid points
f_rho  =   np.zeros(nx)    # rho in main grid points
f_P    =   np.zeros(nx)    # pressure in xy-midpoints
f_etan =   np.zeros(nx)    # viscosity in xy-midpoints


# Some material values and initial values
f_rho[:,:] = 3300
idx = np.ix_(grid[IZ] < 330e3, (grid[IX] < 700e3) & (grid[IX] > 300e3))
f_rho[idx] = 3500

f_etas[:,:] = 1e19
f_etan[:,:] = 1e19

# Some scaling coefs for matrix inversion
visc0       = 1e19
Kcont = 2*visc0 / np.sum(dx)
Kbond = 4*visc0 / np.sum(dx)**2

# Form the solution matrix
A   = np.zeros((dof,dof)) #scipy.sparse.lil_matrix((dof, dof))
rhs = np.zeros(dof)

c = 0
numOfZeroRows(A,c)

# ghost points:
for i in range(nx[IZ]):
    j = nx[IX]-1

    # force vy and P to zero
    A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = Kcont
    rhs[gidx([i, j], nx, DIM) + IZ] = 0
    c += 1

    A[gidx([i, j], nx, DIM) + IP, gidx([i, j], nx, DIM) + IP] = Kcont
    rhs[gidx([i, j], nx, DIM) + IP] = 0
    c += 1

numOfZeroRows(A,c)

for j in range(nx[IX]):
    i = nx[IZ]-1

    # force vx and P to zero
    A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = Kcont
    rhs[gidx([i, j], nx, DIM) + IX] = 0
    c += 1

    if j < nx[IX]-1:
        A[gidx([i, j], nx, DIM) + IP, gidx([i, j], nx, DIM) + IP] = Kcont
        rhs[gidx([i, j], nx, DIM) + IP] = 0
        c += 1

numOfZeroRows(A,c)

### boundaries

# z = 0
i = 0
for j in range(0, nx[IX]):

    # vx extrapolated to be zero from two internal nodes
    if j > 0 and j < nx[IX]-1:
        dx1 = grid[IZ][i+1] - grid[IZ][i  ]
        dx2 = grid[IZ][i+2] - grid[IZ][i+1]
        A[gidx([i, j], nx, DIM) + IX, gidx([i  , j], nx, DIM) + IX] = 1 + dx1 / (dx1+dx2)
        A[gidx([i, j], nx, DIM) + IX, gidx([i+1, j], nx, DIM) + IX] = dx1 / (dx1+dx2)
        rhs[gidx([i, j], nx, DIM) + IX] = 0
        c += 1

    # vz = 0
    if j < nx[IX]-1:
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = 1
        rhs[gidx([i, j], nx, DIM) + IZ] = 0
        c += 1

numOfZeroRows(A,c)

# z = Lz
i = nx[IZ]-1
for j in range(0, nx[IX]):

    # vx extrapolated to be zero from two internal nodes
    if j > 0 and j < nx[IX]-1:
        dx1 = grid[IZ][i] - grid[IZ][i-1]
        dx2 = grid[IZ][i-1] - grid[IZ][i-2]
        A[gidx([i-1, j], nx, DIM) + IX, gidx([i-1, j], nx, DIM) + IX] = 1 + dx1 / (dx1+dx2)
        A[gidx([i-1, j], nx, DIM) + IX, gidx([i-2, j], nx, DIM) + IX] = dx1 / (dx1+dx2)
        rhs[gidx([i-1, j], nx, DIM) + IX] = 0
        c += 1

    # vz = 0
    if j < nx[IX]-1:
        A[gidx([i, j], nx, DIM) + IZ, gidx([i, j], nx, DIM) + IZ] = 1
        rhs[gidx([i, j], nx, DIM) + IZ] = 0
        c += 1
    
numOfZeroRows(A,c)

# x = 0
j = 0
for i in range(0, nx[IZ]):

    # vz extrapolated to be zero from two internal nodes
    if i > 0 and i < nx[IZ]-1:
        dx1 = grid[IX][j+1] - grid[IX][j  ]
        dx2 = grid[IX][j+2] - grid[IX][j+1]
        A[gidx([i, j], nx, DIM) + IZ, gidx([i  , j], nx, DIM) + IZ] = 1 + dx1 / (dx1+dx2)
        A[gidx([i, j], nx, DIM) + IZ, gidx([i+1, j], nx, DIM) + IZ] = dx1 / (dx1+dx2)
        rhs[gidx([i, j], nx, DIM) + IZ] = 0
        c += 1

    # vx = 0
    if i < nx[IZ]-1:
        A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = 1
        rhs[gidx([i, j], nx, DIM) + IX] = 0
        c += 1

numOfZeroRows(A,c)

# x = Lx
j = nx[IX]-1
for i in range(0, nx[IZ]):

    # vz extrapolated to be zero from two internal nodes
    if i > 0 and i < nx[IZ]-1:
        dx1 = grid[IX][j] - grid[IX][j-1]
        dx2 = grid[IX][j-1] - grid[IX][j-2]
        A[gidx([i, j-1], nx, DIM) + IZ, gidx([i, j-1], nx, DIM) + IZ] = 1 + dx1 / (dx1+dx2)
        A[gidx([i, j-1], nx, DIM) + IZ, gidx([i, j-2], nx, DIM) + IZ] = dx1 / (dx1+dx2)
        rhs[gidx([i, j], nx, DIM) + IZ] = 0
        c += 1

    # vx = 0
    if i < nx[IZ]-1:
        A[gidx([i, j], nx, DIM) + IX, gidx([i, j], nx, DIM) + IX] = 1
        rhs[gidx([i, j], nx, DIM) + IX] = 0
        c += 1

numOfZeroRows(A,c)


# continuity at the boundaries
for i in range(nx[IZ]-1):
    for j in range(nx[IX]-1):
        if i > 0 and i < nx[IZ]-2 and j > 0 and j < nx[IX]-2:
            continue
        if  (i == 0 and j == 0) or \
            (i == 0 and j == nx[IX]-2) or \
            (i == nx[IZ]-2 and j == 0) or \
            (i == nx[IZ]-2 and j == nx[IX]-2):
            continue

        # continuity
        mat_row = gidx([i, j], nx, DIM) + IP
        A[mat_row, gidx([i, j+1], nx, DIM) + IX] =  Kcont / (grid[IX][j+1] - grid[IX][j])
        A[mat_row, gidx([i, j], nx, DIM) + IX] = -Kcont / (grid[IX][j+1] - grid[IX][j])
        A[mat_row, gidx([i+1, j], nx, DIM) + IZ] =  Kcont / (grid[IZ][i+1] - grid[IZ][i])
        A[mat_row, gidx([i, j], nx, DIM) + IZ] = -Kcont / (grid[IZ][i+1] - grid[IZ][i])
        rhs[mat_row] = 0
        c += 1

numOfZeroRows(A, c)


# corners, horizontal symmetry for pressure
for i in [0, nx[IZ]-2]:
    j = 0
    A[gidx([i, j], nx, DIM) + IP, gidx([i, j+1], nx, DIM) + IP] =  Kbond
    A[gidx([i, j], nx, DIM) + IP, gidx([i, j  ], nx, DIM) + IP] = -Kbond
    rhs[gidx([i, j], nx, DIM) + IP] = 0
    c += 1

    j = nx[IX]-2
    A[gidx([i, j], nx, DIM) + IP, gidx([i, j-1], nx, DIM) + IP] =  Kbond
    A[gidx([i, j], nx, DIM) + IP, gidx([i, j  ], nx, DIM) + IP] = -Kbond
    rhs[gidx([i, j], nx, DIM) + IP] = 0
    c += 1


numOfZeroRows(A,c)            


# rest of the points
for i in range(1,nx[IZ]-1):
    for j in range(1,nx[IX]-1):

        # ::: z-stokes :::
        if j < nx[IX]-2:
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
        if i < nx[IZ]-2:
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
        if j < nx[IX]-2 and i < nx[IZ]-2:
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

            if i == 2 and j == 3:
                A[mat_row, gidx([i, j  ], nx, DIM) + IP] += Kcont
                rhs[mat_row] += 0




# Solve it!
#x = scipy.sparse.linalg.bicgstab(scipy.sparse.csc_matrix(A), rhs)[0]
x = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), rhs)

for i in range(nx[IZ]):
    for j in range(nx[IX]):
        f_vel[IZ][i,j] = x[gidx([i, j], nx, DIM) + IZ]
        f_vel[IX][i,j] = x[gidx([i, j], nx, DIM) + IX]
        f_P[i, j] = x[gidx([i, j], nx, DIM) + IP]


plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(221)
ax.pcolormesh(f_vel[IZ])
ax = fig.add_subplot(222)
ax.pcolormesh(f_vel[IX])
ax = fig.add_subplot(223)
ax.pcolormesh(f_P)
plt.show()

sys.exit()

# :::: x-stokes ::::
# For vx-node vx_i+½_j
#
# Sxy_i_j = 2*etas_i_j  *  ([vx_i+½_j - vx_i-½_j] / [y_i+1 - y_i-1]] + [vy_i_j+½ - vy_i_j-½] / [x_j+1 - x_j-1]) 
# Sxy_i+1_j = 2*etas_i+1_j  *  ([vx_i+1+½_j - vx_i+½_j] / [y_i+2 - y_i]] + [vy_i+1_j+½ - vy_i+1_j-½] / [x_j+1 - x_j-1]) 
#
# Sxx_i+½_j-½ = 2*etan_i+½_j-½ * (vx_i+½_j - vx_i+½_j-1) / (x_j - x_j-1)
# Sxx_i+½_j+½ = 2*etan_i+½_j+½ * (vx_i+½_j+1 - vx_i+½_j) / (x_j+1 - x_j)
#
# => components:
#
# Sxy_i_j     = vx_i+½_j * 2*etas_i_j / (y_i+1 - y_i-1)    +    vx_i-½_j * (-1) * 2*etas_i_j / (y_i+1 - y_i-1)     +
#               vy_i_j+½ * 2*etas_i_j / (x_j+1 - x_j-1)    +    vy_i_j-½ * (-1) * 2*etas_i_j / (x_j+1 - x_j-1)
# Sxy_i+1_j   = vx_i+1+½_j * 2*etas_i+1_j / (y_i+2 - y_i)  +    vx_i+½_j * (-1) * 2*etas_i+1_j / (y_i+2 - y_i)     +
#               vy_i+1_j+½ * 2*etas_i+1_j / (x_j+1 - x_j-1)+    vy_i+1_j-½ * (-1) * 2*etas_i+1_j / (x_j+1 - x_j-1)
# Sxx_i+½_j-½ = vx_i+½_j   * 2*etan_i+½_j-½ / (x_j - x_j-1)  +    vx_i+½_j-1 * (-1) * 2*etan_i+½_j-½ / (x_j - x_j-1)
# Sxx_i+½_j+½ = vx_i+½_j+1 * 2*etan_i+½_j+½ / (x_j+1 - x_j)  +    vx_i+½_j * (-1) * 2*etan_i+½_j+½ / (x_j+1 - x_j) 
#
# Stokes eq:

# 2 * (Sxx_i+½_j+½ - Sxx_i+½_j-½) / (x_j+1 - x_j-1)  +  (Sxy_i+1_j - Sxy_i_j) / (y_i+1 - y_i)  +  2 * (-P_i+½_j+½ + P_i+½_j-½) / (x_j+1 - x_j-1)
# = - ½ * (rho_i_j + rho_i+1_j) * g_x
#
# =>
#    2 * (Sxx_i+½_j+½ - Sxx_i+½_j-½) / (x_j+1 - x_j-1) 
# +  (Sxy_i+1_j - Sxy_i_j) / (y_i+1 - y_i)
# +  -2 * (P_i+½_j+½ - P_i+½_j-½) / (x_j+1 - x_j-1)
# =  - ½ * (rho_i_j + rho_i+1_j) * g_x
#
# => 
#    2 * 
#        (
#          vx_i+½_j+1 * 2*etan_i+½_j+½ / (x_j+1 - x_j)  +    vx_i+½_j * (-1) * 2*etan_i+½_j+½ / (x_j+1 - x_j)   +
#         -(vx_i+½_j   * 2*etan_i+½_j-½ / (x_j - x_j-1)  +    vx_i+½_j-1 * (-1) * 2*etan_i+½_j-½ / (x_j - x_j-1))
#        ) / (x_j+1 - x_j-1)
#    +
#        (
#          vx_i+1+½_j * 2*etas_i+1_j / (y_i+2 - y_i)  +    vx_i+½_j * (-1) * 2*etas_i+1_j / (y_i+2 - y_i)     +
#          vy_i+1_j+½ * 2*etas_i+1_j / (x_j+1 - x_j-1)+    vy_i+1_j-½ * (-1) * 2*etas_i+1_j / (x_j+1 - x_j-1) + 
#         -(vx_i+½_j * 2*etas_i_j / (y_i+1 - y_i-1)    +    vx_i-½_j * (-1) * 2*etas_i_j / (y_i+1 - y_i-1)     +
#           vy_i_j+½ * 2*etas_i_j / (x_j+1 - x_j-1)    +    vy_i_j-½ * (-1) * 2*etas_i_j / (x_j+1 - x_j-1)
#          )
#        ) / (y_i+1 - y_i)
#    +
#     -2 * (P_i+½_j+½ - P_i+½_j-½) / (x_j+1 - x_j-1)
# = - ½ * (rho_i_j + rho_i+1_j) * g_x
#
# components:
# 
#   vx_i+½_j+1                          
#   2*2*etan_i+½_j+½ / (x_j+1 - x_j) / (x_j+1 - x_j-1)
#
#   vx_i+½_j
#   2*(-1) * 2*etan_i+½_j+½ / (x_j+1 - x_j) / (x_j+1 - x_j-1) + 
#   2*(-1)*2*etan_i+½_j-½ / (x_j - x_j-1)   / (x_j+1 - x_j-1) + 
#   (-1) * 2*etas_i+1_j / (y_i+2 - y_i)     / (y_i+1 - y_i)   +
#   (-1) * 2*etas_i_j / (y_i+1 - y_i-1)     / (y_i+1 - y_i)
#
#   vx_i+½_j-1
#   2*(-1)*(-1) * 2*etan_i+½_j-½ / (x_j - x_j-1) / (x_j+1 - x_j-1)
#
#   vx_i+1+½_j
#   2*etas_i+1_j / (y_i+2 - y_i) / (y_i+1 - y_i)
#
#   vy_i+1_j+½
#   2*etas_i+1_j / (x_j+1 - x_j-1) / (y_i+1 - y_i)
#
#   vy_i+1_j-½
#   (-1) * 2*etas_i+1_j / (x_j+1 - x_j-1) / (y_i+1 - y_i)
#
#   vx_i-½_j
#   (-1) * 2*etas_i_j / (y_i+1 - y_i-1) / (y_i+1 - y_i)
#
#   vy_i_j+½ 
#   2*etas_i_j / (x_j+1 - x_j-1) / (y_i+1 - y_i)
#
#   vy_i_j-½
#   (-1) * 2*etas_i_j / (x_j+1 - x_j-1) / (y_i+1 - y_i)
#
#   P_i+½_j+½
#   -2 / (x_j+1 - x_j-1)
# 
#   P_i+½_j-½
#   2 / (x_j+1 - x_j-1)
#
#
# :::: y-stokes ::::
# For vy-node vy_i_j+½
# 
# take components from x-stokes, swap all i<->j, x<->y
#
#   vy_j+½_i+1                                                                  
#   2*2*etan_j+½_i+½ / (y_i+1 - y_i) / (y_i+1 - y_i-1)                          
#                                                                               
#   vy_j+½_i                                                                    
#   2*(-1) * 2*etan_j+½_i+½ / (y_i+1 - y_i) / (y_i+1 - y_i-1) +                 
#   2*(-1)*2*etan_j+½_i-½ / (y_i - y_i-1)   / (y_i+1 - y_i-1) +                    
#   (-1) * 2*etas_j+1_i / (x_j+2 - x_j)     / (x_j+1 - x_j)   +                 
#   (-1) * 2*etas_j_i / (x_j+1 - x_j-1)     / (x_j+1 - x_j)                     
#                                                                               
#   vy_j+½_i-1                                                                  
#   2*(-1)*(-1) * 2*etan_j+½_i-½ / (y_i - y_i-1) / (y_i+1 - y_i-1)              
#                                                                               
#   vy_j+1+½_i                                                                  
#   2*etas_j+1_i / (x_j+2 - x_j) / (x_j+1 - x_j)                                
#                                                                               
#   vx_j+1_i+½                                                                  
#   2*etas_j+1_i / (y_i+1 - y_i-1) / (x_j+1 - x_j)                              
#                                                                               
#   vx_j+1_i-½                                                                  
#   (-1) * 2*etas_j+1_i / (y_i+1 - y_i-1) / (x_j+1 - x_j)                       
#                                                                               
#   vy_j-½_i                                                                    
#   (-1) * 2*etas_j_i / (x_j+1 - x_j-1) / (x_j+1 - x_j)                         
#                                                                               
#   vx_j_i+½                                                                    
#   2*etas_j_i / (y_i+1 - y_i-1) / (x_j+1 - x_j)                                
#                                                                               
#   vx_j_i-½                                                                    
#   (-1) * 2*etas_j_i / (y_i+1 - y_i-1) / (x_j+1 - x_j)                         
#                                                                               
#   P_j+½_i+½                                                                   
#   -2 / (y_i+1 - y_i-1)                                                        
#                                                                               
#   P_j+½_i-½                                                                   
#   2 / (y_i+1 - y_i-1)   
#
#
# :::: continuity ::::
# For P-node P_i-½_j+½
#
# (vx_i-½_j - vx_i-½_j-1) / (x_j - x_j-1)   +   (vy_i_j-½ - vy_i-1_j-½) / (y_i - y_i-1)  = 0
#
# Components:
#
# vx_i-½_j
# 1 / (x_j - x_j-1)
#
# vx_i-½_j-1
# -1 / (x_j - x_j-1)
# 
# vy_i_j-½
# 1 / (y_i - y_i-1)
#
# vy_i-1_j-½
#
