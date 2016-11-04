#!/usr/bin/python3

from pylamp_const import *
import pylamp_stokes 
import pylamp_trac
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
import importlib

importlib.reload(pylamp_stokes)
importlib.reload(pylamp_trac)

def pprint(*arg):
    comm = MPI.COMM_WORLD
    iproc = comm.Get_rank()
    nproc = comm.Get_size()
    
    if iproc == 0:
        print(''.join(map(str,arg)))
    
    return


#### MAIN ####

# Configurable options
nx    =   [100,101]         # use order z,x,y
L     =   [660e3, 1000e3]
tracdens = [16*nx[IZ], 16*nx[IX]] 

# Derived options
dx    =   [L[i]/(nx[i]-1) for i in range(DIM)]

# Form the grids
grid   =   [np.linspace(0, L[i], nx[i]) for i in range(DIM)] 
mesh   =   np.meshgrid(*grid, indexing='ij')
gridmp =   [(grid[i][1:nx[i]] + grid[i][0:(nx[i]-1)]) / 2 for i in range(DIM)]

for i in range(DIM):
    gridmp[i] = np.append(gridmp[i], gridmp[i][-1] + (gridmp[i][-1]-gridmp[i][-2]))

meshmp =   np.meshgrid(*gridmp, indexing='ij')

# Variable fields
f_vel  =   [np.zeros(nx) for i in range(DIM)]  # vx in y-midpoint field
                                               # vy in x-midpoint field
f_etas =   np.zeros(nx)    # viscosity in main grid points
f_T    =   np.zeros(nx)    # temperature in main grid points
f_rho  =   np.zeros(nx)    # rho in main grid points
f_P    =   np.zeros(nx)    # pressure in xy-midpoints
f_etan =   np.zeros(nx)    # viscosity in xy-midpoints


# Tracers
ntrac = np.prod(tracdens[0:DIM])

tr_x = np.random.rand(ntrac, DIM)  # tracer coordinates
tr_x = np.multiply(tr_x, L)
tr_f = np.zeros((ntrac, NFTRAC))     # tracer functions (values)


# Some material values and initial values
tr_f[:, TR_RHO] = 3300
idxx = (tr_x[:, IX] < 550e3) & (tr_x[:, IX] > 450e3)
idxz = (tr_x[:, IZ] < 380e3) & (tr_x[:, IZ] > 280e3)
tr_f[idxx & idxz, TR_RHO] = 3350
#f_rho[:,:] = 3300
#idx = np.ix_((grid[IZ] < 380e3) & (grid[IZ] > 280e3), (grid[IX] < 550e3) & (grid[IX] > 450e3))
#f_rho[idx] = 3350

tr_f[:, TR_ETA] = 1e19
tr_f[idxx & idxz, TR_ETA] = 1e17
#f_etas[:,:] = 1e19
#f_etan[:,:] = 1e19

plt.ion()
plt.close('all')
fig = plt.figure()

it = 0
while (True):
    it += 1
    print("Properties trac2grid")
    pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_RHO, TR_ETA]], mesh, grid, [f_rho, f_etas], nx, 
            avgscheme=[pylamp_trac.INTERP_AVG_ARITHMETIC, pylamp_trac.INTERP_AVG_GEOMETRIC])
    pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_ETA]], meshmp, gridmp, [f_etan], nx, avgscheme=[pylamp_trac.INTERP_AVG_GEOMETRIC])

    print("Build stokes")
    (A, rhs) = pylamp_stokes.makeStokesMatrix(nx, grid, f_etas, f_etan, f_rho)

    print("Solve stokes")
    # Solve it!
    #x = scipy.sparse.linalg.bicgstab(scipy.sparse.csc_matrix(A), rhs)[0]
    x = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), rhs)

    (newvel, newpres) = pylamp_stokes.x2vp(x, nx)

    print("Tracer advection")
    trac_vel, tracs_new = pylamp_trac.RK(tr_x, grid, newvel, nx, 10*SECINKYR, order=2)
    tr_x[:,:] = tracs_new[:,:]

    if it % 10 == 1:
        print("Plot")
        fig.clf()
        ax = fig.add_subplot(221)
        ax.pcolormesh(newvel[0])
        ax = fig.add_subplot(222)
        ax.pcolormesh(newvel[1])
        ax = fig.add_subplot(223)
        #ax.pcolormesh(f_rho)
        ax.quiver(tr_x[::10,IX], tr_x[::10,IZ], trac_vel[::10,IX], trac_vel[::10,IZ])
        ax = fig.add_subplot(224)
        ax.pcolormesh(f_etan)
        plt.show()

        dummy = input()

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
# NB! Here indices should have all +1, i.e. pressure ghost points are on the right
# and below instead of left and above as in Gerya's book
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
