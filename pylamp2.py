#!/usr/bin/python3

from pylamp_const import *
import pylamp_stokes 
import pylamp_trac
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.sparse.linalg
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
if __name__ == "__main__":

    # Configurable options
    nx    =   [64,65]         # use order z,x,y
    L     =   [660e3, 1000e3]
    tracdens = 20   # how many tracers per element

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
    f_vel  =   [np.zeros(nx) for i in range(DIM)]  # vx in z-midpoint field
                                                   # vz in x-midpoint field
    f_etas =   np.zeros(nx)    # viscosity in main grid points
    f_T    =   np.zeros(nx)    # temperature in main grid points
    f_rho  =   np.zeros(nx)    # rho in main grid points
    f_Cp   =   np.zeros(nx)    # Cp in main grid points
    f_P    =   np.zeros(nx)    # pressure in xy-midpoints
    f_etan =   np.zeros(nx)    # viscosity in xy-midpoints
    f_k    =   [np.zeros(nx) for i in range(DIM)]  # kz in z-midpoint field
                                                   # kx in x-midpoint field

    # Tracers
    ntrac = np.prod(nx)*tracdens

    tr_x = np.random.rand(ntrac, DIM)  # tracer coordinates
    tr_x = np.multiply(tr_x, L)
    tr_f = np.zeros((ntrac, NFTRAC))     # tracer functions (values)


    ## Some material values and initial values
    # Density
    tr_f[:, TR_RHO] = 3300
    idxx = (tr_x[:, IX] < 550e3) & (tr_x[:, IX] > 450e3)
    idxz = (tr_x[:, IZ] < 380e3) & (tr_x[:, IZ] > 280e3)
    tr_f[idxx & idxz, TR_RHO] = 3340

    ## ... sticky air test
    #idxz2 = tr_x[:, IZ] < 50e3
    #tr_f[idxz2, TR_RHO] = 1000
    #tr_f[idxz2, TR_ETA] = 1e17

    # Viscosity
    tr_f[:, TR_ETA] = 1e19
    tr_f[idxx & idxz, TR_ETA] = 1e19

    # Passive markers
    inixdiv = np.linspace(0, L[IX], 10)
    inizdiv = np.linspace(0, L[IZ], 10)
    for i in range(0,9,2):
        tr_f[(tr_x[:,IZ] >= inizdiv[i]) & (tr_x[:,IZ] < inizdiv[i+1]), TR_MRK] += 1
    for i in range(1,9,2):
        tr_f[(tr_x[:,IZ] >= inizdiv[i]) & (tr_x[:,IZ] < inizdiv[i+1]), TR_MRK] += 2
    for i in range(0,9,2):
        tr_f[(tr_x[:,IX] >= inixdiv[i]) & (tr_x[:,IX] < inixdiv[i+1]), TR_MRK] *= -1


    it = 0
    totaltime = 0
    while (True):
        it += 1
        print("\n --- Time step:", it, "---")

        print("Properties trac2grid")
        pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_RHO, TR_ETA]], mesh, grid, [f_rho, f_etas], nx, 
                avgscheme=[pylamp_trac.INTERP_AVG_ARITHMETIC, pylamp_trac.INTERP_AVG_GEOMETRIC])
        pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_ETA]], meshmp, gridmp, [f_etan], nx, avgscheme=[pylamp_trac.INTERP_AVG_GEOMETRIC])

        print("Build stokes")
        bcstokes = [ \
                pylamp_stokes.BC_TYPE_FREESLIP, \
                pylamp_stokes.BC_TYPE_NOSLIP, \
                pylamp_stokes.BC_TYPE_NOSLIP, \
                pylamp_stokes.BC_TYPE_NOSLIP  \
                ]  # idx is DIM*side + DIR 
        (A, rhs) = pylamp_stokes.makeStokesMatrix(nx, grid, f_etas, f_etan, f_rho, bcstokes)

        print("Solve stokes")
        # Solve it!
        #x = scipy.sparse.linalg.bicgstab(scipy.sparse.csc_matrix(A), rhs)[0]
        x = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), rhs)

        (newvel, newpres) = pylamp_stokes.x2vp(x, nx)

        tstep = 0.25 * np.min(dx) / np.max(newvel)
        totaltime += tstep
        print("Tracer advection")
        print("   time step =", tstep/SECINKYR, "kyrs")
        print("   time now  =", totaltime/SECINKYR, "kyrs")

        # for interpolation of velocity from grid to tracers we need
        # all tracers to be within the grid and so we need to extend the 
        # (vz,vx) grid at x=0 and z=0 boundaries
        preval = [gridmp[d][0] - (gridmp[d][1] - gridmp[d][0]) for d in range(DIM)]
        grids = [                                                   \
                [ grid[IZ], np.insert(gridmp[IX], 0, preval[IX]) ], \
                [ np.insert(gridmp[IZ], 0, preval[IZ]), grid[IX] ]  \
                ]
        vels  = [                                                  \
                np.hstack((np.zeros(nx[IZ])[:,None], newvel[IZ])), \
                np.vstack((np.zeros(nx[IX])[None,:], newvel[IX]))  \
                ]

        if bcstokes[DIM*0 + IZ] == pylamp_stokes.BC_TYPE_NOSLIP:
            vels[IX][0,:] = 0
        elif bcstokes[DIM*0 + IZ] == pylamp_stokes.BC_TYPE_FREESLIP:
            vels[IX][0,:] = vels[IX][1,:]
        vels[IZ][0,:] = 0

        if bcstokes[DIM*0 + IX] == pylamp_stokes.BC_TYPE_NOSLIP:
            vels[IZ][:,0] = 0
        elif bcstokes[DIM*0 + IX] == pylamp_stokes.BC_TYPE_FREESLIP:
            vels[IZ][:,0] = vels[IZ][:,1]
        vels[IX][:,0] = 0

        trac_vel, tracs_new = pylamp_trac.RK(tr_x, grids, vels, nx, tstep)
        #trac_vel, tracs_new = pylamp_trac.RK(tr_x, gridmp, newvel, nx, tstep)
        tr_x[:,:] = tracs_new[:,:]

        # do not allow tracers to advect outside the domain
        for d in range(DIM):
            tr_x[tr_x[:,d] <= 0, d] = EPS
            tr_x[tr_x[:,d] >= L[d], d] = L[d]-EPS

        if it % 100 == 1:
            print("Plot")
            plt.close('all')
            fig = plt.figure()

            ax = fig.add_subplot(221)
            #ax.pcolormesh(newvel[0])
            ax.pcolormesh(f_rho)

            #ax = fig.add_subplot(222)
            #ax.pcolormesh(newvel[1])
            #ax.pcolormesh(np.log10(f_etan))

            ax = fig.add_subplot(222)
            vxgrid = (newvel[IX][:-1,1:] + newvel[IX][:-1,0:-1]) / 2
            vzgrid = (newvel[IZ][1:,:-1] + newvel[IZ][0:-1,:-1]) / 2
            ax.pcolormesh(vzgrid)
#            ax.quiver(meshmp[IX].flatten('F'), meshmp[IZ].flatten('F'), vxgrid.flatten('F'), vzgrid.flatten('F'))

            ax = fig.add_subplot(223)
            ax.quiver(tr_x[::1,IX], tr_x[::1,IZ], trac_vel[::1,IX]*tstep, trac_vel[::1,IZ]*tstep, angles='xy', scale_units='xy', scale=0.5)
            ax.set_xticks(grid[IX])
            ax.set_yticks(grid[IZ])
            ax.xaxis.grid()
            ax.yaxis.grid()
            #ax = fig.add_subplot(224)
            #ax.pcolormesh(f_rho)
            #plt.show()

            ### divergence field
            #ax = fig.add_subplot(222)
            #div_dv = newvel[IZ][1:,:-1] - newvel[IZ][:-1,:-1]
            #div_du = newvel[IX][:-1,1:] - newvel[IX][:-1,:-1]
            #div_dz = mesh[IZ][1:,:-1] - mesh[IZ][:-1,:-1]
            #div_dx = mesh[IX][:-1,1:] - mesh[IX][:-1,:-1]
            ##divergence = (div_dv.T / div_dz).T[:,:-1] + (div_du / div_dx)[:-1,:]
            #divergence = div_dv / div_dz + div_du / div_dx
            #cs = ax.pcolormesh(divergence)
            #plt.colorbar(cs)
            ###CS=ax.contourf(midp_x, midp_y, divfield)
            ###plt.colorbar(CS)

            ## marker field with triangulated interpolation
            #ax = fig.add_subplot(224)
            #ax.tripcolor(tr_x[:,IX], tr_x[:,IZ], tr_f[:,TR_MRK])

            ax = fig.add_subplot(224)
            xi = np.linspace(0,L[IX],100)
            yi = np.linspace(0,L[IZ],100)
            zi = scipy.interpolate.griddata((tr_x[:,IX], tr_x[:,IZ]), tr_f[:,TR_MRK], (xi[None,:], yi[:,None]), method='linear')
            cs = ax.contourf(xi,yi,zi,15,cmap=plt.cm.jet)

            plt.show()

            #dummy = input()

    sys.exit()

#
#  :::: temperature ::::
#
#         j
#
#   x--v--x--v--x--v--x  v         x:  T
#   |     |     |     |            v:  qx, kx
#   o     o     o     o            o:  qz, kz
#   |     |     |     |
# i x--v--X--V--x--v--x  v
#   |     |     |     |
#   o     O     o     o
#   |     |     |     |
#   x--v--x--v--x--v--x  v
#
#   o     o     o     o
#
#   
#   qx_i_j = kx_i_j * (T_i_j+1 - T_i_j) / (x_i_j+1 - x_i_j)
#   qz_i_j = kz_i_j * (T_i+1_j - T_i_j) / (x_i+1_j - x_i_j)
# =>
#   qx_i_j-1 = kx_i_j-1 * (T_i_j - T_i_j-1) / (x_i_j - x_i_j-1)
#   qz_i-1_j = kz_i-1_j * (T_i_j - T_i-1_j) / (x_i_j - x_i-1_j)
#
#
#   rho_i_j * Cp_i_j * dT_i_j = dt * [ (qx_i_j - qx_i_j-1) / (x_i_j - x_i_j-1) + (qz_i_j - qz_i-1_j) / (x_i_j - x_i-1_j) ]
#
#   dT_i_j = A_i_j * {
#              [ kx_i_j * (T_i_j+1 - T_i_j) / (x_i_j+1 - x_i_j) - kx_i_j-1 * (T_i_j - T_i_j-1) / (x_i_j - x_i_j-1) ] / (x_i_j - x_i_j-1) + 
#              [ kz_i_j * (T_i+1_j - T_i_j) / (x_i+1_j - x_i_j) - kz_i-1_j * (T_i_j - T_i-1_j) / (x_i_j - x_i-1_j) ] / (x_i_j - x_i-1_j) 
#   }
#
#  A_i_j = dt / (rho_i_j * Cp_i_j)
#
#  T_i_j+1:    kx_i_j / (x_i_j+1 - x_i_j) / (x_i_j - x_i_j-1)
#  T_i_j  :    -kx_i_j / (x_i_j+1 - x_i_j) / (x_i_j - x_i_j-1) +
#              -kx_i_j-1 / (x_i_j - x_i_j-1) / (x_i_j - x_i_j-1) +
#              -kz_i_j / (x_i+1_j - x_i_j) / (x_i_j - x_i-1_j) +
#              -kz_i-1_j / (x_i_j - x_i-1_j) / (x_i_j - x_i-1_j)
#  T_i_j-1:    kx_i_j-1 / (x_i_j - x_i_j-1) / (x_i_j - x_i_j-1)
#  T_i+1_j:    kz_i_j / (x_i+1_j - x_i_j) / (x_i_j - x_i-1_j)
#  T_i-1_j:    kz_i-1_j / (x_i_j - x_i-1_j) / (x_i_j - x_i-1_j)
#

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
