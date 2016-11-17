#!/usr/bin/python3

from pylamp_const import *
import pylamp_stokes 
import pylamp_trac
import pylamp_diff
import pylamp_io
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.sparse.linalg
#from mpi4py import MPI
import importlib
#import vtk
#from vtk.util import numpy_support
from scipy.stats import gaussian_kde

importlib.reload(pylamp_stokes)
importlib.reload(pylamp_trac)
importlib.reload(pylamp_diff)
importlib.reload(pylamp_io)

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
    nx    =   [33,50]         # use order z,x,y
    L     =   [660e3, 1000e3]
    tracdens = 40   # how many tracers per element
    
    do_stokes = True
    do_advect = True
    do_heatdiff = True

    tstep_adv_max = 50e9 * SECINYR
    tstep_adv_min = 50e-9 * SECINYR
    tstep_dif_max = 50e9 * SECINYR
    tstep_dif_min = 50e-9 * SECINYR
    tstep_modifier = 0.67             # coefficient for automatic tsteps

    output_file = True
    output_screen = False
    output_vtk = False
    output_stride = 5

    tdep_rho = True
    tdep_eta = True
    etamin = 1e18
    etamax = 1e23
    Tref = 1723
    
    force_trac2grid_T = True       # force tracer to grid interpolation even in the case when there is no advection
    max_it = 99999
    bc_internal_type = 1           # 0 = disabled
    surface_stabilization = False   # use if "sticky air" free surface present
    surfstab_theta = 0.5           


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
    f_H    =   np.zeros(nx)    # internal heating

    # Tracers
    ntrac = np.prod(nx)*tracdens

    tr_x = np.random.rand(ntrac, DIM)  # tracer coordinates
    tr_x = np.multiply(tr_x, L)
    tr_f = np.zeros((ntrac, NFTRAC))     # tracer functions (values)


    ## Some material values and initial values

    ## Falling block
    # Density
    #tr_f[:, TR_RH0] = 3300
    #idxx = (tr_x[:, IX] < 550e3) & (tr_x[:, IX] > 450e3)
    #idxz = (tr_x[:, IZ] < 380e3) & (tr_x[:, IZ] > 280e3)
    ##tr_f[idxx & idxz, TR_RH0] = 3340
    #tr_f[idxx & idxz, TR_RH0] = 3260
    #tr_f[:, TR_ALP] = 0

    ## ... sticky air test
    #idxz2 = tr_x[:, IZ] < 50e3
    #tr_f[idxz2, TR_RHO] = 1
    #tr_f[idxz2, TR_ET0] = 1e17

    ## Viscosity
    #tr_f[:, TR_ET0] = 1e17
    #tr_f[idxx & idxz, TR_ET0] = 1e20

    #### RT instability, double-sided
    #tr_f[:, TR_RH0] = 3300
    #idxz = (tr_x[:, IZ] < 150e3)
    #tr_f[idxz, TR_RH0] = 3340
    #tr_f[:, TR_ALP] = 0.0
    #tr_f[:, TR_ET0] = 1e19
    #tr_f[idxz, TR_ET0] = 1e20
    #idxz = (tr_x[:, IZ] > 510e3)
    #tr_f[idxz, TR_RH0] = 3260
    #tr_f[idxz, TR_ET0] = 1e20

    ### heat diffusion test
    #tr_f[:, TR_RH0] = 3300
    #tr_f[:, TR_ALP] = 0.0
    #tr_f[:, TR_HCD] = 4.0
    #tr_f[:, TR_HCP] = 1250
    #tr_f[:, TR_TMP] = 273

    

    #### lava lamp
    #tr_f[:, TR_RH0] = 3300
    #tr_f[:, TR_ALP] = 3.5e-5

    #tr_f[:, TR_ET0] = 1e18
    #tr_f[tr_x[:,IZ] > 560e3, TR_ET0] = 1e19

    #tr_f[:, TR_HCD] = 4.0
    #tr_f[:, TR_HCP] = 1250
    #tr_f[:, TR_TMP] = 273


    #### lava lamp with free surface
    #idxzair = tr_x[:, IZ] < 0e3
    #tr_f[:, TR_RH0] = 3300
    #tr_f[:, TR_ALP] = 3.5e-5
    #tr_f[:, TR_MAT] = 1
    #tr_f[:, TR_ET0] = 1e19
    #tr_f[:, TR_HCD] = 4.0
    #tr_f[:, TR_HCP] = 1250
    #tr_f[:, TR_TMP] = 273

    #tr_f[idxzair, TR_MAT] = 0    # water
    #tr_f[idxzair, TR_RH0] = 1000 
    #tr_f[idxzair, TR_ALP] = 0
    #tr_f[idxzair, TR_ET0] = 1e17
    #tr_f[idxzair, TR_HCD] = 1e-10    # we use internal bc to force all water to constant temp
    #tr_f[idxzair, TR_HCP] = 1000
    #tr_f[idxzair, TR_TMP] = 273


    # Stagnant lid?
    tr_f[:, TR_RH0] = 3300
    tr_f[:, TR_ALP] = 3.5e-5
    tr_f[:, TR_MAT] = 1
    tr_f[:, TR_ET0] = 1e19
    tr_f[:, TR_HCD] = 4.0
    tr_f[:, TR_HCP] = 1250
    tr_f[:, TR_TMP] = 1623
    tr_f[tr_x[:,IZ] < 50e3, TR_TMP] = 273
    tr_f[:, TR_ACE] = 360e3
    tr_f[:, TR_IHT] = 0.1e-6


    ## Passive markers
    inixdiv = np.linspace(0, L[IX], 10)
    inizdiv = np.linspace(0, L[IZ], 10)
    for i in range(0,9,2):
        tr_f[(tr_x[:,IZ] >= inizdiv[i]) & (tr_x[:,IZ] < inizdiv[i+1]), TR_MRK] += 1
    for i in range(1,9,2):
        tr_f[(tr_x[:,IZ] >= inizdiv[i]) & (tr_x[:,IZ] < inizdiv[i+1]), TR_MRK] += 2
    for i in range(0,9,2):
        tr_f[(tr_x[:,IX] >= inixdiv[i]) & (tr_x[:,IX] < inixdiv[i+1]), TR_MRK] *= -1

    if do_advect ^ do_stokes:
        raise Exception("Not implemented yet. Both do_advect and do_stokes need to be either disabled or enabled.")

    it = 0
    totaltime = 0
    while (it < max_it):
        it += 1
        print("\n --- Time step:", it, "---")

        print("Calculate physical properties")
        if tdep_rho:
            # Effective density, rho=rho(T, inherent density)
            tr_f[:, TR_RHO] = ((tr_f[:, TR_ALP] * (tr_f[:, TR_TMP] - Tref) + 1) / tr_f[:, TR_RH0])**(-1)
        else:
            tr_f[:, TR_RHO] = tr_f[:, TR_RH0]

        if tdep_eta:
            # Effective viscosity, eta=eta(T, inherent viscosity)
            tr_f[:, TR_ETA] = tr_f[:, TR_ET0] * np.exp(tr_f[:, TR_ACE] / (GASR * tr_f[:, TR_TMP]) - tr_f[:, TR_ACE] / (GASR * Tref))
            print (np.max(tr_f[:, TR_ETA]), np.min(tr_f[:, TR_ETA]))
            tr_f[tr_f[:, TR_ETA] < etamin, TR_ETA] = etamin
            tr_f[tr_f[:, TR_ETA] > etamax, TR_ETA] = etamax
        else:
            tr_f[:, TR_ETA] = tr_f[:, TR_ET0]


        if bc_internal_type > 0:

            if bc_internal_type == 1:
                # force material zero (water, air) to constant temperature
                idxmat = tr_f[:, TR_MAT] == 0
                tr_f[idxmat, TR_TMP] = 273

        print("Properties trac2grid")

        if do_advect and do_heatdiff:
            # Interpolation done once on each different grid, multiple value fields at once
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_RHO, TR_ETA, TR_HCP, TR_TMP, TR_IHT]], mesh, grid, [f_rho, f_etas, f_Cp, f_T, f_H], nx, \
                    avgscheme=[pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_GEOMW, pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_ARITHW])
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_ETA]], meshmp, gridmp, [f_etan], nx, avgscheme=[pylamp_trac.INTERP_AVG_GEOMW])
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_HCD]], [meshmp[IZ], mesh[IX]], [gridmp[IZ], grid[IX]], [f_k[IZ]], nx, avgscheme=[pylamp_trac.INTERP_AVG_ARITHW])
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_HCD]], [mesh[IZ], meshmp[IX]], [grid[IZ], gridmp[IX]], [f_k[IX]], nx, avgscheme=[pylamp_trac.INTERP_AVG_ARITHW])

        elif do_advect:
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_RHO, TR_ETA]], mesh, grid, [f_rho, f_etas], nx, \
                    avgscheme=[pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_GEOMW])
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_ETA]], meshmp, gridmp, [f_etan], nx, avgscheme=[pylamp_trac.INTERP_AVG_GEOMETRIC])

        elif do_heatdiff:
            if it == 1 or tdep_rho or force_trac2grid_T:
                pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_RHO, TR_HCP, TR_TMP]], mesh, grid, [f_rho, f_Cp, f_T], nx, 
                        avgscheme=[pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_ARITHW])
                pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_HCD]], [meshmp[IZ], mesh[IX]], [gridmp[IZ], grid[IX]], [f_k[IZ]], nx, avgscheme=[pylamp_trac.INTERP_AVG_ARITHW])
                pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_HCD]], [mesh[IZ], meshmp[IX]], [grid[IZ], gridmp[IX]], [f_k[IX]], nx, avgscheme=[pylamp_trac.INTERP_AVG_ARITHW])
            else:
                ### after the first time step (if no temperature dependen rho) we only need to interpolate temperature, since there is no advection
                ### actually, let's skip that, too, and copy the grid directly
                f_T = newtemp

        if do_heatdiff:
            diffusivity = f_k[IZ] / (f_rho * f_Cp)
            tstep_temp = 0.67 * np.min(dx)**2 / np.max(2*diffusivity)
            tstep_temp = min(tstep_temp, tstep_dif_max)
            tstep_temp = max(tstep_temp, tstep_dif_min)

        newvel = 0
        newpres = 0

        if do_stokes:
            print("Build stokes")
            bcstokes = [[]] * 4
            bcstokes[DIM*0 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
            bcstokes[DIM*1 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
            bcstokes[DIM*0 + IX] = pylamp_stokes.BC_TYPE_FREESLIP
            bcstokes[DIM*1 + IX] = pylamp_stokes.BC_TYPE_FREESLIP

            (A, rhs) = pylamp_stokes.makeStokesMatrix(nx, grid, f_etas, f_etan, f_rho, bcstokes)

            print("Solve stokes")
            # Solve it!
            #x = scipy.sparse.linalg.bicgstab(scipy.sparse.csc_matrix(A), rhs)[0]
            x = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), rhs)

            (newvel, newpres) = pylamp_stokes.x2vp(x, nx)

            tstep_stokes = 0.67 * np.min(dx) / np.max(newvel)
            tstep_stokes = min(tstep_stokes, tstep_adv_max)
            tstep_stokes = max(tstep_stokes, tstep_adv_min)

        if do_heatdiff and do_advect:
            tstep = min(tstep_temp, tstep_stokes)
        elif do_heatdiff:
            tstep = tstep_temp
        else:
            tstep = tstep_stokes

        if do_stokes and surface_stabilization:
            print ("Redo stokes with surface stabilization")
            (A, rhs) = pylamp_stokes.makeStokesMatrix(nx, grid, f_etas, f_etan, f_rho, bcstokes, surfstab=True, tstep=tstep, surfstab_theta=surfstab_theta)

            print ("Resolve stokes")
            x = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), rhs)
            #(x, Aerr) = scipy.sparse.linalg.bicgstab(scipy.sparse.csc_matrix(A), rhs, x0=x)
            #print ("  resolve error: ", Aerr)
            (newvel, newpres) = pylamp_stokes.x2vp(x, nx)

            tstep_stokes = 0.67 * np.min(dx) / np.max(newvel)
            tstep = min(tstep_stokes, tstep)

        totaltime += tstep
        print("   time step =", tstep/SECINKYR, "kyrs")
        print("   time now  =", totaltime/SECINKYR, "kyrs")


        if do_heatdiff:
            print("Build heatdiff")
            bcheat = [[]] * 4
            bcheat[DIM*0 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP
            bcheat[DIM*1 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP
            bcheat[DIM*0 + IX] = pylamp_diff.BC_TYPE_FIXFLOW
            bcheat[DIM*1 + IX] = pylamp_diff.BC_TYPE_FIXFLOW

            bcheatvals = [[]] * 4
            bcheatvals[DIM*0 + IZ] = 273
            bcheatvals[DIM*1 + IZ] = 1623
            bcheatvals[DIM*0 + IX] = 0
            bcheatvals[DIM*1 + IX] = 0

            (A, rhs) = pylamp_diff.makeDiffusionMatrix(nx, grid, gridmp, f_T, f_k, f_Cp, f_rho, f_H, bcheat, bcheatvals, tstep)

            print("Solve diffusion")

            x = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), rhs)

            newtemp = pylamp_diff.x2t(x, nx)
            
            old_tr_f = np.array(tr_f, copy=True)

            interp_tracvals = np.zeros((tr_f.shape[0], 1))
            if it >= 1:
                # On first timestep, interpolate absolute temperature values to tracers ...
                print("grid2trac T")
                pylamp_trac.grid2trac(tr_x, interp_tracvals, grid, [newtemp], nx, method=pylamp_trac.INTERP_METHOD_LINEAR, stopOnError=True)
                tr_f[:, TR_TMP] = interp_tracvals[:, 0]
            else:
                # ... on subsequent timesteps interpolate only the change to avoid numerical diffusion
                print("grid2trac dT")
                newdT = newtemp - f_T
                pylamp_trac.grid2trac(tr_x, interp_tracvals, grid, [newdT], nx, method=pylamp_trac.INTERP_METHOD_LINEAR + pylamp_trac.INTERP_METHOD_DIFF, stopOnError=True)
                tr_f[:, TR_TMP] += interp_tracvals[:, 0]

                #### subgrid diffusion
                ## correction at tracers
                #subgrid_corr_dt0 = tr_f[:, TR_HCP] * tr_f[:, TR_RHO] / (tr_f[:, TR_HCD] * (2/dx[IX]**2 + 2/dx[IZ]**2))
                #subgrid_corr_T = old_tr_f[:, TR_TMP] - (old_tr_f[:, TR_TMP] - tr_f[:, TR_TMP]) * np.exp(-d * tstep / subgrid_corr_dt0)
                ## compensation at eulerian nodes


        if do_advect:
            print("Tracer advection")
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
            tr_x[:,:] = tracs_new[:,:]

            # do not allow tracers to advect outside the domain
            for d in range(DIM):
                tr_x[tr_x[:,d] <= 0, d] = EPS
                tr_x[tr_x[:,d] >= L[d], d] = L[d]-EPS

        #vtkdata = numpy_support.numpy_to_vtk(num_array=f_T.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        #
        #vtkxcoords = vtk.vtkFloatArray()
        #vtkzcoords = vtk.vtkFloatArray()
        #vtkycoords = vtk.vtkFloatArray()
        #for i in grid[IZ]:
        #    vtkzcoords.InsertNextValue(i)
        #for j in grid[IX]:
        #    vtkxcoords.InsertNextValue(j)
        #if DIM == 3:
        #    for k in grid[IY]:
        #        vtkycoords.InsertNextValue(k)
        #else:
        #    vtkycoords.InsertNextValue(0)

        #vtkrlg = vtk.vtkRectilinearGrid()
        #if DIM == 2:
        #    vtkdim = [nx[IX], 1, nx[IZ]]
        #else:
        #    vtkdim = [nx[IX], nx[IY], nx[IZ]]
        #vtkrlg.SetDimensions(*vtkdim)
        #vtkrlg.SetZCoordinates(vtkzcoords)
        #vtkrlg.SetXCoordinates(vtkxcoords)
        #vtkrlg.SetYCoordinates(vtkycoords)
        #vtkarr = vtk.vtkShortArray()
        #vtkarr.SetNumberOfComponents(1)

        if output_vtk:
            pylamp_io.vtkOutPoints(tr_x, tr_f[:,[TR_TMP]], ["temperature"], "tracs_{:06d}.vtk".format(it))

        if it % 1 == 1:
            print("Plot")
            plt.close('all')
            cs = plt.pcolormesh(f_T)
            plt.colorbar(cs)
            plt.show()


        if output_screen or output_file:
            if (it-1) % output_stride == 0:
                print("Plot")
                plt.close('all')
                fig = plt.figure()

                ax = fig.add_subplot(221)
                if do_advect:
                    #ax.pcolormesh(newvel[0])
                    #cs = ax.pcolormesh(f_rho)
                    cs = ax.pcolormesh(np.log10(f_etas))
                    plt.colorbar(cs)
                    #xi = np.linspace(0, L[IX], 100)
                    #yi = np.linspace(0, L[IZ], 100)
                    #zi = scipy.interpolate.griddata((tr_x[:,IX], tr_x[:,IZ]), tr_f[:,TR_RHO], (xi[None,:], yi[:,None]), method='linear')
                    #cs = ax.contourf(xi,yi,zi,cmap=plt.cm.jet,levels=[0,980,1020,3240,3280,3300,3320])
                    #ax.scatter(tr_x[::10,IX], tr_x[::10,IZ], c=tr_f[::10,TR_RHO], marker='.', linewidths=(0,))
                    #nskip=10
                    #ax.tripcolor(tr_x[::nskip,IX], tr_x[::nskip,IZ], tr_f[::nskip,TR_RHO])
                elif do_heatdiff:
                    cs = plt.scatter(tr_x[:,IX], tr_x[:,IZ], marker='+', s=20, linewidths=1, c=tr_f[:,TR_TMP], cmap=plt.cm.coolwarm)
                    plt.colorbar(cs)
                    ax.set_xticks(grid[IX])
                    ax.set_yticks(grid[IZ])
                    ax.xaxis.grid()
                    ax.yaxis.grid()

                #print("calc")
                #idxz = tr_x[:, IZ] > 510e3
                #xy = np.vstack([tr_x[idxz,IX],tr_x[idxz,IZ]])
                #z = gaussian_kde(xy)(xy)
                #print("plot")
                #cs = ax.scatter(tr_x[idxz,IX], tr_x[idxz, IZ], c=z, s=100, edgecolor='')
                #plt.colorbar(cs)



                #ax = fig.add_subplot(222)
                #ax.pcolormesh(newvel[1])
                #ax.pcolormesh(np.log10(f_etan))

                if do_advect:
                    ax = fig.add_subplot(222)
                    #vxgrid = (newvel[IX][:-1,1:] + newvel[IX][:-1,0:-1]) / 2
                    #vzgrid = (newvel[IZ][1:,:-1] + newvel[IZ][0:-1,:-1]) / 2
                    #cs = ax.pcolormesh(vzgrid)
                    #plt.colorbar(cs)
                    #ax.quiver(meshmp[IX].flatten('F'), meshmp[IZ].flatten('F'), vxgrid.flatten('F'), vzgrid.flatten('F'))
                    #cs = ax.pcolormesh(f_rho)
                    #plt.colorbar(cs)
                    cs = ax.scatter(tr_x[::10,IX], tr_x[::10,IZ], c=tr_f[::10,TR_RHO], marker='+', s=50, linewidths=4, cmap=plt.cm.coolwarm)
                    plt.colorbar(cs)

                    ax = fig.add_subplot(223)
                    stride = 10
                    ax.quiver(tr_x[::stride,IX], tr_x[::stride,IZ], trac_vel[::stride,IX]*tstep, trac_vel[::stride,IZ]*tstep, angles='xy', scale_units='xy', scale=0.2)
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
                #xi = np.linspace(0,L[IX],100)
                #yi = np.linspace(0,L[IZ],100)
                #zi = scipy.interpolate.griddata((tr_x[:,IX], tr_x[:,IZ]), tr_f[:,TR_TMP], (xi[None,:], yi[:,None]), method='linear')
                ##zi = scipy.interpolate.griddata((tr_x[:,IX], tr_x[:,IZ]), trac_dT[:,0], (xi[None,:], yi[:,None]), method='linear')
                #cs = ax.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
                #plt.colorbar(cs)
                cs = plt.pcolormesh(f_T)
                plt.colorbar(cs)

                if output_file:
                    fig.savefig("fig_{:04d}.png".format(it))

                if output_screen:
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


#   [ kx_i_j * (T_i_j+1 - T_i_j) / (x_i_j+1 - x_i_j)  -  kx_i_j-1 * (T_i_j - T_i_j-1) / (x_i_j - x_i_j-1) ] / (x_i_j+½ - x_i_j-½)  +
#   [ kz_i_j * (T_i+1_j - T_i_j) / (x_i+1_j - x_i_j)  -  kz_i-1_j * (T_i_j - T_i-1_j) / (x_i_j - x_i-1_j) ] / (x_i+½_j - x_i-½_j)
#    = rho_i_j * Cp_i_j * (T_i_j - T_i_j^old) / dt
#
#   T_i_j+1 : kx_i_j / (x_i_j+1 - x_i_j) / (x_i_j+½ - x_i_j-½)
#   T_i_j   : -kx_i_j / (x_i_j+1 - x_i_j) / (x_i_j+½ - x_i_j-½) +
#             -kx_i_j-1 / (x_i_j - x_i_j-1) / (x_i_j+½ - x_i_j-½) +
#             -kz_i_j / (x_i+1_j - x_i_j) / (x_i+½_j - x_i-½_j) +
#             -kz_i-1_j / (x_i_j - x_i-1_j) / (x_i+½_j - x_i-½_j)
#   T_i_j-1 : kx_i_j-1 / (x_i_j - x_i_j-1) / (x_i_j+½ - x_i_j-½)
#   T_i+1_j : kz_i_j / (x_i+1_j - x_i_j) / (x_i+½_j - x_i-½_j) 
#   T_i-1_j : kz_i-1_j / (x_i_j - x_i-1_j) / (x_i+½_j - x_i-½_j)


#   
#   qx_i_j = -kx_i_j * (T_i_j+1 - T_i_j) / (x_i_j+1 - x_i_j)
#   qz_i_j = -kz_i_j * (T_i+1_j - T_i_j) / (x_i+1_j - x_i_j)
# =>
#   qx_i_j-1 = -kx_i_j-1 * (T_i_j - T_i_j-1) / (x_i_j - x_i_j-1)
#   qz_i-1_j = -kz_i-1_j * (T_i_j - T_i-1_j) / (x_i_j - x_i-1_j)
#
#   rho_i_j * Cp_i_j * dT_i_j = dt * [ (qx_i_j - qx_i_j-1) / (x_i_j - x_i_j-1) + (qz_i_j - qz_i-1_j) / (x_i_j - x_i-1_j) ]
#
#   if A_i_j = dt / (rho_i_j * Cp_i_j):
#   
#     dT_i_j = A_i_j * [ (qx_i_j - qx_i_j-1) / (x_i_j - x_i_j-1) + (qz_i_j - qz_i-1_j) / (x_i_j - x_i-1_j) ]    + dt*H/(rho*Cp)
#     T_i_j - T_i_j^old = A_i_j * [ (qx_i_j - qx_i_j-1) / (x_i_j - x_i_j-1) + (qz_i_j - qz_i-1_j) / (x_i_j - x_i-1_j) ]    + dt*H/(rho*Cp)
#     -T_i_j^old = A_i_j * [ (qx_i_j - qx_i_j-1) / (x_i_j - x_i_j-1) + (qz_i_j - qz_i-1_j) / (x_i_j - x_i-1_j) ] - T_i_j    + dt*H/(rho*Cp)
#     T_i_j^old = -A_i_j * [ (qx_i_j - qx_i_j-1) / (x_i_j - x_i_j-1) + (qz_i_j - qz_i-1_j) / (x_i_j - x_i-1_j) ] + T_i_j    - dt*H/(rho*Cp)
#
#   T_i_j^old + dt*H/(rho*Cp) = -A_i_j * {
#              [ -kx_i_j * (T_i_j+1 - T_i_j) / (x_i_j+1 - x_i_j) + -kx_i_j-1 * (T_i_j - T_i_j-1) / (x_i_j - x_i_j-1) ] / (x_i_j - x_i_j-1) + 
#              [ -kz_i_j * (T_i+1_j - T_i_j) / (x_i+1_j - x_i_j) + -kz_i-1_j * (T_i_j - T_i-1_j) / (x_i_j - x_i-1_j) ] / (x_i_j - x_i-1_j) 
#   } + T_i_j
#
#  Components 
#  T_i_j+1:    kx_i_j / (x_i_j+1 - x_i_j) / (x_i_j - x_i_j-1)
#  T_i_j  :    -kx_i_j / (x_i_j+1 - x_i_j) / (x_i_j - x_i_j-1) +
#              -kx_i_j-1 / (x_i_j - x_i_j-1) / (x_i_j - x_i_j-1) +
#              -kz_i_j / (x_i+1_j - x_i_j) / (x_i_j - x_i-1_j) +
#              -kz_i-1_j / (x_i_j - x_i-1_j) / (x_i_j - x_i-1_j) +
#              1
#  T_i_j-1:    kx_i_j-1 / (x_i_j - x_i_j-1) / (x_i_j - x_i_j-1)
#  T_i+1_j:    kz_i_j / (x_i+1_j - x_i_j) / (x_i_j - x_i-1_j)
#  T_i-1_j:    kz_i-1_j / (x_i_j - x_i-1_j) / (x_i_j - x_i-1_j)
#
#  BC:
#    x = 0 (, z = 0):
#      T=Tx0
#      or 
#      k*dT/dx = qx0 
#      i.e. k_i_j * (T_i_j+1 - T_i_j) / (x_i_j+1 - x_i_j) = qx0
#       =>  T_i_j+1:  k_i_j / (x_i_j+1 - x_i_j)
#           T_i_j  : -k_i_j / (x_i_j+1 - x_i_j)
#           rhs    :  qx0



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
#  Correction term by Kaus or Duretz et al:
#    d[Sxy] / dy + T * dt * (d[rho] / dx) * vx * gx + T * dt * (d[rho] / dy) * vy * gx = -rho*gx (???)    ### for x-stokes
#    d[Syx] / dx + T * dt * (d[rho] / dy) * vy * gy + T * dt * (d[rho] / dx) * vx * gy = -rho*gy          ### for y-stokes
#    Extra terms from these:
#      x-stokes:
#        vx_i_j : theta * dt * gx * (0.5 * (rho_i_j+1 + rho_i+1_j+1) - 0.5 * (rho_i_j-1 + rho_i+1_j-1)) / (x_i_j+1 - x_i_j-1)
#        vy_i_j : theta * dt * gx * (rho_i+1_j - rho_i_j) / (y_i+1_j - y_i_j)
#      y-stokes:
#        vy_i_j : theta * dt * gy * (0.5 * (rho_i+1_j + rho_i+1_j+1) - 0.5 * (rho_i-1_j + rho_i-1_j+1)) / (y_i+1_j - y_i-1_j)
#        vx_i_j : theta * dt * gy * (rho_i_j+1 - rho_i_j) / (x_i_j+1 - x_i_j)
#
#  T = theta = 0.5(?)
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
