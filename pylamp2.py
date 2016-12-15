#!/usr/bin/python3

from mpi4py import MPI
from pylamp_const import *
import pylamp_stokes 
import pylamp_trac
import pylamp_diff
import numpy as np
import sys
import scipy.sparse.linalg
from scipy.stats import gaussian_kde
import cProfile, pstats, io
from pylamp_tool import *

### PyLamp
#
# Python code to solve the conservation of energy, momentum and mass
# incompressible viscous flow
#
# – implicit (scipy direct solver)
# – marker-in-cell for material and temperature advection
# – linear (Newtonian) viscosity
# – temperature dependent viscosity and density (buoyancy)
#
#

#### MAIN ####
if __name__ == "__main__":

    MPICOMM = MPI.COMM_WORLD
    IPROC = MPICOMM.Get_rank()
    NPROC = MPICOMM.Get_size()

    pprint("=== Running on", NPROC, "processors")

    # Configurable options
    nx    =   [33+1,45+1]         # use order z,x,y
    L     =   [660e3, 1800e3] 
    tracdens = 40   # how many tracers per element
    
    do_stokes = True
    do_advect = True
    do_heatdiff = True

    tstep_adv_max = 50e9 * SECINYR
    tstep_adv_min = 50e-9 * SECINYR
    tstep_dif_max = 50e9 * SECINYR
    tstep_dif_min = 50e-9 * SECINYR
    tstep_modifier = 0.67             # coefficient for automatic tsteps

    output_numpy = True
    output_stride = 2
    output_stride_ma = 1            # used if output_stride < 0: output fields every x million years
    output_outdir = "out"

    tdep_rho = True
    tdep_eta = True
    etamin = 1e17
    etamax = 1e23
    Tref = 1623
    
    force_trac2grid_T = False       # force tracer to grid interpolation even in the case when there is no advection
    max_it = 1e10
    max_time = SECINMYR * 5000
    bc_internal_type = 0           # 0 = disabled
                                   # 1 = keep material zero at constant temperature T=273K
    surface_stabilization = False   # use if "sticky air" free surface present
    surfstab_theta = 0.5
    surfstab_tstep = -1 #1*SECINKYR            # if negative, a dynamic tstep is used 

    do_profiling = False

    choose_model = 11


    # Profiling
    if do_profiling:
        pr = cProfile.Profile()
        pr.enable()

    # Derived options
    # dx for regular grid
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
    f_mat  =   np.zeros(nx)    # material numbers

    f_sgc  =   np.zeros(nx)    # subgrid diffusion correction term

    # Tracers
    ntrac = np.prod(nx)*tracdens

    if IPROC == 0:
        tr_x = np.random.rand(ntrac, DIM)  # tracer coordinates
    else:
        tr_x = np.zeros((ntrac, DIM))
    MPICOMM.Bcast(tr_x, root=0)

    tr_x = np.multiply(tr_x, L)
    tr_f = np.zeros((ntrac, NFTRAC))     # tracer functions (values)

    tr_f[:, TR__ID] = np.arange(0, ntrac)

    
    ## Some material values and initial values

    if choose_model == 1 or choose_model == 11:
        if choose_model == 1:
            zair = tr_x[:,IZ] < 0
            zcrust = tr_x[:,IZ] < 50e3
        else:
            zair = tr_x[:,IZ] < 50e3
            extraidx = (tr_x[:, IZ] > 100e3) & (tr_x[:, IZ] < 150e3) & (tr_x[:, IX] < 600e3) & (tr_x[:, IX] > 200e3)
            zcrust = (tr_x[:,IZ] < 100e3) | extraidx
            tstep_modifier = 0.33
        # Stagnant lid?
        tr_f[:, TR_RH0] = 3300
        tr_f[:, TR_ALP] = 3.5e-5
        tr_f[:, TR_MAT] = 2
        tr_f[:, TR_ET0] = 1e20
        tr_f[:, TR_HCD] = 4.0
        tr_f[:, TR_HCP] = 1250
        tr_f[:, TR_TMP] = 1623
        tr_f[:, TR_ACE] = 120e3
        tr_f[:, TR_IHT] = 0.02e-6 / 3300
        tr_f[zcrust, TR_IHT] = 2.5e-6 / 2900 #1e-6
        tr_f[zcrust, TR_RH0] = 2900
        tr_f[zcrust, TR_MAT] = 1
        tr_f[zcrust, TR_ET0] = 1e22
        tr_f[zcrust, TR_HCD] = 2.5
        tr_f[zcrust, TR_HCP] = 1000
        tr_f[zcrust, TR_TMP] = 273
        tr_f[zair, TR_RH0] = 1000
        tr_f[zair, TR_ALP] = 0
        tr_f[zair, TR_ET0] = 1e18
        tr_f[zair, TR_ACE] = 0
        tr_f[zair, TR_TMP] = 273
        tr_f[zair, TR_MAT] = 0
        tr_f[zair, TR_HCD] = 40
        tr_f[zair, TR_IHT] = 0.0
        tr_f[zair, TR_HCP] = 1000

    elif choose_model == 2:
        # Falling block
        do_heatdiff = False
        tdep_rho = False
        tdep_eta = False
        tr_f[:, TR_RH0] = 3300
        tr_f[:, TR_MAT] = 1
        tr_f[:, TR_ET0] = 1e19
        idxb = (tr_x[:, IZ] > 200e3) & (tr_x[:, IZ] < 300e3) & (tr_x[:, IX] > 280e3) & (tr_x[:, IX] < 380e3)
        tr_f[idxb, TR_RH0] = 3350
        tr_f[idxb, TR_MAT] = 2
        tr_f[idxb, TR_ET0] = 1e22

    elif choose_model == 3:
        # Rising block with free surface
        do_heatdiff = False
        tdep_rho = False
        tdep_eta = False
        tr_f[:, TR_RH0] = 3300
        tr_f[:, TR_MAT] = 1
        tr_f[:, TR_ET0] = 1e20
        idxb = (tr_x[:, IZ] > 400e3) & (tr_x[:, IZ] < 500e3) & (tr_x[:, IX] > 280e3) & (tr_x[:, IX] < 380e3)
        tr_f[idxb, TR_RH0] = 3280
        tr_f[idxb, TR_MAT] = 2
        tr_f[idxb, TR_ET0] = 1e22
        idxa = (tr_x[:, IZ] < 50e3)
        tr_f[idxa, TR_RH0] = 1000
        tr_f[idxa, TR_MAT] = 0
        tr_f[idxa, TR_ET0] = 1e18

    elif choose_model == 4:
        tdep_eta = True
        tdep_rho = True
        tr_f[:, TR_RH0] = 3300
        tr_f[:, TR_ALP] = 3.5e-5
        tr_f[:, TR_HCD] = 4
        tr_f[:, TR_HCP] = 1250
        tr_f[:, TR_TMP] = 273
        tr_f[:, TR_MAT] = 1
        idx = (tr_x[:, IZ] < 300e3) & (tr_x[:, IZ] > 200e3) & (tr_x[:, IX] > 200e3) & (tr_x[:, IX] < 600e3)
        tr_f[idx, TR_TMP] = 1623
        tr_f[idx, TR_HCD] = 15
        tr_f[idx, TR_RH0] = 2900
        tr_f[idx, TR_HCP] = 1000
        tr_f[idx, TR_MAT] = 0


    ## Boundary conditions
    bcstokes = [[]] * 4
    bcstokes[DIM*0 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
    bcstokes[DIM*1 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
    bcstokes[DIM*0 + IX] = pylamp_stokes.BC_TYPE_FREESLIP
    bcstokes[DIM*1 + IX] = pylamp_stokes.BC_TYPE_FREESLIP

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
    time_last_output = 0
    while ((it < max_it) and (totaltime < max_time)):
        it += 1
        pprint(" --- Time step:", it, "---")

        #if bc_internal_type > 0:

        #    if bc_internal_type == 1:
        #        # force material zero (water, air) to constant temperature
        #        idxmat = tr_f[:, TR_MAT] == 0
        #        tr_f[idxmat, TR_TMP] = 273
        #    elif bc_internal_type == 2:
        #        idx = (tr_x[:, IZ] < 300e3) & (tr_x[:, IZ] > 250e3) & (tr_x[:, IX] > 300e3) & (tr_x[:,IX] < 350e3)
        #        tr_f[idx, TR_TMP] = 273
        #    elif bc_internal_type == 3:
        #        idxmat = tr_f[:, TR_MAT] <= 1
        #        tr_f[idxmat, TR_TMP] = 273

        pprint("Calculate physical properties")
        if tdep_rho:
            # Effective density, rho=rho(T, inherent density)
            tr_f[:, TR_RHO] = ((tr_f[:, TR_ALP] * (tr_f[:, TR_TMP] - Tref) + 1) / tr_f[:, TR_RH0])**(-1)
        else:
            tr_f[:, TR_RHO] = tr_f[:, TR_RH0]

        if tdep_eta:
            # Effective viscosity, eta=eta(T, inherent viscosity)
            tr_f[:, TR_ETA] = tr_f[:, TR_ET0] * np.exp(tr_f[:, TR_ACE] / (GASR * tr_f[:, TR_TMP]) - tr_f[:, TR_ACE] / (GASR * Tref))
            tr_f[tr_f[:, TR_ETA] < etamin, TR_ETA] = etamin
            tr_f[tr_f[:, TR_ETA] > etamax, TR_ETA] = etamax
        else:
            tr_f[:, TR_ETA] = tr_f[:, TR_ET0]

        pprint("Properties trac2grid")

        if do_advect and do_heatdiff:
            # Interpolation done once on each different grid, multiple value fields at once
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_RHO, TR_ETA, TR_HCP, TR_TMP, TR_IHT, TR_MAT]], mesh, grid, [f_rho, f_etas, f_Cp, f_T, f_H, f_mat], nx, \
                    avgscheme=[pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_GEOMW, pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_ARITHW])
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_ETA]], meshmp, gridmp, [f_etan], nx, avgscheme=[pylamp_trac.INTERP_AVG_GEOMW])
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_HCD]], [meshmp[IZ], mesh[IX]], [gridmp[IZ], grid[IX]], [f_k[IZ]], nx, avgscheme=[pylamp_trac.INTERP_AVG_ARITHW])
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_HCD]], [mesh[IZ], meshmp[IX]], [grid[IZ], gridmp[IX]], [f_k[IX]], nx, avgscheme=[pylamp_trac.INTERP_AVG_ARITHW])


        elif do_advect:
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_RHO, TR_ETA]], mesh, grid, [f_rho, f_etas], nx, \
                    avgscheme=[pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_GEOMW])
            pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_ETA]], meshmp, gridmp, [f_etan], nx, avgscheme=[pylamp_trac.INTERP_AVG_GEOMETRIC])

        elif do_heatdiff:
            if it == 1 or tdep_rho or force_trac2grid_T:
                pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_RHO, TR_HCP, TR_TMP, TR_MAT]], mesh, grid, [f_rho, f_Cp, f_T, f_mat], nx, 
                        avgscheme=[pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_ARITHW, pylamp_trac.INTERP_AVG_ARITHW])
                pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_HCD]], [meshmp[IZ], mesh[IX]], [gridmp[IZ], grid[IX]], [f_k[IZ]], nx, avgscheme=[pylamp_trac.INTERP_AVG_ARITHW])
                pylamp_trac.trac2grid(tr_x, tr_f[:,[TR_HCD]], [mesh[IZ], meshmp[IX]], [grid[IZ], gridmp[IX]], [f_k[IX]], nx, avgscheme=[pylamp_trac.INTERP_AVG_ARITHW])

            else:
                ### after the first time step (if no temperature dependent rho) we only need to interpolate temperature, since there is no advection
                ### actually, let's skip that, too, and copy the grid directly
                f_T = np.copy(newtemp)

        if do_heatdiff and it > 1:
            f_T[:, 0] = newtemp[:, 0]
            f_T[:, -1] = newtemp[:, -1]
            f_T[0, :] = newtemp[0, :]
            f_T[-1, :] = newtemp[-1, :]

        if do_heatdiff:
            diffusivity = f_k[IZ] / (f_rho * f_Cp)
            tstep_temp = tstep_modifier * np.min(dx)**2 / np.max(2*diffusivity)
            tstep_temp = min(tstep_temp, tstep_dif_max)
            tstep_temp = max(tstep_temp, tstep_dif_min)

        newvel = 0
        newpres = 0
        tstep_limiter = ""

        if do_stokes:
            pprint("Build stokes")

            if surface_stabilization == False or surfstab_tstep < 0:
                (A, rhs) = pylamp_stokes.makeStokesMatrix(nx, grid, f_etas, f_etan, f_rho, bcstokes, surfstab=False)
            else:
                (A, rhs) = pylamp_stokes.makeStokesMatrix(nx, grid, f_etas, f_etan, f_rho, bcstokes, surfstab=True, tstep=surfstab_tstep, surfstab_theta=surfstab_theta)

            pprint("Solve stokes")
            # Solve it!
            #x = scipy.sparse.linalg.bicgstab(scipy.sparse.csc_matrix(A), rhs)[0]
            x = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), rhs)

            (newvel, newpres) = pylamp_stokes.x2vp(x, nx)

            tstep_stokes = tstep_modifier * np.min(dx) / np.max(newvel)
            tstep_stokes = min(tstep_stokes, tstep_adv_max)
            tstep_stokes = max(tstep_stokes, tstep_adv_min)

            if surfstab_tstep > 0:
                if tstep_stokes < surfstab_tstep:
                    pprint ("WARNING: tstep_stokes " + str(tstep_stokes/SECINKYR) + " kyrs < surfstab_tstep " + str(surfstab_tstep/SECINKYR) + " kyrs")
                    pprint ("         using surfstab_tstep")
                tstep_stokes = surfstab_tstep

        if do_heatdiff and do_advect:
            if tstep_temp < tstep_stokes:
                tstep_limiter = "H"
            else:
                tstep_limiter = "S"
            tstep = min(tstep_temp, tstep_stokes)
        elif do_heatdiff:
            tstep = tstep_temp
            tstep_limiter = "H"
        else:
            tstep = tstep_stokes
            tstep_limiter = "S"

        if do_stokes and surface_stabilization and surfstab_tstep < 0:
            stabRedoDone = False
            while not stabRedoDone:
                pprint ("Redo stokes with surface stabilization")
                (A, rhs) = pylamp_stokes.makeStokesMatrix(nx, grid, f_etas, f_etan, f_rho, bcstokes, surfstab=True, tstep=tstep, surfstab_theta=surfstab_theta)

                print ("Resolve stokes")
                x = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), rhs)
                #(x, Aerr) = scipy.sparse.linalg.bicgstab(scipy.sparse.csc_matrix(A), rhs, x0=x)
                #print ("  resolve error: ", Aerr)
                (newvel, newpres) = pylamp_stokes.x2vp(x, nx)

                check_tstep_stokes = tstep_modifier * np.min(dx) / np.max(newvel)
                if check_tstep_stokes < tstep:
                    pprint ("WARNING: tstep after surfstab is less than actual used:", check_tstep_stokes/SECINKYR, tstep/SECINKYR)
                    tstep = check_tstep_stokes
                    tstep_limiter = "Ss"
                else:
                    stabRedoDone = True

        totaltime += tstep
        pprint("   time step =", tstep/SECINKYR, "kyrs", tstep_limiter)
        pprint("   time now  =", totaltime/SECINKYR, "kyrs")


        if do_heatdiff:
            pprint("Build heatdiff")

            (A, rhs) = pylamp_diff.makeDiffusionMatrix(nx, grid, gridmp, f_T, f_k, f_Cp, f_rho, f_H, bcheat, bcheatvals, tstep)

            pprint("Solve diffusion")

            x = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), rhs)

            newtemp = pylamp_diff.x2t(x, nx)

            #if bc_internal_type == 1:
            #    mat_air = 0
            #    mat_sfc = np.min(tr_f[tr_f[:,TR_MAT] > 0, TR_MAT])
            #    if it == 1:
            #        print ("mat surface:", str(mat_sfc))
            #    idx = f_mat < (mat_air + mat_sfc) / 3
            #    newtemp[idx] = 273
            #elif bc_internal_type == 2:
            #    idx = (mesh[IX] < 300e3) & (mesh[IX] > 200e3) & (mesh[IZ] < 200e3) & (mesh[IZ] > 100e3)
            #    newtemp[idx] = 273
            #else:
            #    pass

            old_tr_f = np.array(tr_f, copy=True)

            l_interp_tracvals = np.zeros((tr_f.shape[0], 1))
            g_interp_tracvals = np.zeros((tr_f.shape[0], 1))

            if it == 1:
                # On first timestep interpolate absolute temperature values to tracers ...
                pprint("grid2trac T")
                pylamp_trac.grid2trac(tr_x[IPROC::NPROC], l_interp_tracvals[IPROC::NPROC], grid, [newtemp], nx, method=pylamp_trac.INTERP_METHOD_LINEAR, stopOnError=True)
                MPICOMM.Allreduce([l_interp_tracvals, MPI.DOUBLE], [g_interp_tracvals, MPI.DOUBLE], op=MPI.SUM)
                tr_f[:, TR_TMP] = g_interp_tracvals[:, 0]
            else:
                # ... on subsequent timesteps interpolate only the change to avoid numerical diffusion
                pprint("grid2trac dT")
                newdT = newtemp - f_T
                pylamp_trac.grid2trac(tr_x[IPROC::NPROC], l_interp_tracvals[IPROC::NPROC], grid, [newdT], nx, method=pylamp_trac.INTERP_METHOD_LINEAR, stopOnError=True)
                MPICOMM.Allreduce([l_interp_tracvals, MPI.DOUBLE], [g_interp_tracvals, MPI.DOUBLE], op=MPI.SUM)
                tr_f[:, TR_TMP] = tr_f[:, TR_TMP] + g_interp_tracvals[:, 0]

                #if bc_internal_type == 1:
                #    # force material zero (water, air) to constant temperature
                #    idxmat = tr_f[:, TR_MAT] == 0
                #    tr_f[idxmat, TR_TMP] = 273
                #elif bc_internal_type == 2:
                #    idx = (tr_x[:, IZ] < 300e3) & (tr_x[:, IZ] > 250e3) & (tr_x[:, IX] > 300e3) & (tr_x[:,IX] < 350e3)
                #    tr_f[idx, TR_TMP] = 273
                #elif bc_internal_type == 3:
                #    idxmat = tr_f[:, TR_MAT] <= 1
                #    tr_f[idxmat, TR_TMP] = 273
                
                ## subgrid diffusion
                # NB! Assumes regular grid
                # correction at tracers
                d = 0.5
                subgrid_corr_dt0 = tr_f[:, TR_HCP] * tr_f[:, TR_RHO] / (tr_f[:, TR_HCD] * ((2/dx[IX])**2 + (2/dx[IZ])**2))
                subgrid_corr_T = old_tr_f[:, TR_TMP] - (old_tr_f[:, TR_TMP] - tr_f[:, TR_TMP]) * np.exp(-d * tstep / subgrid_corr_dt0)
                # compensation at eulerian nodes
                dT_subgrid_diff = subgrid_corr_T - tr_f[:, TR_TMP]
                dT_subgrid_diff_back = np.zeros_like(dT_subgrid_diff)
                pylamp_trac.trac2grid(tr_x, dT_subgrid_diff[:, None], mesh, grid, [f_sgc], nx, avgscheme=[pylamp_trac.INTERP_AVG_ARITHW])
                pylamp_trac.grid2trac(tr_x, dT_subgrid_diff_back[:,None], grid, [f_sgc], nx, method=pylamp_trac.INTERP_METHOD_LINEAR, stopOnError=True)
                tr_f[:, TR_TMP] = subgrid_corr_T - dT_subgrid_diff_back

        if do_advect:
            pprint("Tracer advection")

            # For grid2trac called in RK() the grid needs to span outside
            # every tracer (so that we can INTERpolate). Here we first
            # make sure that this is the case by creating temporary
            # grid nodes using BC values for velocity
            velsmp = [[]] * DIM
            velsmp[IZ] = 0.5 * (newvel[IZ][1:,:-1] + newvel[IZ][:-1,:-1])
            velsmp[IX] = 0.5 * (newvel[IX][:-1,1:] + newvel[IX][:-1,:-1])
            preval = [gridmp[d][0] - (gridmp[d][1] - gridmp[d][0]) for d in range(DIM)]
            newgridx = np.insert(gridmp[IX], 0, preval[IX])
            newgridz = np.insert(gridmp[IZ], 0, preval[IZ])

            vels = [ \
                    np.hstack(( np.zeros(nx[IZ]+1)[:,None], np.vstack((np.zeros(nx[IX]-1)[None,:], velsmp[IZ], np.zeros(nx[IX]-1)[None,:])), np.zeros(nx[IZ]+1)[:,None] )), \
                    np.hstack(( np.zeros(nx[IZ]+1)[:,None], np.vstack((np.zeros(nx[IX]-1)[None,:], velsmp[IX], np.zeros(nx[IX]-1)[None,:])), np.zeros(nx[IZ]+1)[:,None] ))  \
                   ]

            if bcstokes[DIM*0 + IZ] == pylamp_stokes.BC_TYPE_NOSLIP:
                vels[IX][0,:] = -vels[IX][1,:]  # i.e. vel is zero at bnd
                vels[IZ][0,:] = -vels[IZ][1,:]      # i.e. vel out/in of bnd is zero (no in/outflow)
            elif bcstokes[DIM*0 + IZ] == pylamp_stokes.BC_TYPE_FREESLIP:
                vels[IX][0,:] = vels[IX][1,:]   # i.e. vel change is zero across bnd
                vels[IZ][0,:] = -vels[IZ][1,:]      # i.e. vel out/in of bnd is zero (no in/outflow)
            elif bcstokes[DIM*0 + IZ] == pylamp_stokes.BC_TYPE_CYCLIC:
                vels[IX][0,:] = vels[IX][-2,:]  
                vels[IZ][0,:] = vels[IZ][-2,:]

            if bcstokes[DIM*0 + IX] == pylamp_stokes.BC_TYPE_NOSLIP:
                vels[IZ][:,0] = -vels[IZ][:,1]
                vels[IX][:,0] = -vels[IX][:,1]
            elif bcstokes[DIM*0 + IX] == pylamp_stokes.BC_TYPE_FREESLIP:
                vels[IZ][:,0] = vels[IZ][:,1]
                vels[IX][:,0] = -vels[IX][:,1]
            elif bcstokes[DIM*0 + IX] == pylamp_stokes.BC_TYPE_CYCLIC:
                vels[IZ][:,0] = vels[IZ][:,-2]
                vels[IX][:,0] = vels[IX][:,-2]

            if bcstokes[DIM*1 + IZ] == pylamp_stokes.BC_TYPE_NOSLIP:
                vels[IX][-1,:] = -vels[IX][-2,:]
                vels[IZ][-1,:] = -vels[IZ][-2,:]
            elif bcstokes[DIM*1 + IZ] == pylamp_stokes.BC_TYPE_FREESLIP:
                vels[IX][-1,:] = vels[IX][-2,:]
                vels[IZ][-1,:] = -vels[IZ][-2,:]
            elif bcstokes[DIM*1 + IZ] == pylamp_stokes.BC_TYPE_CYCLIC:
                vels[IX][-1,:] = vels[IX][1,:]
                vels[IZ][-1,:] = vels[IZ][1,:]

            if bcstokes[DIM*1 + IX] == pylamp_stokes.BC_TYPE_NOSLIP:
                vels[IZ][:,-1] = -vels[IZ][:,-2]
                vels[IX][:,-1] = -vels[IX][:,-2]
            elif bcstokes[DIM*1 + IX] == pylamp_stokes.BC_TYPE_FREESLIP:
                vels[IZ][:,-1] = vels[IZ][:,-2]
                vels[IX][:,-1] = -vels[IX][:,-2]
            elif bcstokes[DIM*1 + IX] == pylamp_stokes.BC_TYPE_CYCLIC:
                vels[IZ][:,-1] = vels[IZ][:,1]
                vels[IX][:,-1] = vels[IX][:,1]


            MPICOMM.Allreduce([l_interp_tracvals, MPI.DOUBLE], [g_interp_tracvals, MPI.DOUBLE], op=MPI.SUM)

            trac_vel = np.zeros((ntrac, DIM))
            l_tr_x = np.zeros_like(tr_x)
            l_trac_vel = np.zeros((ntrac, DIM))
            ls_trac_vel, ls_tr_x = pylamp_trac.RK(tr_x[IPROC::NPROC], [newgridz, newgridx], vels, nx, tstep)
            l_tr_x[IPROC::NPROC,:] = ls_tr_x[:,:]
            l_trac_vel[IPROC::NPROC,:] = ls_trac_vel[:,:]

            MPICOMM.Allreduce([l_tr_x, MPI.DOUBLE], [tr_x, MPI.DOUBLE], op=MPI.SUM)
            MPICOMM.Allreduce([l_trac_vel, MPI.DOUBLE], [trac_vel, MPI.DOUBLE], op=MPI.SUM)

            # do not allow tracers to advect outside the domain
            for d in range(DIM):
                if bcstokes[d*0 + IX] == pylamp_stokes.BC_TYPE_CYCLIC :
                    tr_x[tr_x[:,d] <= 0, d] += L[d]
                    tr_x[tr_x[:,d] >= L[d], d] -= L[d]
                tr_x[tr_x[:,d] <= 0, d] = EPS
                tr_x[tr_x[:,d] >= L[d], d] = L[d]-EPS


        if IPROC == 0 and output_numpy and ((output_stride > 0 and (it-1) % output_stride == 0) or (output_stride < 0 and (totaltime - time_last_output)/SECINMYR > output_stride_ma)):
            if output_stride > 0:
                time_last_output = totaltime
            else:
                time_last_output = SECINMYR * output_stride_ma * float(int(totaltime/(SECINMYR*output_stride_ma)))
            if do_stokes:
                if do_heatdiff:
                    np.savez(output_outdir + "/griddata.{:06d}.npz".format(it), gridz=grid[IZ], gridx=grid[IX], velz=newvel[IZ], velx=newvel[IX], pres=newpres, rho=f_rho, temp=newtemp, tstep=it, time=totaltime)
                else:
                    np.savez(output_outdir + "/griddata.{:06d}.npz".format(it), gridz=grid[IZ], gridx=grid[IX], velz=newvel[IZ], velx=newvel[IX], pres=newpres, rho=f_rho, temp=newvel[IX]*0.0, tstep=it, time=totaltime)
                np.savez(output_outdir + "/tracs.{:06d}.npz".format(it), tr_x=tr_x, tr_f=tr_f, tr_v=trac_vel, tstep=it, time=totaltime)
            else:
                np.savez(output_outdir + "/griddata.{:06d}.npz".format(it), gridz=grid[IZ], gridx=grid[IX], rho=f_rho, temp=newtemp, tstep=it, time=totaltime)
                np.savez(output_outdir + "/tracs.{:06d}.npz".format(it), tr_x=tr_x, tr_f=tr_f, tstep=it, time=totaltime)


    if do_profiling:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        print(s.getvalue())

    sys.exit()




### Discretization of the equations:

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
