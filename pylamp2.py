#!/usr/bin/python3

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
import os

### PyLamp
#
# Python code to solve the conservation of energy, momentum and mass
# incompressible viscous flow
#
# - implicit (scipy direct solver)
# - marker-in-cell for material and temperature advection
# - linear (Newtonian) viscosity
# - temperature dependent viscosity and density (buoyancy)
#
#

#### MAIN ####
if __name__ == "__main__":
    ###################################
    ### SELECT MODEL SET-UP VERSION ###
    ###################################
    choose_model = 'stagnant lid'
    # possible values:
    #  - thick crust
    #  - falling block
    #  - rifting
    #  - slab
    #  - lavalamp
    #  - graphite
    #  - stagnant lid
    ###################################

    model_parameters = [ 0 ]


    #######################################
    ### Grid settings                   ###
    #######################################

    if choose_model == 'thick crust':
        nx    =   [33,45]         # use order z,x
        L     =   [660e3, 1800e3] 
    elif choose_model == 'rifting':
        nx    =   [33,45]         # use order z,x
        L     =   [330e3, 900e3] 
    elif choose_model == 'falling block':
        nx    =   [33,33]         # use order z,x
        L     =   [660e3, 660e3] 
    elif choose_model == 'slab':
        nx    =   [33,45]         # use order z,x
        L     =   [660e3, 1800e3] 
    elif choose_model == 'lavalamp':
        nx    =   [33,33]         # use order z,x
        L     =   [660e3, 660e3] 
    elif choose_model == 'rifting with temp':
        nx    =   [33,45]         # use order z,x
        L     =   [330e3, 900e3] 
    elif choose_model == 'graphite':
        nx    =   [31, 5]
        L     =   [30e3, 4e3]
    elif choose_model == 'stagnant lid':
        nx    =   [34, 34]
        L     =   [660e3, 660e3]
    else:
        raise Exception("Invalid model name '" + choose_model + "'")

    #######################################
    ### Default configuration           ###
    #######################################

    tracdens = 50     # how many tracers per element on average
    tracdens_min = 30 # minimum number of tracers per element
    tracs_fence_enabled = True # stop tracers at the boundary
                                # if they are about to flow out

    do_stokes = False
    do_advect = False
    do_heatdiff = False
    do_subgrid_heatdiff = False
    subgrid_corr_coefficient = 0.5

    tstep_adv_max = 50e9 * SECINYR
    tstep_adv_min = 50e-9 * SECINYR
    tstep_dif_max = 50e9 * SECINYR
    tstep_dif_min = 50e-9 * SECINYR
    tstep_modifier = 0.67             # coefficient for automatic tsteps

    output_numpy = False
    output_stride = 1
    output_stride_ma = 1            # used if output_stride < 0: output fields every x million years
    output_outdir = "out"

    tdep_rho = False
    tdep_eta = False
    etamin = 1e17
    etamax = 1e23
    Tref = 1623
    
    force_trac2grid_T = False       # force tracer to grid interpolation even in the case when there is no advection
    max_it = 100
    max_time = SECINMYR * 5000
    bc_internal_type = 0           # 0 = disabled
                                   # 1 = keep material zero at constant temperature T=273K
    surface_stabilization = False  # use if "sticky air" free surface present
    surfstab_theta = 0.5
    surfstab_tstep = -1 #1*SECINKYR            # if negative, a dynamic tstep is used 

    do_profiling = False

    event_codes = []
    event_enable_times = []
    event_disable_times = []
    event_counts = []

    #######################################
    ### General setup of things         ###
    #######################################

    # Profiling
    if do_profiling:
        pr = cProfile.Profile()
        pr.enable()

    # Derived options
    # dx for regular grid
    dx    =   [L[i]/(nx[i]-1) for i in range(DIM)]
    pprint(dx)

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

    # Containers for BCs
    bcstokes = [[]] * 4
    bcstokesvals = [None] * 4
    bcheat = [[]] * 4
    bcheatvals = [[]] * 4

    # Tracers
    ntrac = np.prod(nx)*tracdens

    tr_x = np.random.rand(ntrac, DIM)  # tracer coordinates

    tr_x = np.multiply(tr_x, L)
    tr_f = np.zeros((ntrac, NFTRAC))     # tracer functions (values)

    tr_f[:, TR__ID] = np.arange(0, ntrac)



    #############################################
    ### Define (non-default) model paramaters ###
    #############################################

    if choose_model == 'lavalamp':

        do_stokes = True
        do_advect = True       # Calculate Stokes and material advection?

        do_heatdiff = True     # Calculate heat equation?

        tdep_rho = True        # temperature dependent density?
        tdep_eta = True        # temperature dependent viscosity?

        etamin = 1e17          # global minimum viscosity
        etamax = 1e24          # global maximum viscosity

        max_it = 100000               # maximum number of time steps to take
        max_time = SECINMYR * 5000    # maximum model time to run
        surface_stabilization = False # switch to True if an air/water layer is present

        # Material parameters:

        tr_f[:, TR_RH0] = 3300        # inherent density
        tr_f[:, TR_ALP] = 3.5e-5      # coefficient of thermal expansion
        tr_f[:, TR_MAT] = 1           # material number (for plotting)
        tr_f[:, TR_ET0] = 1e19        # inherent viscosity
        tr_f[:, TR_HCD] = 4.0         # heat conductivity
        tr_f[:, TR_HCP] = 1250        # heat capacity
        tr_f[:, TR_TMP] = 1573        # temperature (initial)
        tr_f[:, TR_ACE] = 120e3       # activation energy
        tr_f[:, TR_IHT] = 0.0         # internal heating rate

        idx_blobs = tr_x[:, IZ] > 500e3
        tr_f[idx_blobs, TR_RH0] = 3300
        tr_f[idx_blobs, TR_ALP] = 3.5e-5
        tr_f[idx_blobs, TR_MAT] = 2
        tr_f[idx_blobs, TR_ET0] = 1e21
        tr_f[idx_blobs, TR_HCD] = 4.0
        tr_f[idx_blobs, TR_HCP] = 1250
        tr_f[idx_blobs, TR_TMP] = 1623
        tr_f[idx_blobs, TR_ACE] = 120e3
        tr_f[idx_blobs, TR_IHT] = 0.0 
        
        # Boundary conditions for Stokes
        # Possible values:
        #   pylamp_stokes.BC_TYPE_FREESLIP
        #   pylamp_stokes.BC_TYPE_NOSLIP
        bcstokes[DIM*0 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP   # Upper bnd
        bcstokes[DIM*1 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP   # Lower bnd
        bcstokes[DIM*0 + IX] = pylamp_stokes.BC_TYPE_FREESLIP   # Left bnd
        bcstokes[DIM*1 + IX] = pylamp_stokes.BC_TYPE_FREESLIP   # Right bnd
        
        # Boundary condition types for heat equation
        # Possible values:
        #   pylamp_diff.BC_TYPE_FIXTEMP
        #   pylamp_diff.BC_TYPE_FIXFLOW
        bcheat[DIM*0 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP  # Upper
        bcheat[DIM*1 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP  # Lower
        bcheat[DIM*0 + IX] = pylamp_diff.BC_TYPE_FIXFLOW  # Left 
        bcheat[DIM*1 + IX] = pylamp_diff.BC_TYPE_FIXFLOW  # Right

        # B.C. values for heat eq.
        bcheatvals[DIM*0 + IZ] = 273    # Upper bnd, i.e. surface
        bcheatvals[DIM*1 + IZ] = 1623   # Lower bnd
        bcheatvals[DIM*0 + IX] = 0      # Left bnd
        bcheatvals[DIM*1 + IX] = 0      # Right bnd

        output_numpy = True       # Write output?
        output_stride = -1        # Write output every nth time step
        output_stride_ma = 0.5    # Write output every x million years,
                                  # used if output_stride < 0
        output_outdir = "lavalamp"  # To which folder output is written

    elif choose_model == 'thick crust':

        do_stokes = True
        do_advect = True
        do_heatdiff = True

        tdep_rho = True
        tdep_eta = True

        etamin = 1e17
        etamax = 1e23
        Tref = 1623
        
        max_it = 100000
        max_time = SECINMYR * 5000
        surface_stabilization = True
        surfstab_theta = 0.5
        surfstab_tstep = -1 #1*SECINKYR    # if negative, a dynamic tstep is used 

        zair = tr_x[:,IZ] < 50e3
        extraidx = (tr_x[:, IZ] > 100e3) & (tr_x[:, IZ] < 150e3) & (tr_x[:, IX] < 600e3) & (tr_x[:, IX] > 200e3)
        zcrust = (tr_x[:,IZ] < 100e3) | extraidx
        tstep_modifier = 0.33

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

        bcstokes[DIM*0 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*1 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*0 + IX] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*1 + IX] = pylamp_stokes.BC_TYPE_FREESLIP 
        
        bcheat[DIM*0 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP
        bcheat[DIM*1 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP
        bcheat[DIM*0 + IX] = pylamp_diff.BC_TYPE_FIXFLOW
        bcheat[DIM*1 + IX] = pylamp_diff.BC_TYPE_FIXFLOW

        bcheatvals[DIM*0 + IZ] = 273
        bcheatvals[DIM*1 + IZ] = 1623
        bcheatvals[DIM*0 + IX] = 0
        bcheatvals[DIM*1 + IX] = 0

        bc_internal_type = 1

    elif choose_model == 'falling block':
        # Falling block
        do_stokes = True
        do_advect = True       # Calculate Stokes and material advection?
 
        do_heatdiff = False     # Calculate heat equation?

        tdep_rho = False        # temperature dependent density?
        tdep_eta = False        # temperature dependent viscosity?

        etamin = 1e17          # global minimum viscosity
        etamax = 1e24          # global maximum viscosity
        
        max_it = 100               # maximum number of time steps to take
        max_time = SECINMYR * 5000 # maximum model time to run

        ### Material properties
        tr_f[:, TR_RH0] = 3300
        tr_f[:, TR_MAT] = 1
        tr_f[:, TR_ET0] = 1e19
        idxb = (tr_x[:, IZ] > 200e3) & (tr_x[:, IZ] < 300e3) & (tr_x[:, IX] > 280e3) & (tr_x[:, IX] < 380e3)
        tr_f[idxb, TR_RH0] = 3350
        tr_f[idxb, TR_MAT] = 2
        tr_f[idxb, TR_ET0] = 1e22

        bcstokes[DIM*0 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*1 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*0 + IX] = pylamp_stokes.BC_TYPE_FREESLIP + pylamp_stokes.BC_TYPE_FLOWTHRU
        bcstokes[DIM*1 + IX] = pylamp_stokes.BC_TYPE_FREESLIP

        surface_stabilization = False
        
        bcstokesvals[DIM*0 + IX] = np.empty(nx[IZ])
        bcstokesvals[DIM*0 + IX][:] = np.nan

        output_numpy = True
        output_stride = 1
        output_stride_ma = 0.1            # used if output_stride < 0: output fields every x million years
        output_outdir = "out_falling_block"

    elif choose_model == 'rifting with temp':

        do_stokes = True
        do_advect = True
        do_heatdiff = True

        tdep_rho = False
        tdep_eta = False

        etamin = 1e17
        etamax = 1e24
        Tref = 1623
        
        max_it = 100000
        max_time = SECINMYR * 5000
        surface_stabilization = True
        surfstab_theta = 0.5
        surfstab_tstep = -1 #1*SECINKYR    # if negative, a dynamic tstep is used 

        h_crust = 30e3
        h_lmantle = 60e3
        h_litho = h_crust + h_lmantle
        h_air = 45e3

        zair = tr_x[:,IZ] <= h_air
        zcrust = (tr_x[:,IZ] <= h_air + h_crust) & (tr_x[:,IZ] > h_air)
        zlmantle = (tr_x[:, IZ] <= h_air + h_litho) & (tr_x[:,IZ] > h_air + h_crust)
        zseed = (tr_x[:,IZ] < h_air + 1.33*h_crust) & (tr_x[:,IZ] > h_air + 0.33*h_crust)
        xseed = (tr_x[:,IX] > 0.5*L[IX] - 15e3) & (tr_x[:,IX] < 0.5*L[IX] + 15e3)
        zxseed = zseed & xseed

        tstep_modifier = 0.50

        tr_f[:, TR_RH0] = 3300
        tr_f[:, TR_MAT] = 1
        tr_f[:, TR_ET0] = 1e20
        tr_f[:, TR_ALP] = 3.5e-5
        tr_f[:, TR_HCD] = 4.0
        tr_f[:, TR_HCP] = 1250
        tr_f[:, TR_TMP] = 1623
        tr_f[:, TR_ACE] = 120e3
        tr_f[:, TR_IHT] = 0.0
        tr_f[zair, TR_RH0] = 1000
        tr_f[zair, TR_ET0] = 1e20
        tr_f[zair, TR_MAT] = 0
        tr_f[zair, TR_ALP] = 0
        tr_f[zair, TR_HCD] = 4.0
        tr_f[zair, TR_HCP] = 1250
        tr_f[zair, TR_TMP] = 273
        tr_f[zair, TR_ACE] = 0.0
        tr_f[zair, TR_IHT] = 0.0
        tr_f[zcrust, TR_ALP] = 3.5e-5
        tr_f[zcrust, TR_HCD] = 2.5
        tr_f[zcrust, TR_HCP] = 1250
        tr_f[zcrust, TR_TMP] = (600 + 273) * (tr_x[zcrust,IZ] - h_air) / (h_crust)
        tr_f[zcrust, TR_ACE] = 120e3
        tr_f[zcrust, TR_MAT] = 2
        tr_f[zcrust, TR_ET0] = 1e23
        tr_f[zcrust, TR_IHT] = 0.0
        tr_f[zlmantle, TR_RH0] = 3200
        tr_f[zlmantle, TR_ET0] = 1e21
        tr_f[zlmantle, TR_MAT] = 3
        tr_f[zlmantle, TR_TMP] = (1350 + 273) * (tr_x[zlmantle,IZ] - h_air) / (h_crust + h_lmantle)
        tr_f[zxseed, TR_RH0] = 2900
        tr_f[zxseed, TR_ET0] = 1e20
        tr_f[zxseed, TR_MAT] = 4

        bcstokes[DIM*0 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*1 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*0 + IX] = pylamp_stokes.BC_TYPE_FREESLIP + pylamp_stokes.BC_TYPE_FLOWTHRU
        bcstokes[DIM*1 + IX] = pylamp_stokes.BC_TYPE_FREESLIP + pylamp_stokes.BC_TYPE_FLOWTHRU
        
        bcstokesvals[DIM*0 + IX] = np.empty(nx[IZ])
        # bcstokesvals[DIM*1 + IX][:] = np.nan  # every point is stress free
        bcstokesvals[DIM*0 + IX][grid[IZ] < h_air] = 0.0
        bcstokesvals[DIM*0 + IX][(grid[IZ] > h_air) & (grid[IZ] <= h_air + h_litho)] = -0.1 / SECINYR
        bcstokesvals[DIM*0 + IX][grid[IZ] > h_air + h_litho] = (h_crust * 0.1 / SECINYR) / (L[IZ]-(h_air+h_litho))

        bcstokesvals[DIM*1 + IX] = np.empty(nx[IZ])
        # bcstokesvals[DIM*1 + IX][:] = np.nan  # every point is stress free
        bcstokesvals[DIM*1 + IX][grid[IZ] <= h_air] = 0.0
        bcstokesvals[DIM*1 + IX][(grid[IZ] > h_air) & (grid[IZ] <= h_air + h_litho)] = 0.1 / SECINYR
        bcstokesvals[DIM*1 + IX][grid[IZ] > h_air + h_litho] = -(h_crust * 0.1 / SECINYR) / (L[IZ]-(h_air+h_litho))
        # Boundary condition types for heat equation
        # Possible values:
        #   pylamp_diff.BC_TYPE_FIXTEMP
        #   pylamp_diff.BC_TYPE_FIXFLOW
        bcheat[DIM*0 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP  # Upper
        bcheat[DIM*1 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP  # Lower
        bcheat[DIM*0 + IX] = pylamp_diff.BC_TYPE_FIXFLOW  # Left 
        bcheat[DIM*1 + IX] = pylamp_diff.BC_TYPE_FIXFLOW  # Right

        # B.C. values for heat eq.
        bcheatvals[DIM*0 + IZ] = 273    # Upper bnd, i.e. surface
        bcheatvals[DIM*1 + IZ] = 1623   # Lower bnd
        bcheatvals[DIM*0 + IX] = 0      # Left bnd
        bcheatvals[DIM*1 + IX] = 0      # Right bnd

        bc_internal_type = 1

        output_numpy = True
        output_stride = 1
        output_stride_ma = 0.1            # used if output_stride < 0: output fields every x million years
        output_outdir = "out_rift_with_temp"

    elif choose_model == 'rifting':

        do_stokes = True
        do_advect = True
        do_heatdiff = False

        tdep_rho = False
        tdep_eta = False

        etamin = 1e17
        etamax = 1e24
        Tref = 1623
        
        max_it = 100000
        max_time = SECINMYR * 5000
        surface_stabilization = True
        surfstab_theta = 0.5
        surfstab_tstep = -1 #1*SECINKYR    # if negative, a dynamic tstep is used 

        h_crust = 30e3
        h_lmantle = 60e3
        h_litho = h_crust + h_lmantle
        h_air = 45e3

        zair = tr_x[:,IZ] <= h_air
        zcrust = (tr_x[:,IZ] <= h_air + h_crust) & (tr_x[:,IZ] > h_air)
        zlmantle = (tr_x[:, IZ] <= h_air + h_litho) & (tr_x[:,IZ] > h_air + h_crust)
        zseed = (tr_x[:,IZ] < h_air + 1.5*h_crust) & (tr_x[:,IZ] > h_air + 0.33*h_crust)
        xseed = (tr_x[:,IX] > 0.5*L[IX] - 15e3) & (tr_x[:,IX] < 0.5*L[IX] + 15e3)
        zxseed = zseed & xseed

        tstep_modifier = 0.33

        tr_f[:, TR_RH0] = 3300
        tr_f[:, TR_MAT] = 2
        tr_f[:, TR_ET0] = 1e20
        tr_f[zair, TR_RH0] = 1000
        tr_f[zair, TR_ET0] = 1e18
        tr_f[zair, TR_MAT] = 0
        tr_f[zcrust, TR_RH0] = 2900
        tr_f[zcrust, TR_MAT] = 1
        tr_f[zcrust, TR_ET0] = 1e22
        tr_f[zlmantle, TR_RH0] = 3200
        tr_f[zlmantle, TR_ET0] = 1e21
        tr_f[zlmantle, TR_MAT] = 3
        tr_f[zxseed, TR_RH0] = 2900
        tr_f[zxseed, TR_ET0] = 1e19
        tr_f[zxseed, TR_MAT] = 4

        bcstokes[DIM*0 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*1 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*0 + IX] = pylamp_stokes.BC_TYPE_FREESLIP + pylamp_stokes.BC_TYPE_FLOWTHRU
        bcstokes[DIM*1 + IX] = pylamp_stokes.BC_TYPE_FREESLIP + pylamp_stokes.BC_TYPE_FLOWTHRU
        
        bcstokesvals[DIM*0 + IX] = np.empty(nx[IZ])
        # bcstokesvals[DIM*1 + IX][:] = np.nan  # every point is stress free
        bcstokesvals[DIM*0 + IX][grid[IZ] < h_air] = 0.0
        bcstokesvals[DIM*0 + IX][(grid[IZ] > h_air) & (grid[IZ] <= h_air + h_litho)] = -0.1 / SECINYR
        bcstokesvals[DIM*0 + IX][grid[IZ] > h_air + h_litho] = (h_crust * 0.1 / SECINYR) / (L[IZ]-(h_air+h_litho))

        bcstokesvals[DIM*1 + IX] = np.empty(nx[IZ])
        # bcstokesvals[DIM*1 + IX][:] = np.nan  # every point is stress free
        bcstokesvals[DIM*1 + IX][grid[IZ] <= h_air] = 0.0
        bcstokesvals[DIM*1 + IX][(grid[IZ] > h_air) & (grid[IZ] <= h_air + h_litho)] = 0.1 / SECINYR
        bcstokesvals[DIM*1 + IX][grid[IZ] > h_air + h_litho] = -(h_crust * 0.1 / SECINYR) / (L[IZ]-(h_air+h_litho))

        output_numpy = True
        output_stride = -1
        output_stride_ma = 0.1            # used if output_stride < 0: output fields every x million years
        output_outdir = "out_rift"

    elif choose_model == 'slab':

        do_stokes = True
        do_advect = True
        do_heatdiff = True

        tdep_rho = True
        tdep_eta = True

        etamin = 1e17
        etamax = 1e24
        Tref = 1623
        
        max_it = 100000
        max_time = SECINMYR * 5000
        surface_stabilization = True
        surfstab_theta = 0.5
        surfstab_tstep = -1 #1*SECINKYR    # if negative, a dynamic tstep is used 

        h_litho = 100e3
        h_air = 45e3

        zair = tr_x[:,IZ] <= h_air
        zlitho = (tr_x[:,IZ] <= h_air + h_litho) & (tr_x[:,IZ] > h_air)
        zxslab = (tr_x[:,IZ] <= h_air + h_litho + 50e3) & (tr_x[:,IZ] > h_air) & (tr_x[:,IX] <= 0.5*L[IX]+h_litho/2) & (tr_x[:,IX] > 0.5*L[IX]-h_litho/2)
        zxdetach = (tr_x[:,IX] < 0.5*L[IX]-h_litho/2) & (tr_x[:,IX] > 0.5*L[IX]-h_litho/2-20e3) & zlitho

        tstep_modifier = 0.33

        tr_f[:, TR_RH0] = 3300
        tr_f[:, TR_ALP] = 3.5e-5
        tr_f[:, TR_MAT] = 1
        tr_f[:, TR_ET0] = 1e20
        tr_f[:, TR_HCD] = 4.0
        tr_f[:, TR_HCP] = 1250
        tr_f[:, TR_TMP] = 1623
        tr_f[:, TR_ACE] = 120e3
        tr_f[:, TR_IHT] = 0.02e-6 

        tr_f[zair, TR_RH0] = 1000
        tr_f[zair, TR_ALP] = 0
        tr_f[zair, TR_MAT] = 0
        tr_f[zair, TR_ET0] = 1e18
        tr_f[zair, TR_HCD] = 4.0
        tr_f[zair, TR_HCP] = 1250
        tr_f[zair, TR_TMP] = 273
        tr_f[zair, TR_ACE] = 0
        tr_f[zair, TR_IHT] = 0.0

        tr_f[zlitho, TR_RH0] = 3250
        tr_f[zlitho, TR_ALP] = 3.5e-5
        tr_f[zlitho, TR_MAT] = 2
        tr_f[zlitho, TR_ET0] = 1e22
        tr_f[zlitho, TR_HCD] = 4.0
        tr_f[zlitho, TR_HCP] = 1250
        tr_f[zlitho, TR_TMP] = 900
        tr_f[zlitho, TR_ACE] = 120e3
        tr_f[zlitho, TR_IHT] = 0.02e-6 

        tr_f[zxslab, TR_RH0] = 3350
        tr_f[zxslab, TR_ALP] = 3.5e-5
        tr_f[zxslab, TR_MAT] = 2
        tr_f[zxslab, TR_ET0] = 1e22
        tr_f[zxslab, TR_HCD] = 4.0
        tr_f[zxslab, TR_HCP] = 1250
        tr_f[zxslab, TR_TMP] = 900
        tr_f[zxslab, TR_ACE] = 120e3
        tr_f[zxslab, TR_IHT] = 0.02e-6 

        tr_f[zxdetach, TR_ET0] = 1e19
        
        bcstokes[DIM*0 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*1 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*0 + IX] = pylamp_stokes.BC_TYPE_FREESLIP 
        bcstokes[DIM*1 + IX] = pylamp_stokes.BC_TYPE_FREESLIP + pylamp_stokes.BC_TYPE_FLOWTHRU
        
        bcstokesvals[DIM*1 + IX] = np.empty(nx[IZ])
        bcstokesvals[DIM*1 + IX][:] = np.nan
        #bcstokesvals[DIM*1 + IX][grid[IZ] <= h_air] = 0.0
        #bcstokesvals[DIM*1 + IX][(grid[IZ] > h_air) & (grid[IZ] <= h_air + h_litho)] = -0.1 / SECINYR
        #bcstokesvals[DIM*1 + IX][grid[IZ] > h_air + h_litho] = (h_litho * 0.1 / SECINYR) / (L[IZ]-(h_air+h_litho))

        bcheat[DIM*0 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP
        bcheat[DIM*1 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP
        bcheat[DIM*0 + IX] = pylamp_diff.BC_TYPE_FIXFLOW
        bcheat[DIM*1 + IX] = pylamp_diff.BC_TYPE_FIXFLOW

        bcheatvals[DIM*0 + IZ] = 273
        bcheatvals[DIM*1 + IZ] = 1623
        bcheatvals[DIM*0 + IX] = 0
        bcheatvals[DIM*1 + IX] = 0

        bc_internal_type = 1

        output_numpy = True
        output_stride = 1
        output_stride_ma = 0.1            # used if output_stride < 0: output fields every x million years
        output_outdir = "out_slab"

    elif choose_model == 'graphite':
        tracdens = 50
        tracdens_min = 30
        tracs_fence_enable = True

        do_stokes = False
        do_advect = False
        do_heatdiff = True
        do_subgrid_heatdiff = False

        tstep_dif_min = 1e-15
        tstep_dif_max = 1e15
        tstep_adv_min = tstep_dif_min
        tstep_adv_max = tstep_dif_max
        tstep_modifier = 0.67

        output_numpy = True
        output_stride = -1
        output_stride_ma = 1.0

        if model_parameters == 1:
            output_outdir = 'out_graphite'
        else:
            output_outdir = 'out_graphite_no'

        tdep_rho = False
        tdep_eta = False
        etamin = 1e17
        etamax = 1e24

        Tref = 1623
        
        max_it = 1e20
        max_time = SECINMYR * 100

        surface_stabilization = False
        surfstab_theta = 0.5
        surfstab_tstep = -1 

        do_profiling = False

        bc_internal_type = 1


        ### mat defs ###
        d_air = 0e3
        zair = tr_x[:, IZ] < d_air
        zcrust = (tr_x[:, IZ] >= d_air) & (tr_x[:, IZ] < d_air + 30e3)
        zunder = tr_x[:, IZ] >= d_air + 30e3
        zgraphite = (tr_x[:, IZ] >= d_air + 25e3) & (tr_x[:, IZ] < d_air + 30e3)

        tr_f[zair, TR_RH0] = 3300 # real value is 1.293
        tr_f[zair, TR_MAT] = 0
        tr_f[zair, TR_ET0] = 1e23
        tr_f[zair, TR_HCD] = 3.0 # real value is 0.0243
        tr_f[zair, TR_HCP] = 10000 # real value is ~1.005
        tr_f[zair, TR_TMP] = 273 

        tr_f[zcrust, TR_RH0] = 3300
        tr_f[zcrust, TR_MAT] = 1
        tr_f[zcrust, TR_ET0] = 1e23
        tr_f[zcrust, TR_HCD] = 3.0
        tr_f[zcrust, TR_HCP] = 850
        tr_f[zcrust, TR_TMP] = 273 + (tr_x[zcrust, IZ]-d_air) * 600 / 30e3

        tr_f[zunder, TR_RH0] = 3300
        tr_f[zunder, TR_MAT] = 1
        tr_f[zunder, TR_ET0] = 1e23
        tr_f[zunder, TR_HCD] = 3.0
        tr_f[zunder, TR_HCP] = 850
        tr_f[zunder, TR_TMP] = 273 + (tr_x[zunder, IZ]-d_air) * 600 / 30e3

        if model_parameters[0] == 1:
            tr_f[zgraphite, TR_HCD] = 12.0
            tr_f[zgraphite, TR_MAT] = 2
        else:
            pass

        tr_f[:, TR_IHT] = 0.0

        bcheat[DIM*0 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP
        bcheat[DIM*1 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP
        bcheat[DIM*0 + IX] = pylamp_diff.BC_TYPE_FIXFLOW
        bcheat[DIM*1 + IX] = pylamp_diff.BC_TYPE_FIXFLOW

        bcheatvals[DIM*0 + IZ] = 0 + 273
        bcheatvals[DIM*1 + IZ] = 600 + 273
        bcheatvals[DIM*0 + IX] = 0
        bcheatvals[DIM*1 + IX] = 0


        event_codes.append("output_stride_ma = 0.1; bcheatvals[DIM*1 + IZ] = \
                           800 + 273;")
        event_counts.append(1)
        event_enable_times.append(60*60*24*365.25*50e6)
        event_disable_times.append(-1)

    elif choose_model == 'stagnant lid':
        tracdens = 50
        tracdens_min = 30
        tracs_fence_enable = True

        do_stokes = True
        do_advect = True
        do_heatdiff = True
        do_subgrid_heatdiff = True
        subgrid_corr_coefficient = 0.5

        tstep_dif_min = 1e-15
        tstep_dif_max = 1e15
        tstep_adv_min = tstep_dif_min
        tstep_adv_max = tstep_dif_max
        tstep_modifier = 0.67

        output_numpy = True
        output_stride = 10
        output_stride_ma = 0.01

        output_outdir = 'out_staglid'

        tdep_rho = True
        tdep_eta = True
        etamin = 1e17
        etamax = 1e24

        Tref = 1623
        
        max_it = 1e20
        max_time = SECINMYR * 5000

        surface_stabilization = False
        surfstab_theta = 0.5
        surfstab_tstep = -1 

        do_profiling = False

        bc_internal_type = 0

        ### mat defs ###
        zmantle = tr_x[:, IZ] < 660e3

        tr_f[zmantle, TR_MAT] = 1
        tr_f[zmantle, TR_RH0] = 3300
        tr_f[zmantle, TR_ET0] = 1e19
        tr_f[zmantle, TR_HCD] = 3.0
        tr_f[zmantle, TR_HCP] = 1250
        tr_f[zmantle, TR_ALP] = 3.5e-5
        tr_f[zmantle, TR_ACE] = 120e3
        tr_f[zmantle, TR_IHT] = 0.0

        tr_f[zmantle, TR_TMP] = 1623

        bcstokes[DIM*0 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*1 + IZ] = pylamp_stokes.BC_TYPE_FREESLIP
        bcstokes[DIM*0 + IX] = pylamp_stokes.BC_TYPE_FREESLIP #+ pylamp_stokes.BC_TYPE_FLOWTHRU
        bcstokes[DIM*1 + IX] = pylamp_stokes.BC_TYPE_FREESLIP #+ pylamp_stokes.BC_TYPE_FLOWTHRU
        
        bcheat[DIM*0 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP
        bcheat[DIM*1 + IZ] = pylamp_diff.BC_TYPE_FIXTEMP
        bcheat[DIM*0 + IX] = pylamp_diff.BC_TYPE_FIXFLOW
        bcheat[DIM*1 + IX] = pylamp_diff.BC_TYPE_FIXFLOW

        bcheatvals[DIM*0 + IZ] = 0 + 273
        bcheatvals[DIM*1 + IZ] = 1350 + 273
        bcheatvals[DIM*0 + IX] = 0
        bcheatvals[DIM*1 + IX] = 0


    else:
        raise Exception("Invalid model name '" + choose_model + "'")



    #############################################
    ### Config done. Setup rest of things.    ###
    #############################################

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

    if not do_advect:
        staticTracs = True
    else:
        staticTracs = False

    #############################################
    ### PREPS DONE, START THE MAIN TIME LOOP  ###
    #############################################

    it = 0
    totaltime = 0
    time_last_output = 0
    while ((it < max_it) and (totaltime < max_time)):
        it += 1
        pprint(" ****** Time step:", it, "******")

        if bc_internal_type > 0:

            if bc_internal_type == 1:
                # force material zero (water, air) to constant temperature
                idxmat = tr_f[:, TR_MAT] == 0
                tr_f[idxmat, TR_TMP] = 273
            elif bc_internal_type == 2:
                idx = (tr_x[:, IZ] < 300e3) & (tr_x[:, IZ] > 250e3) & (tr_x[:, IX] > 300e3) & (tr_x[:,IX] < 350e3)
                tr_f[idx, TR_TMP] = 273
            elif bc_internal_type == 3:
                idxmat = tr_f[:, TR_MAT] <= 1
                tr_f[idxmat, TR_TMP] = 273

        # handle events
        for ievent in range(len(event_codes)):
            if event_counts[ievent] > 0 and event_enable_times[ievent] <= totaltime \
               and (event_disable_times[ievent] > totaltime or \
               event_disable_times[ievent] < 0):
                pprint("Calling EVENT number " + str(ievent))
                exec(event_codes[ievent])
                event_counts[ievent] = event_counts[ievent] - 1


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

        if bc_internal_type > 0:
            if bc_internal_type == 1:
                idx = f_mat == 0
                f_T[idx] = 273.0

        if do_heatdiff:
            diffusivity = f_k[IZ] / (f_rho * f_Cp)
            #pprint("Min dx, max diff: " + str(np.min(dx)) + ", " + str(np.max(diffusivity)))
            tstep_temp = tstep_modifier * np.min(dx)**2 / np.max(2*diffusivity)
            tstep_temp = min(tstep_temp, tstep_dif_max)
            tstep_temp = max(tstep_temp, tstep_dif_min)

        newvel = 0
        newpres = 0
        tstep_limiter = ""

        if do_stokes:
            pprint("Build stokes")

            if surface_stabilization == False or surfstab_tstep < 0:
                (A, rhs) = pylamp_stokes.makeStokesMatrix(nx, grid, f_etas, f_etan, f_rho, bcstokes, surfstab=False, bcvals=bcstokesvals)
            else:
                (A, rhs) = pylamp_stokes.makeStokesMatrix(nx, grid, f_etas, f_etan, f_rho, bcstokes, surfstab=True, tstep=surfstab_tstep, surfstab_theta=surfstab_theta, bcvals=bcstokesvals)

            pprint("Solve stokes")
            # Solve it!
            #x = scipy.sparse.linalg.bicgstab(scipy.sparse.csc_matrix(A), rhs)[0]
            x = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), rhs)

            (newvel, newpres) = pylamp_stokes.x2vp(x, nx)

            pprint("Min/max vel: ", SECINYR*np.min(np.sqrt(newvel[IZ]**2 + newvel[IX]**2)), SECINYR*np.max(np.sqrt(newvel[IZ]**2 + newvel[IX]**2)), "m/yr")

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
                (A, rhs) = pylamp_stokes.makeStokesMatrix(nx, grid, f_etas, f_etan, f_rho, bcstokes, surfstab=True, tstep=tstep, surfstab_theta=surfstab_theta, bcvals=bcstokesvals)

                pprint ("Resolve stokes")
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

            if it == 1:
                # On first timestep interpolate absolute temperature values to tracers ...
                # Also, assume that all tracers are within the domain at this point
                pprint("grid2trac T")
                pylamp_trac.grid2trac(tr_x[:], l_interp_tracvals[:], grid, [newtemp], nx, method=pylamp_trac.INTERP_METHOD_LINEAR, stopOnError=True, staticTracs=staticTracs)
                tr_f[:, TR_TMP] = l_interp_tracvals[:, 0]
            else:
                # ... on subsequent timesteps interpolate only the change to avoid numerical diffusion
                # (and exclude those that are outside the domain)
                pprint("grid2trac dT")
                newdT = newtemp - f_T
                ##pylamp_trac.grid2trac(tr_x[:], l_interp_tracvals[:], grid, [newdT], nx, method=pylamp_trac.INTERP_METHOD_LINEAR, stopOnError=True, staticTracs=staticTracs)
                pylamp_trac.grid2trac(tr_x[:], l_interp_tracvals[:], grid, [newdT], nx, method=pylamp_trac.INTERP_METHOD_NEAREST, stopOnError=True, staticTracs=staticTracs)
                tr_f[:, TR_TMP] = tr_f[:, TR_TMP] + l_interp_tracvals[:, 0]

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
                # correction at tracers
                # NB! Assumes regular grid, 2D
                if do_subgrid_heatdiff:
                    # prepare diffusivity on nodes for interpolation
                    f_diffusivity_mp = 0.5 * (f_k[IZ][:-1, :-1] + f_k[IZ][:-1, 1:]) + 0.5 * (f_k[IX][0:-1,:-1] + f_k[IX][1:,:-1])
                    f_diffusivity = np.zeros_like(f_T)
                    f_diffusivity_count = np.zeros_like(f_T)
                    f_diffusivity[0:-1,0:-1] += f_diffusivity_mp[:,:]
                    f_diffusivity[0:-1,1:] += f_diffusivity_mp[:,:]
                    f_diffusivity[1:,0:-1] += f_diffusivity_mp[:,:]
                    f_diffusivity[1:,1:] += f_diffusivity_mp[:,:]
                    f_diffusivity_count[0:-1,0:-1] += 1
                    f_diffusivity_count[0:-1,1:] += 1
                    f_diffusivity_count[1:,0:-1] += 1
                    f_diffusivity_count[1:,1:] += 1
                    f_diffusivity = f_diffusivity / f_diffusivity_count

                    tr_fi_heat = np.zeros((tr_f.shape[0], 4))
                    pylamp_trac.grid2trac(tr_x, tr_fi_heat, grid, [f_diffusivity, f_rho, f_Cp, f_T], nx, method=pylamp_trac.INTERP_METHOD_LINEAR, stopOnError=True, staticTracs=staticTracs)

                    # subgrid diffusion
                    d = subgrid_corr_coefficient
                    
                    dt_diff_tr = (tr_fi_heat[:,2] * tr_fi_heat[:,1] / tr_fi_heat[:,0]) / (2.0/dx[IX]**2 + 2.0/dx[IZ]**2)
                    #print("dt_diff_tr: ", dt_diff_tr.shape)
                    dT_subg_tr = (tr_fi_heat[:,3] - tr_f[:,TR_TMP]) * (1.0 - np.exp(-d * tstep / dt_diff_tr))
                    #print("dT_subg_tr: ", dT_subg_tr.shape)

                    pylamp_trac.trac2grid(tr_x, dT_subg_tr[:,None], mesh, grid, [f_sgc], nx, avgscheme=[pylamp_trac.INTERP_AVG_ARITHW])
                    #print("f_sgc: ", f_sgc.shape)
                    ## TODO: correct f_sgc at boundaries (???)
                    dT_rem_nodes = newdT - f_sgc #f_sgc == "dT_subg_nodes"
                    #print("dT_rem_nodes: ", dT_rem_nodes.shape)
                    dT_rem_tr = np.zeros((tr_f.shape[0]))
                    #print("dT_rem_tr: ", dT_rem_tr.shape)
                    pylamp_trac.grid2trac(tr_x, dT_rem_tr[:,None], grid, [f_sgc], nx, method=pylamp_trac.INTERP_METHOD_LINEAR, stopOnError=True, staticTracs=staticTracs)
                    #print("dT_rem_tr: ", dT_rem_tr.shape)
                    #print("tr_f[:,TR_TMP]: ", tr_f[:,TR_TMP].shape)
                    #print("dT_subg_tr: ", dT_subg_tr)
                    #print("dT_rem_tr: ", dT_rem_tr)
                    tr_f[:, TR_TMP] = tr_f[:,TR_TMP] + dT_subg_tr + dT_rem_tr

            # end of heat diffusion 

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

            if bcstokes[DIM*0 + IZ] & pylamp_stokes.BC_TYPE_NOSLIP:
                vels[IX][0,:] = -vels[IX][1,:]  # i.e. vel is zero at bnd
                vels[IZ][0,:] = -vels[IZ][1,:]  # i.e. vel out/in of bnd is zero (no in/outflow)
            elif bcstokes[DIM*0 + IZ] & pylamp_stokes.BC_TYPE_FREESLIP:
                vels[IX][0,:] = vels[IX][1,:]   # i.e. vel change is zero across bnd
                vels[IZ][0,:] = -vels[IZ][1,:]  # i.e. vel out/in of bnd is zero (no in/outflow)
            elif bcstokes[DIM*0 + IZ] & pylamp_stokes.BC_TYPE_CYCLIC:
                vels[IX][0,:] = vels[IX][-2,:]  
                vels[IZ][0,:] = vels[IZ][-2,:]

            if bcstokes[DIM*0 + IX] & pylamp_stokes.BC_TYPE_NOSLIP:
                vels[IZ][:,0] = -vels[IZ][:,1]
                vels[IX][:,0] = -vels[IX][:,1]
            elif bcstokes[DIM*0 + IX] & pylamp_stokes.BC_TYPE_FREESLIP:
                vels[IZ][:,0] = vels[IZ][:,1]
                vels[IX][:,0] = -vels[IX][:,1]
            elif bcstokes[DIM*0 + IX] & pylamp_stokes.BC_TYPE_CYCLIC:
                vels[IZ][:,0] = vels[IZ][:,-2]
                vels[IX][:,0] = vels[IX][:,-2]
            if bcstokes[DIM*0 + IX] & pylamp_stokes.BC_TYPE_FLOWTHRU:
                vels[IX][:,0] = vels[IX][:,1]

            if bcstokes[DIM*1 + IZ] & pylamp_stokes.BC_TYPE_NOSLIP:
                vels[IX][-1,:] = -vels[IX][-2,:]
                vels[IZ][-1,:] = -vels[IZ][-2,:]
            elif bcstokes[DIM*1 + IZ] & pylamp_stokes.BC_TYPE_FREESLIP:
                vels[IX][-1,:] = vels[IX][-2,:]
                vels[IZ][-1,:] = -vels[IZ][-2,:]
            elif bcstokes[DIM*1 + IZ] & pylamp_stokes.BC_TYPE_CYCLIC:
                vels[IX][-1,:] = vels[IX][1,:]
                vels[IZ][-1,:] = vels[IZ][1,:]

            if bcstokes[DIM*1 + IX] & pylamp_stokes.BC_TYPE_NOSLIP:
                vels[IZ][:,-1] = -vels[IZ][:,-2]
                vels[IX][:,-1] = -vels[IX][:,-2]
            elif bcstokes[DIM*1 + IX] & pylamp_stokes.BC_TYPE_FREESLIP:
                vels[IZ][:,-1] = vels[IZ][:,-2]
                vels[IX][:,-1] = -vels[IX][:,-2]
            elif bcstokes[DIM*1 + IX] & pylamp_stokes.BC_TYPE_CYCLIC:
                vels[IZ][:,-1] = vels[IZ][:,1]
                vels[IX][:,-1] = vels[IX][:,1]
            if bcstokes[DIM*1 + IX] & pylamp_stokes.BC_TYPE_FLOWTHRU:
                vels[IX][:,-1] = vels[IX][:,-2]

            trac_vel, tr_x = pylamp_trac.RK(tr_x[:,:], [newgridz, newgridx], vels, nx, tstep)

            # do not allow tracers to advect outside the domain
            for d in range(DIM):
                if bcstokes[d*0 + IX] & pylamp_stokes.BC_TYPE_CYCLIC:
                    tr_x[tr_x[:,d] <= 0, d] += L[d]
                    tr_x[tr_x[:,d] >= L[d], d] -= L[d]
                else:
                    idx = tr_x[:,d] <= 0
                    if tracs_fence_enabled and not (bcstokes[DIM*0 + d] & pylamp_stokes.BC_TYPE_FLOWTHRU):
                        tr_x[idx, d] = EPS
                    else:
                        tr_f[idx, TR__ID] = -1
                    idx = tr_x[:,d] >= L[d]
                    if tracs_fence_enabled and not (bcstokes[DIM*1 + d] & pylamp_stokes.BC_TYPE_FLOWTHRU):
                        tr_x[idx, d] = L[d]-EPS
                    else:
                        tr_f[idx, TR__ID] = -1

            # delete tracers that have advected outside the domain
            idx_tracs_outside = tr_f[:,TR__ID] < 0
            dntrac = np.sum(idx_tracs_outside)
            pprint("Removing", dntrac, "tracers")
            tr_x = np.delete(tr_x, np.where(idx_tracs_outside)[0], axis=0)
            tr_f = np.delete(tr_f, np.where(idx_tracs_outside)[0], axis=0)
            trac_vel = np.delete(trac_vel, np.where(idx_tracs_outside)[0], axis=0)
            ntrac = ntrac - dntrac

            ### TODO:
            # fill in the gaps where there are no tracers, or num of
            # tracers per element is too low,
            # then do grid2trac for those

            ielem = np.floor((nx[IZ]-1) * tr_x[:,IZ] / L[IZ]).astype(int)
            jelem = np.floor((nx[IX]-1) * tr_x[:,IX] / L[IX]).astype(int)

            kelem = ielem * (nx[IX]-1) + jelem
            kelem = np.append(kelem, np.arange(np.prod(np.array(nx)-1))) # to make sure every element is accounted for at least once in the bincount()
            ntrac_per_elem = np.bincount(kelem)-1
            idx_toofewtracs = ntrac_per_elem < tracdens_min
            
            prev_tr_f = np.copy(tr_f) # used later for output writing
            prev_tr_x = np.copy(tr_x) # used later for output writing

            if np.sum(idx_toofewtracs) == 0:
                # nothing to do here
                pprint("Injecting zero tracers")
            else:
                kelems_missing_tracs = np.where(idx_toofewtracs)[0]
                ielems_missing_tracs = np.floor(kelems_missing_tracs / (nx[IX]-1)).astype(int)
                jelems_missing_tracs = (kelems_missing_tracs % (nx[IX]-1)).astype(int)

                n_missing_tracs = tracdens - ntrac_per_elem[idx_toofewtracs]
                dntrac = np.sum(n_missing_tracs)
                pprint("Injecting", dntrac, "new tracers")
                jelem_bnds = [grid[IX][jelems_missing_tracs], grid[IX][jelems_missing_tracs+1]] 
                ielem_bnds = [grid[IZ][ielems_missing_tracs], grid[IZ][ielems_missing_tracs+1]] 

                prev_tr_f = np.copy(tr_f)
                prev_tr_x = np.copy(tr_x)
                prev_trac_vel = np.copy(trac_vel)

                for i in range(n_missing_tracs.size):
                    tr_x_tmp = np.random.rand(n_missing_tracs[i], DIM)
                    tr_x_tmp[:,IX] = tr_x_tmp[:,IX] * (jelem_bnds[1][i] - jelem_bnds[0][i]) + jelem_bnds[0][i]
                    tr_x_tmp[:,IZ] = tr_x_tmp[:,IZ] * (ielem_bnds[1][i] - ielem_bnds[0][i]) + ielem_bnds[0][i]
                    tr_f_tmp = np.zeros((n_missing_tracs[i], NFTRAC))
                    trac_vel_tmp = np.zeros((n_missing_tracs[i], DIM))

                    maxid_current = np.max(tr_f[:,TR__ID])
                    tr_f_tmp[:,TR__ID] = np.arange(maxid_current, maxid_current + n_missing_tracs[i])

                    for itracf in range(NFTRAC):
                        # the new tracers will get values for the tracer functions
                        # from the existing tracers in the element (simple average)
                        if itracf != TR__ID:
                            idx_tracs_in_elem = (ielem == ielems_missing_tracs[i]) & (jelem == jelems_missing_tracs[i])
                            tr_f_tmp[:,itracf] = np.sum(prev_tr_f[idx_tracs_in_elem,itracf]) / np.sum(idx_tracs_in_elem)

                    tr_f = np.append(tr_f, tr_f_tmp, axis=0)
                    tr_x = np.append(tr_x, tr_x_tmp, axis=0)
                    trac_vel = np.append(trac_vel, trac_vel_tmp, axis=0)
                    ntrac = ntrac + n_missing_tracs[i]
                    


        if output_numpy and (it == 1 or (output_stride > 0 and (it-1) % output_stride == 0) or (output_stride < 0 and (totaltime - time_last_output)/SECINMYR > output_stride_ma)):
            if not os.path.exists(output_outdir):
                os.makedirs(output_outdir)
            if output_stride > 0:
                time_last_output = totaltime
            else:
                time_last_output = SECINMYR * output_stride_ma * float(int(totaltime/(SECINMYR*output_stride_ma)))
            if do_stokes:
                if do_heatdiff:
                    np.savez(output_outdir + "/griddata.{:06d}.npz".format(it), gridz=grid[IZ], gridx=grid[IX], velz=newvel[IZ], velx=newvel[IX], pres=newpres, rho=f_rho, temp=newtemp, sgc=f_sgc, eta=f_etas, tstep=it, time=totaltime)
                else:
                    np.savez(output_outdir + "/griddata.{:06d}.npz".format(it), gridz=grid[IZ], gridx=grid[IX], velz=newvel[IZ], velx=newvel[IX], pres=newpres, rho=f_rho, eta=f_etas, temp=newvel[IX]*0.0, tstep=it, time=totaltime)
                np.savez(output_outdir + "/tracs.{:06d}.npz".format(it), tr_x=tr_x, tr_f=tr_f, tr_v=trac_vel, tstep=it, time=totaltime)
            else:
                np.savez(output_outdir + "/griddata.{:06d}.npz".format(it), gridz=grid[IZ], gridx=grid[IX], rho=f_rho, temp=newtemp, eta=f_etas, sgc=f_sgc, tstep=it, time=totaltime)
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
