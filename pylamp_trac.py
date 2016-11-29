#!/usr/bin/python3

INTERP_AVG_ARITHMETIC = 1
INTERP_AVG_GEOMETRIC = 2
INTERP_AVG_WEIGHTED = 4
INTERP_AVG_ARITHW = INTERP_AVG_ARITHMETIC + INTERP_AVG_WEIGHTED
INTERP_AVG_GEOMW = INTERP_AVG_GEOMETRIC + INTERP_AVG_WEIGHTED

INTERP_METHOD_IDW = 1
INTERP_METHOD_GRIDDATA = 2
INTERP_METHOD_ELEM = 4
INTERP_METHOD_NEAREST = 8
INTERP_METHOD_LINEAR = 16   # actually, bilinear, i.e. non-linear ...
INTERP_METHOD_VELDIV = 32
INTERP_METHOD_DIFF = 64

from pylamp_const import *
import numpy as np
from scipy.interpolate import griddata
import itertools
import sys

def grid2trac(tr_x, tr_f, grid, gridfield, nx, defval=np.nan, method=INTERP_METHOD_LINEAR, stopOnError=False):
    # Interpolate values (gridfield) from grid to tracer
    # value (tr_f). 

    # NB! Designed for regular grids!

    assert len(gridfield) == tr_f.shape[1]
    assert (method & INTERP_METHOD_LINEAR) or (method & INTERP_METHOD_NEAREST) or\
            (method & INTERP_METHOD_VELDIV)

    nfield = len(gridfield)

    Lmax = [grid[i][-1] for i in range(DIM)]
    Lmin = [grid[i][0] for i in range(DIM)]
    L = [Lmax[i]-Lmin[i] for i in range(DIM)]

    ielem = np.floor((nx[IZ]-1) * (tr_x[:,IZ]-Lmin[IZ]) / L[IZ]).astype(int)
    jelem = np.floor((nx[IX]-1) * (tr_x[:,IX]-Lmin[IX]) / L[IX]).astype(int)

    #idxdefval = (tr_x[:,IZ] < Lmin[IZ]) | (tr_x[:,IZ] > Lmax[IZ]) | \
    #        (tr_x[:,IX] < Lmin[IX]) | (tr_x[:,IX] > Lmax[IX])

    idxdefval = (ielem < 0) | (ielem > nx[IZ]-1) | (jelem < 0) | (jelem > nx[IX]-1)
    if stopOnError and np.sum(idxdefval) > 0:
        raise Exception("stopOnError in grid2trac")

    if np.sum(idxdefval) > 0:
        print("!!! Warning, grid2trac(): Using default value for extrapolation in ", np.sum(idxdefval), "tracers")
#        raise Exception("!!! Error, grid2trac(): Using default value for extrapolation in ", np.sum(idxdefval), "tracers")

    ielem[idxdefval] = 0
    jelem[idxdefval] = 0

    dx = [grid[d][1:]-grid[d][:-1] for d in range(DIM)]

    ntrac = tr_x.shape[0]

    # distances of tra
    distx = np.zeros((ntrac,4))
    distz = np.zeros((ntrac,4))

    for di in [0,1]:
        for dj in [0,1]:
            icorner = di * 2 + dj
            distz[:,icorner] = np.abs(tr_x[:,IZ] - grid[IZ][ielem+di])
            distx[:,icorner] = np.abs(tr_x[:,IX] - grid[IX][jelem+dj])

    if method & INTERP_METHOD_NEAREST:
        disttot = distz**2 + distx**2
        nearestcorner = np.argmin(disttot, axis=1)
        nearestcorner_dj = (nearestcorner % 2).astype(int)
        nearestcorner_di = ((nearestcorner - nearestcorner_dj) / 2).astype(int)

    for ifield in range(nfield):
        if method & INTERP_METHOD_NEAREST:
            if method & INTERP_METHOD_DIFF:
                tr_f[:,ifield] += gridfield[ifield][ielem + nearestcorner_di, jelem + nearestcorner_dj]
            else:
                tr_f[:,ifield] = gridfield[ifield][ielem + nearestcorner_di, jelem + nearestcorner_dj]
        elif method & INTERP_METHOD_LINEAR:
            xavg1 = gridfield[ifield][ielem + 0, jelem + 0] * distx[:, 1] + gridfield[ifield][ielem + 0, jelem + 1] * distx[:, 0]
            xavg1 = xavg1 / (distx[:, 0] + distx[:, 1])
            xavg2 = gridfield[ifield][ielem + 1, jelem + 0] * distx[:, 3] + gridfield[ifield][ielem + 1, jelem + 1] * distx[:, 2]
            xavg2 = xavg2 / (distx[:, 1] + distx[:, 2])
            zavg1 = xavg1 * distz[:,2] + xavg2 * distz[:,0]
            zavg1 = zavg1 / (distz[:,2] + distz[:,0])
            if method & INTERP_METHOD_DIFF:
                tr_f[:,ifield] += zavg1[:]
            else:
                tr_f[:,ifield] = zavg1[:]

        elif method & INTERP_METHOD_VELDIV:
            # See Wang et al 2015 (or Meyer and Jenny 2004)
            # 2nd order bilinear interpolation
            # conserves the divergence (=0) also during the advection

            # Only 2D implemented!
            # Does not work properly yet as vz,vx are on different grids...
            # Current invocation of grid2trac is not compatible with this


            # local interpolated velocity at (x1,x2):
            # U_i^L = (1-x1)*(1-x2)*U_i^a + x1*(1-x2)*U_i^b + (1-x1)*x2*U_i^c + x1*x2*U_i^d
            #
            # Correction term to conserve the divergence
            # U_i = U_i^L + \Delta U_i
            # 
            # in 2D:
            # \Delta U_1 = x1*(1-x1)*(C10 + x2*C12)
            # \Delta U_2 = x2*(1-x2)*C20
            #
            # where:
            # C12 = (\delta x1 / 2\delta x3) * f(U_3) = 0
            # C10 = (\delta x1 / 2\delta x2) * (U_2^a - U_2^c + U_2^d - U_2^b)
            # C20 = (\delta x2 / 2\delta x1) * (U_1^a - U_1^b + U_1^d - U_1^c + C12)
            #
            # here we use x1: X, x2: Z

            if nfield != 2:
                raise Exception("grid2trac(): method INTERP_METHOD_VELDIV only works in 2D and expects field to be (vz,vx)")

            # normalize distances to unit cube
            # assumes rectangular mesh
            dxn = distx[:, 0] / (distx[:, 0] + distx[:, 1])
            dzn = distz[:, 0] / (distz[:, 0] + distz[:, 2])

            Ux = \
                    (1-dxn) * (1-dzn) * gridfield[IX][ielem + 0, jelem + 0] + \
                    dxn * (1-dzn) * gridfield[IX][ielem + 0, jelem + 1] + \
                    (1-dxn) * dzn * gridfield[IX][ielem + 1, jelem + 0] + \
                    dxn * dzn * gridfield[IX][ielem + 1, jelem + 1]
            Uz = \
                    (1-dxn) * (1-dzn) * gridfield[IZ][ielem + 0, jelem + 0] + \
                    dxn * (1-dzn) * gridfield[IZ][ielem + 0, jelem + 1] + \
                    (1-dxn) * dzn * gridfield[IZ][ielem + 1, jelem + 0] + \
                    dxn * dzn * gridfield[IZ][ielem + 1, jelem + 1]


            # correction terms
            C10 = (0.5 * dx[IX][jelem] / dx[IZ][ielem]) * \
                    (gridfield[IZ][ielem + 0, jelem + 0] - gridfield[IZ][ielem + 1, jelem + 0] +\
                    gridfield[IZ][ielem + 1, jelem + 1] - gridfield[IZ][ielem + 0, jelem + 1])
            C20 = (0.5 * dx[IZ][ielem] / dx[IX][jelem]) * \
                    (gridfield[IX][ielem + 0, jelem + 0] - gridfield[IX][ielem + 0, jelem + 1] +\
                    gridfield[IX][ielem + 1, jelem + 1] - gridfield[IX][ielem + 1, jelem + 0])

            dU1 = dxn * (1-dxn) * C10
            dU2 = dzn * (1-dzn) * C20

            tr_f[:,IX] = Ux[:] + dU1[:]
            tr_f[:,IZ] = Uz[:] + dU2[:]

        tr_f[idxdefval,ifield] = defval

    return


def trac2grid(tr_x, tr_f, mesh, grid, gridfield, nx, distweight=None, avgscheme=None, method=INTERP_METHOD_ELEM, debug=False):
    # NB! Designed for regular grids

    assert len(gridfield) == tr_f.shape[1]

    if avgscheme is None:
        avgscheme = [INTERP_AVG_ARITHMETIC + INTERP_AVG_WEIGHTED for i in range(len(gridfield))]

    avgsch_has_weighted = False
    avgsch_has_nonweighted = False

    for i in range(len(gridfield)):
        if avgscheme[i] & INTERP_AVG_WEIGHTED:
            avgsch_has_weighted = True
        else:
            avgsch_has_nonweighted = True

    assert type(avgscheme) == type([])
    assert len(avgscheme) == len(gridfield)

    ntracfield = len(gridfield)
    ntrac = tr_x.shape[0]

    if method & INTERP_METHOD_GRIDDATA:
        if DIM != 2:
            print("!!! NOT IMPLEMENTED")
        
        gridval = griddata(tr_x, tr_f, (mesh[0], mesh[1]), 'nearest')

        for d in range(ntracfield):
            gridfield[d][:,:] = gridval[:,:,d]

    elif method & INTERP_METHOD_ELEM:
        # make local copies of grid/mesh in case we need to modify them
        localgrid = [np.array(grid[d], copy=True) for d in range(DIM)]
        localmesh = [np.array(mesh[d], copy=True) for d in range(DIM)]
        localnx = [nx[d] for d in range(DIM)]

        grid = localgrid
        mesh = localmesh
        nx = localnx

        gridmodified = False

        addednodesright = [0] * DIM
        addednodesleft  = [0] * DIM

        for d in range(DIM):
            while np.min(tr_x[:,d]) < grid[d][0]:
                gridmodified = True
                grid[d] = np.concatenate([np.array([grid[d][0] - (grid[d][1]-grid[d][0])]), np.array(grid[d])])
                nx[d] += 1
                addednodesleft[d] += 1
            while np.max(tr_x[:,d]) > grid[d][-1]:
                gridmodified = True
                grid[d] = npconcatenate([np.array(grid[d]), [np.array([grid[d][-1] + (grid[d][-1]-grid[d][-2])])]])
                nx[d] += 1
                addednodesright[d] += 1

        if gridmodified:
            mesh = np.meshgrid(*grid, indexing='ij')

        Lmax = [grid[i][-1] for i in range(DIM)]
        Lmin = [grid[i][0] for i in range(DIM)]
        L = [Lmax[i]-Lmin[i] for i in range(DIM)]

        ielem = np.floor((nx[IZ]-1) * (tr_x[:,IZ]-Lmin[IZ]) / L[IZ]).astype(int)
        jelem = np.floor((nx[IX]-1) * (tr_x[:,IX]-Lmin[IX]) / L[IX]).astype(int)
        elem = [ielem, jelem]

        #inode = np.concatenate(ielem, ielem+1, ielem, ielem+1)
        #jnode = np.concatenate(jelem, jelem, jelem+1, jelem+1)
        #tr_idx = np.concatenate(4*[np.arange(tr_x.shape[0])])

        #valid_idx = (inode >= 0) | (inode < nx[IZ]) | \
        #        (jnode >= 0) | (jnode < nx[IX])

        #inode = inode[valid_idx]
        #jnode = jnode[valid_idx]
        #tr_idx = tr_idx[valid_idx]

        traccount = np.zeros(np.array(mesh[0].shape))
        tracsum = np.zeros(np.array(mesh[0].shape))
        tracweightsum = np.zeros(np.array(mesh[0].shape))
        tracweight = np.zeros((tr_x.shape[0], 4)) # 4 <- one for each corner (2D), i.e. surrounding node

        dxnorm = [[]] * 2  # distance to node on left, node on right
        dxnorm[0] = [(tr_x[:,d] - grid[d][elem[d]]) / (grid[d][elem[d]+1] - grid[d][elem[d]]) for d in range(DIM)]
        #dxnorm[1] = [(grid[d][elem[d]+1] - tr_x[:,d]) / (grid[d][elem[d]+1] - grid[d][elem[d]]) for d in range(DIM)]
        dxnorm[1] = [1 - dxnorm[0][d] for d in range(DIM)]  # this should be equivalent to the line above

        if avgsch_has_weighted:
            tracweight[:,0] = (1 - dxnorm[0][IX]) * (1 - dxnorm[0][IZ]) #/ (dxnorm[0][IZ]*dxnorm[0][IX])
            tracweight[:,1] = (1 - dxnorm[0][IX]) * (1 - dxnorm[1][IZ]) #/ ((1-dxnorm[0][IZ])*dxnorm[1][IX])
            tracweight[:,2] = (1 - dxnorm[1][IX]) * (1 - dxnorm[0][IZ]) #/ (dxnorm[1][IZ]*(1-dxnorm[0][IX]))
            tracweight[:,3] = (1 - dxnorm[1][IX]) * (1 - dxnorm[1][IZ]) #/ ((1-dxnorm[1][IZ])*(1-dxnorm[1][IX]))

            np.add.at(tracweightsum, [ielem, jelem], tracweight[:,0])
            np.add.at(tracweightsum, [ielem+1, jelem], tracweight[:,1])
            np.add.at(tracweightsum, [ielem, jelem+1], tracweight[:,2])
            np.add.at(tracweightsum, [ielem+1, jelem+1], tracweight[:,3])

        if avgsch_has_nonweighted:
            np.add.at(traccount, [ielem,jelem], 1)
            np.add.at(traccount, [ielem+1,jelem], 1)
            np.add.at(traccount, [ielem,jelem+1], 1)
            np.add.at(traccount, [ielem+1,jelem+1], 1)

        # TODO: check and repair elements where there are no tracers at all
        # TODO: Find faster alternative for np.add.at

        for ifield in range(ntracfield):
            tracsum[:,:] = 0

            if avgscheme[ifield] & INTERP_AVG_ARITHMETIC:
                if avgscheme[ifield] & INTERP_AVG_WEIGHTED:
                    np.add.at(tracsum, [ielem, jelem], tr_f[:,ifield] * tracweight[:,0])
                    np.add.at(tracsum, [ielem+1, jelem], tr_f[:,ifield] * tracweight[:,1])
                    np.add.at(tracsum, [ielem, jelem+1], tr_f[:,ifield] * tracweight[:,2])
                    np.add.at(tracsum, [ielem+1, jelem+1], tr_f[:,ifield] * tracweight[:,3])

                    newgridfield = tracsum[:,:] / tracweightsum[:,:]
                else:
                    np.add.at(tracsum, [ielem, jelem], tr_f[:,ifield])
                    np.add.at(tracsum, [ielem+1, jelem], tr_f[:,ifield])
                    np.add.at(tracsum, [ielem, jelem+1], tr_f[:,ifield])
                    np.add.at(tracsum, [ielem+1, jelem+1], tr_f[:,ifield])
                    newgridfield = tracsum[:,:] / traccount[:,:]
            elif avgscheme[ifield] & INTERP_AVG_GEOMETRIC:
                if avgscheme[ifield] & INTERP_AVG_WEIGHTED:
                    np.add.at(tracsum, [ielem, jelem], np.log(tr_f[:,ifield]) * tracweight[:,0])
                    np.add.at(tracsum, [ielem+1, jelem], np.log(tr_f[:,ifield]) * tracweight[:,1])
                    np.add.at(tracsum, [ielem, jelem+1], np.log(tr_f[:,ifield]) * tracweight[:,2])
                    np.add.at(tracsum, [ielem+1, jelem+1], np.log(tr_f[:,ifield]) * tracweight[:,3])
                else:
                    np.add.at(tracsum, [ielem, jelem], np.log(tr_f[:,ifield]))
                    np.add.at(tracsum, [ielem+1, jelem], np.log(tr_f[:,ifield]))
                    np.add.at(tracsum, [ielem, jelem+1], np.log(tr_f[:,ifield]))
                    np.add.at(tracsum, [ielem+1, jelem+1], np.log(tr_f[:,ifield]))

                # if original value is zero, log returns -inf... fix that
                tracsum[np.isinf(tracsum)] = 0

                if avgscheme[ifield] & INTERP_AVG_WEIGHTED:
                    newgridfield = np.exp(tracsum[:,:] / tracweightsum[:,:])
                else:
                    newgridfield = np.exp(tracsum[:,:] / traccount[:,:])

            else:
                print ("!!! ERROR INVALID AVERAGING SCHEME")

            #gridfield[ifield][nonzeroelemidx] = newgridfield[nonzeroelemidx]

            if gridmodified:
                gridfield[ifield][:,:] = newgridfield[addednodesleft[0]:(nx[0]-addednodesright[0]), addednodesleft[1]:(nx[1]-addednodesright[1])]
            else:
                gridfield[ifield][:,:] = newgridfield[:,:]

        return

        
def RK(tr_x, grids, vels, nx, tstep, order=4):
    # grids[IZ] holds the grid (i.e. both x and z coords) for vz 
    # and grids[IX] for vx. I.e. grids[IZ][IX] are the x coordinates
    # of the z-velocity locations

    if order != 2 and order != 4:
        raise Exception("Sorry, don't know how to do that")

    if len(nx) != 2:
        raise Exception("Sorry, only 2D supported at the moment")

    if order == 2:
        # TODO: implement as in RK4
        print ("Warning! RK2 not properly working")
        trac_vel = np.zeros((tr_x.shape[0], DIM))
        tracs_half_h = np.zeros((tr_x.shape[0], DIM))
        tracs_full_h = np.zeros((tr_x.shape[0], DIM))
        tracvel_half_h = np.zeros((tr_x.shape[0], DIM))
        grid2trac(tr_x, trac_vel, grid, gridvel, nx, defval=0)
        for d in range(DIM):
            tracs_half_h[:,d] = tr_x[:,d] + 0.5 * tstep * trac_vel[:,d]
        grid2trac(tracs_half_h, tracvel_half_h, grid, gridvel, nx, defval=0)
        for d in range(DIM):
            tracs_full_h[:,d] = tr_x[:,d] + tstep * tracvel_half_h[:,d]
        return trac_vel, tracs_full_h

    elif order == 4:
        # TODO: Combine two calls to grid2trac() to one
        k1vel = np.zeros((tr_x.shape[0], DIM))
        tmp   = np.zeros((tr_x.shape[0], DIM))
        k2vel = np.zeros_like(k1vel)
        k3vel = np.zeros_like(k1vel)
        k4vel = np.zeros_like(k1vel)
        k2loc = np.zeros_like(k1vel)
        k3loc = np.zeros_like(k1vel)
        k4loc = np.zeros_like(k1vel)
        tr_x_final = np.zeros_like(k1vel)
        vel_final = np.zeros_like(k1vel)

        interpmethod = INTERP_METHOD_VELDIV

        grid2trac(tr_x, tmp, grids, vels, [nx[IZ]+1, nx[IX]+1], defval=0, method=interpmethod)
        k1vel[:,IZ] = tmp[:,IZ]
        k1vel[:,IX] = tmp[:,IX]
        
        for d in range(DIM):
            k2loc[:,d] = tr_x[:,d] + 0.5 * tstep * k1vel[:,d]
        grid2trac(k2loc, tmp, grids, vels, [nx[IZ]+1, nx[IX]+1], defval=0, method=interpmethod)
        k2vel[:,IZ] = tmp[:,IZ]
        k2vel[:,IX] = tmp[:,IX]

        for d in range(DIM):
            k3loc[:,d] = tr_x[:,d] + 0.5 * tstep * k2vel[:,d]
        grid2trac(k3loc, tmp, grids, vels, [nx[IZ]+1, nx[IX]+1], defval=0, method=interpmethod)
        k3vel[:,IZ] = tmp[:,IZ]
        k3vel[:,IX] = tmp[:,IX]

        for d in range(DIM):
            k4loc[:,d] = tr_x[:,d] + tstep * k3vel[:,d]
        grid2trac(k4loc, tmp, grids, vels, [nx[IZ]+1, nx[IX]+1], defval=0, method=interpmethod)
        k4vel[:,IZ] = tmp[:,IZ]
        k4vel[:,IX] = tmp[:,IX]


        for d in range(DIM):
            tr_x_final[:,d] = tr_x[:,d] + (1/6)*tstep * (k1vel[:,d] + k2vel[:,d] + k3vel[:,d] + k4vel[:,d])
            vel_final[:,d] = (tr_x_final[:,d] - tr_x[:,d]) / tstep

        return vel_final, tr_x_final
