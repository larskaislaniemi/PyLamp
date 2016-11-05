#!/usr/bin/python3

INTERP_AVG_ARITHMETIC = 0
INTERP_AVG_GEOMETRIC = 1

INTERP_METHOD_IDW = 0
INTERP_METHOD_GRIDDATA = 1
INTERP_METHOD_ELEM = 2

from pylamp_const import *
import numpy as np
from scipy.interpolate import griddata
import itertools
import sys

def grid2trac(tr_x, tr_f, grid, gridfield, nx):
    # NB! Designed for regular grids!

    assert len(gridfield) == tr_f.shape[1]

    nfield = len(gridfield)

    Lmax = [grid[i][-1] for i in range(DIM)]
    Lmin = [grid[i][0] for i in range(DIM)]
    L = [Lmax[i]-Lmin[i] for i in range(DIM)]

    ielem = np.floor((nx[IZ]-1) * (tr_x[:,IZ]-Lmin[IZ]) / L[IZ]).astype(int)
    jelem = np.floor((nx[IX]-1) * (tr_x[:,IX]-Lmin[IX]) / L[IX]).astype(int)

    ntrac = tr_x.shape[0]

    # distances of tra
    distx = np.zeros((ntrac,4))
    distz = np.zeros((ntrac,4))

    for di in [0,1]:
        for dj in [0,1]:
            icorner = di * 2 + dj
            distz[:,icorner] = tr_x[:,IZ] - grid[IZ][ielem+di]
            distx[:,icorner] = tr_x[:,IX] - grid[IX][jelem+dj]
    disttot = distz**2 + distx**2
    nearestcorner = np.argmin(disttot, axis=1)
    nearestcorner_dj = (nearestcorner % 2).astype(int)
    nearestcorner_di = ((nearestcorner - nearestcorner_dj) / 2).astype(int)

    for ifield in range(nfield):
        tr_f[:,ifield] = gridfield[ifield][ielem + nearestcorner_di, jelem + nearestcorner_dj]
    
    return


def trac2grid(tr_x, tr_f, mesh, grid, gridfield, nx, distweight=None, avgscheme=None, method=INTERP_METHOD_ELEM):
    # NB! Designed for regular grids

    assert len(gridfield) == tr_f.shape[1]

    if avgscheme is None:
        avgscheme = [INTERP_AVG_ARITHMETIC for i in range(len(gridfield))]

    assert type(avgscheme) == type([])
    assert len(avgscheme) == len(gridfield)

    ntracfield = len(gridfield)
    ntrac = tr_x.shape[0]

    if method == INTERP_METHOD_GRIDDATA:
        if DIM != 2:
            print("!!! NOT IMPLEMENTED")
        
        gridval = griddata(tr_x, tr_f, (mesh[0], mesh[1]), 'nearest')

        for d in range(ntracfield):
            gridfield[d][:,:] = gridval[:,:,d]

    elif method == INTERP_METHOD_ELEM:
        # TODO: weight by distance to the grid point

        Lmax = [grid[i][-1] for i in range(DIM)]
        Lmin = [grid[i][0] for i in range(DIM)]
        L = [Lmax[i]-Lmin[i] for i in range(DIM)]

        ielem = np.floor((2*nx[IZ]-1) * (tr_x[:,IZ]-Lmin[IZ]) / L[IZ]).astype(int)
        jelem = np.floor((2*nx[IX]-1) * (tr_x[:,IX]-Lmin[IX]) / L[IX]).astype(int)

        traccount = np.zeros(2*np.array(mesh[0].shape))
        tracsum = np.zeros(2*np.array(mesh[0].shape))

        traccount[ielem,jelem] += 1
        traccount[ielem+1,jelem] += 1
        traccount[ielem,jelem+1] += 1
        traccount[ielem+1,jelem+1] += 1

        idx = traccount == 0
        if np.sum(idx) > 0:
            print("!!! Zero values in: ")
            print(np.where(idx))

        for ifield in range(ntracfield):
            tracsum[:,:] = 0

            if avgscheme[ifield] == INTERP_AVG_ARITHMETIC:
                tracsum[ielem, jelem] += tr_f[:,ifield]
                tracsum[ielem+1, jelem] += tr_f[:,ifield]
                tracsum[ielem, jelem+1] += tr_f[:,ifield]
                tracsum[ielem+1,jelem+1] += tr_f[:,ifield]
                gridfield[ifield][:,:] = tracsum[::2,::2] / traccount[::2,::2]
            elif avgscheme[ifield] == INTERP_AVG_GEOMETRIC:
                tracsum[ielem, jelem] += np.log(tr_f[:,ifield])
                tracsum[ielem+1, jelem] += np.log(tr_f[:,ifield])
                tracsum[ielem, jelem+1] += np.log(tr_f[:,ifield])
                tracsum[ielem+1,jelem+1] += np.log(tr_f[:,ifield])

                # if original value is zero, log return -inf... fix that
                tracsum[np.isinf(tracsum)] = 0

                gridfield[ifield][:,:] = np.exp(tracsum[::2,::2] / traccount[::2,::2])

            else:
                print ("!!! ERROR INVALID AVERAGING SCHEME")

        return

        
        iprev = -1
        jprev = -1
        iidx = np.array([])
        jidx = np.array([])

        for gridp in itertools.product(*(range(i) for i in nx[0:DIM])):
            i = gridp[0]
            j = gridp[1]

            if i != iprev:
                iidx = (tr_x[:,IZ] < grid[IZ][min(i+1,nx[IZ]-1)]) & (tr_x[:,IZ] > grid[IZ][max(i-1,0)])
                sys.stdout.write('.')
                sys.stdout.flush()

            if j != jprev:
                jidx = (tr_x[:,IX] < grid[IX][min(j+1,nx[IX]-1)]) & (tr_x[:,IX] > grid[IX][max(j-1,0)])

            idx = iidx & jidx

            if np.sum(idx) == 0:
                print ("! WARNING ", i, j, " is empty")
            for ifield in range(tr_f.shape[1]):
                if avgscheme[ifield] == INTERP_AVG_ARITHMETIC:
                    gridfield[ifield][i,j] = np.mean(tr_f[idx,ifield])
                elif avgscheme[ifield] == INTERP_AVG_GEOMETRIC:
                    gridfield[ifield][i,j] = np.exp(np.sum(np.log(tr_f[idx,ifield])) / np.sum(idx))

            iprev = i
            jprev = j


def RK(tr_x, grid, gridvel, nx, tstep, order=4):
    if order != 2 and order != 4:
        raise Exception("Sorry, don't know how to do that")

    print("Doing tracer advection")

    if order == 2:
        trac_vel = np.zeros((tr_x.shape[0], DIM))
        tracs_half_h = np.zeros((tr_x.shape[0], DIM))
        tracs_full_h = np.zeros((tr_x.shape[0], DIM))
        tracvel_half_h = np.zeros((tr_x.shape[0], DIM))
        grid2trac(tr_x, trac_vel, grid, gridvel, nx)
        for d in range(DIM):
            tracs_half_h[:,d] = tr_x[:,d] + 0.5 * tstep * trac_vel[:,d]
        grid2trac(tracs_half_h, tracvel_half_h, grid, gridvel, nx)
        for d in range(DIM):
            tracs_full_h[:,d] = tr_x[:,d] + tstep * tracvel_half_h[:,d]
        return trac_vel, tracs_full_h

    elif order == 4:
        k1vel = np.zeros((tr_x.shape[0], DIM))
        k2vel = np.zeros_like(k1vel)
        k3vel = np.zeros_like(k1vel)
        k4vel = np.zeros_like(k1vel)
        k2loc = np.zeros_like(k1vel)
        k3loc = np.zeros_like(k1vel)
        k4loc = np.zeros_like(k1vel)
        tr_x_final = np.zeros_like(k1vel)

        grid2trac(tr_x, k1vel, grid, gridvel, nx)
        for d in range(DIM):
            k2loc[:,d] = tr_x[:,d] + 0.5 * tstep * k1vel[:,d]
        grid2trac(k2loc, k2vel, grid, gridvel, nx)
        for d in range(DIM):
            k3loc[:,d] = tr_x[:,d] + 0.5 * tstep * k2vel[:,d]
        grid2trac(k3loc, k3vel, grid, gridvel, nx)
        for d in range(DIM):
            k4loc[:,d] = tr_x[:,d] + tstep * k3vel[:,d]
        grid2trac(k4loc, k4vel, grid, gridvel, nx)
        for d in range(DIM):
            tr_x_final[:,d] = tr_x[:,d] + (1/6)*tstep * (k1vel + k2vel + k3vel + k4vel)

        return k1vel, tr_x_final
