#!/usr/bin/python3

INTERP_AVG_ARITHMETIC = 0
INTERP_AVG_GEOMETRIC = 1

INTERP_METHOD_IDW = 0
INTERP_METHOD_GRIDDATA = 1
INTERP_METHOD_ELEM = 2
INTERP_METHOD_NEAREST = 3
INTERP_METHOD_LINEAR = 4
INTERP_METHOD_VELDIV = 5

from pylamp_const import *
import numpy as np
from scipy.interpolate import griddata
import itertools
import sys

def grid2trac(tr_x, tr_f, grid, gridfield, nx, defval=np.nan, method=INTERP_METHOD_LINEAR):
    # Interpolate values (gridfield) from grid to tracer
    # value (tr_f). Usually, the tracer value is its velocity.

    # NB! Designed for regular grids!

    assert len(gridfield) == tr_f.shape[1]
    assert method == INTERP_METHOD_LINEAR or method == INTERP_METHOD_NEAREST or\
            method == INTERP_METHOD_VELDIV

    nfield = len(gridfield)

    Lmax = [grid[i][-1] for i in range(DIM)]
    Lmin = [grid[i][0] for i in range(DIM)]
    L = [Lmax[i]-Lmin[i] for i in range(DIM)]

    ielem = np.floor((nx[IZ]-1) * (tr_x[:,IZ]-Lmin[IZ]) / L[IZ]).astype(int)
    jelem = np.floor((nx[IX]-1) * (tr_x[:,IX]-Lmin[IX]) / L[IX]).astype(int)

    idxdefval = (tr_x[:,IZ] < Lmin[IZ]) | (tr_x[:,IZ] > Lmax[IZ]) | \
            (tr_x[:,IX] < Lmin[IX]) | (tr_x[:,IX] > Lmax[IX])

    ielem[idxdefval] = 0
    jelem[idxdefval] = 0

    ntrac = tr_x.shape[0]

    # distances of tra
    distx = np.zeros((ntrac,4))
    distz = np.zeros((ntrac,4))

    for di in [0,1]:
        for dj in [0,1]:
            icorner = di * 2 + dj
            distz[:,icorner] = np.abs(tr_x[:,IZ] - grid[IZ][ielem+di])
            distx[:,icorner] = np.abs(tr_x[:,IX] - grid[IX][jelem+dj])

    if method == INTERP_METHOD_NEAREST:
        disttot = distz**2 + distx**2
        nearestcorner = np.argmin(disttot, axis=1)
        nearestcorner_dj = (nearestcorner % 2).astype(int)
        nearestcorner_di = ((nearestcorner - nearestcorner_dj) / 2).astype(int)

    for ifield in range(nfield):
        if method == INTERP_METHOD_NEAREST:
            tr_f[:,ifield] = gridfield[ifield][ielem + nearestcorner_di, jelem + nearestcorner_dj]
        elif method == INTERP_METHOD_LINEAR:
            xavg1 = gridfield[ifield][ielem + 0, jelem + 0] * distx[:, 1] + gridfield[ifield][ielem + 0, jelem + 1] * distx[:, 0]
            xavg1 = xavg1 / (distx[:, 0] + distx[:, 1])
            xavg2 = gridfield[ifield][ielem + 1, jelem + 0] * distx[:, 3] + gridfield[ifield][ielem + 1, jelem + 1] * distx[:, 2]
            xavg2 = xavg2 / (distx[:, 1] + distx[:, 2])
            zavg1 = xavg1 * distz[:,2] + xavg2 * distz[:,0]
            zavg1 = zavg1 / (distz[:,2] + distz[:,0])
            tr_f[:,ifield] = zavg1[:]
        elif method == INTERP_METHOD_VELDIV:
            # See Wang et al 2015 (or Meyer and Jenny 2004)
            # 2nd order bilinear interpolation

            # Only 2D implemented!

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
            distx[:, 0] = distx[:, 0] / (distx[:, 0] + distx[:, 1])
            #distx[:, 1] = distx[:, 1] / (distx[:, 0] + distx[:, 1]) 
            #distx[:, 2] = distx[:, 2] / (distx[:, 2] + distx[:, 3])
            #distx[:, 3] = distx[:, 3] / (distx[:, 2] + distx[:, 3])
            distz[:, 0] = distz[:, 0] / (distz[:, 0] + distz[:, 2])
            #distz[:, 1] = distz[:, 1] / (distz[:, 1] + distz[:, 3])
            #distz[:, 2] = distz[:, 2] / (distz[:, 0] + distz[:, 2])
            #distz[:, 3] = distz[:, 3] / (distz[:, 1] + distz[:, 3])

            Ux = \
                    (1-distx[:,0]) * (1-distz[:,0]) * gridfield[IX][ielem + 0, jelem + 0] + \
                    distx[:,0] * (1-distz[:,0]) * gridfield[IX][ielem + 0, jelem + 1] + \
                    (1-distx[:,0]) * distz[:,0] * gridfield[IX][ielem + 1, jelem + 0] + \
                    distx[:,0] * distz[:,0] * gridfield[IX][ielem + 1, jelem + 1]
            Uz = \
                    (1-distx[:,0]) * (1-distz[:,0]) * gridfield[IZ][ielem + 0, jelem + 0] + \
                    distx[:,0] * (1-distz[:,0]) * gridfield[IZ][ielem + 0, jelem + 1] + \
                    (1-distx[:,0]) * distz[:,0] * gridfield[IZ][ielem + 1, jelem + 0] + \
                    distx[:,0] * distz[:,0] * gridfield[IZ][ielem + 1, jelem + 1]


            # correction terms
            C10 = (0.5 * distx[:,0] / distz[:,0]) * \
                    (gridfield[IZ][ielem + 0, jelem + 0] - gridfield[IZ][ielem + 1, jelem + 0] +\
                    gridfield[IZ][ielem + 1, jelem + 1] - gridfield[IZ][ielem + 0, jelem + 1])
            C20 = (0.5 * distz[:,0] / distx[:,0]) * \
                    (gridfield[IX][ielem + 0, jelem + 0] - gridfield[IX][ielem + 0, jelem + 1] +\
                    gridfield[IX][ielem + 1, jelem + 1] - gridfield[IX][ielem + 1, jelem + 0])
            dU1 = distx[:,0] * (1-distx[:,0]) * C10
            dU2 = distz[:,0] * (1-distz[:,0]) * C20

            tr_f[:,IX] = Ux[:] + dU1[:]
            tr_f[:,IZ] = Uz[:] + dU2[:]

        tr_f[idxdefval,ifield] = defval

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

        zeroidx = traccount == 0
        #if np.sum(zeroidx) > 0:
        #    print("!!! Zero values in: ")
        #    print(np.where(zeroidx))

        zeroelemidx = zeroidx[0::2,0::2] + zeroidx[1::2,0::2] + \
                zeroidx[0::2,1::2] + zeroidx[1::2,1::2]
        nonzeroelemidx = np.invert(zeroelemidx > 0)

        if np.sum(zeroelemidx) > 0:
            print("!!! Zero values in elems: ")
            print(np.where(zeroelemidx))

        for ifield in range(ntracfield):
            tracsum[:,:] = 0

            if avgscheme[ifield] == INTERP_AVG_ARITHMETIC:
                tracsum[ielem, jelem] += tr_f[:,ifield]
                tracsum[ielem+1, jelem] += tr_f[:,ifield]
                tracsum[ielem, jelem+1] += tr_f[:,ifield]
                tracsum[ielem+1,jelem+1] += tr_f[:,ifield]
                newgridfield = tracsum[::2,::2] / traccount[::2,::2]
            elif avgscheme[ifield] == INTERP_AVG_GEOMETRIC:
                tracsum[ielem, jelem] += np.log(tr_f[:,ifield])
                tracsum[ielem+1, jelem] += np.log(tr_f[:,ifield])
                tracsum[ielem, jelem+1] += np.log(tr_f[:,ifield])
                tracsum[ielem+1,jelem+1] += np.log(tr_f[:,ifield])

                # if original value is zero, log return -inf... fix that
                tracsum[np.isinf(tracsum)] = 0

                newgridfield = np.exp(tracsum[::2,::2] / traccount[::2,::2])

            else:
                print ("!!! ERROR INVALID AVERAGING SCHEME")

            gridfield[ifield][nonzeroelemidx] = newgridfield[nonzeroelemidx]

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


def RK(tr_x, grid, gridvel, nx, tstep, order=2):
    if order != 2 and order != 4:
        raise Exception("Sorry, don't know how to do that")

    if order == 2:
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
        k1vel = np.zeros((tr_x.shape[0], DIM))
        k2vel = np.zeros_like(k1vel)
        k3vel = np.zeros_like(k1vel)
        k4vel = np.zeros_like(k1vel)
        k2loc = np.zeros_like(k1vel)
        k3loc = np.zeros_like(k1vel)
        k4loc = np.zeros_like(k1vel)
        tr_x_final = np.zeros_like(k1vel)

        grid2trac(tr_x, k1vel, grid, gridvel, nx, defval=0, method=INTERP_METHOD_VELDIV)
        for d in range(DIM):
            k2loc[:,d] = tr_x[:,d] + 0.5 * tstep * k1vel[:,d]
        grid2trac(k2loc, k2vel, grid, gridvel, nx, defval=0, method=INTERP_METHOD_VELDIV)
        for d in range(DIM):
            k3loc[:,d] = tr_x[:,d] + 0.5 * tstep * k2vel[:,d]
        grid2trac(k3loc, k3vel, grid, gridvel, nx, defval=0, method=INTERP_METHOD_VELDIV)
        for d in range(DIM):
            k4loc[:,d] = tr_x[:,d] + tstep * k3vel[:,d]
        grid2trac(k4loc, k4vel, grid, gridvel, nx, defval=0, method=INTERP_METHOD_VELDIV)
        for d in range(DIM):
            tr_x_final[:,d] = tr_x[:,d] + (1/6)*tstep * (k1vel[:,d] + k2vel[:,d] + k3vel[:,d] + k4vel[:,d])

        return k1vel, tr_x_final
