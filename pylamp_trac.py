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

def trac2grid(tr_x, tr_f, mesh, grid, gridfield, nx, distweight=None, avgscheme=None, method=INTERP_METHOD_ELEM):

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

    # IDW

    #if distweight is None:
    #    for idx in itertools.product(*(range(0,i) for i in nx[0:DIM])):
    #        distsq = np.sum(np.array([tr_x[:,d] - mesh[d][idx] for d in range(DIM)])**2, axis=0)
    #        distweight = 1.0 / distsq**2

    #for idx in itertools.product(*(range(0,i) for i in nx[0:DIM])):
    #    if distweight is None:
    #        
    #    if avgscheme == IDW_AVG_ARITHMETIC:
    #        gridfield[idx] = np.sum(tr_f * distweight) / np.sum(distweight)
    #    elif avgscheme == IDW_AVG_GEOMETRIC:
    #        gridfield[idx] = np.exp(np.sum(distweight * np.log(tr_f)) / np.sum(distweight))
