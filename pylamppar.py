#!/usr/bin/python3

#import math
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import axes3d
import scipy.sparse.linalg
import itertools
from mpi4py import MPI

def pprint(*arg):
    comm = MPI.COMM_WORLD
    iproc = comm.Get_rank()
    nproc = comm.Get_size()
    
    if iproc == 0:
        print(''.join(map(str,arg)))
    
    return

def derivScalar(field, x):
    """ return df/d[zxy..] as a list of numpy arrays
    """
    if type(field) != np.ndarray:
        raise Exception("must be numpy list")

    nx = field.shape
    dims = len(nx)
    
    if dims != len(x):
        raise Exception("incompatible dimensions")

    df = diffScalar(field)
    dx = diffVector(x)
    midp = midpScalar(x)
    return [df[dim]/dx[dim][dim] for dim in range(dims)], midp


def derivVector(field, x):
    if type(field) != list:
        raise Exception("must be list")
    
    dims = len(field)
    
    nx = field[0].shape
    for i in range(1,dims):
        if type(field[i]) != np.ndarray:
            raise Exception("must be numpy array")
        if field[i].shape != nx:
            raise Exception("dimensions mismatch")
    
    df = diffVector(field)
    dx = diffVector(x)

    ret = []

    for i in range(dims):
        ret.append([])
        for j in range(dims):
            ret[i].append([])
            ret[i][j] = df[i][j] / dx[j][j]

    return ret


def midpScalar(x):
    """ return midpoint value (=average) in each direction
    """
    if type(x) != np.ndarray:
        raise Exception("must be numpy array")
    nx = x.shape
    dim = len(nx)
    midpf = [[]] * dim
    
    for d in range(dim):
        krdelta = [0]*dim
        krdelta[d] = 1

        midpf[d] = 0.5*x[[slice(krdelta[dd],nx[dd]) for dd in range(dim)]]  +  0.5*x[[slice(0,nx[dd]-krdelta[dd]) for dd in range(dim)]]

    return midpf


def midpVector(x):
    """ return midpoint value (=average) in each direction
    """
    if type(x) != list:
        raise Exception("must be list")

    dim = len(x)

    #nx = x[0].shape
    for i in range(1,dim):
        if type(x[i]) != np.ndarray:
            raise Exception("must be numpy array")
        #if x[i].shape != nx:
        #    raise Exception("dimensions mismatch")

    avgx = []

    for ifield in range(dim):
        avgx.append([])
        avgx[ifield] = midpScalar(x[ifield])

    return avgx


def diffScalar(field):
    """ return d[field] in all directions

    field:  a n-dimensional numpy array

    return value is a list of numpy arrays (length
    equals to number of dimensions)
    """
    if type(field) != np.ndarray:
        raise Exception("must be numpy array")
    nx = field.shape
    dim = len(nx)
    df = [[]] * dim
    for d in range(dim):
        krdelta = [0]*dim
        krdelta[d] = 1

        df[d] = field[[slice(krdelta[dd],nx[dd]) for dd in range(dim)]]  -  field[[slice(0,nx[dd]-krdelta[dd]) for dd in range(dim)]]

    return df


def diffVector(field):
    """ return d[field] in all directions

    field:   a list of n-dimensional numpy arrays
             length of list must equal number of dimensions

    e.g. velocity field: list has arrays for u, v, w; function
    will return: * du in x, y, z directions
                 * dv in x, y, z directions
                 * dw in x, y, z directions

    return value is a list of list of numpy arrays
    (e.g. ret[1][2] == dv in z dir)
    """
    if type(field) != list:
        raise Exception("must be list")

    dim = len(field)

    nx = field[0].shape
    for i in range(1,dim):
        if type(field[i]) != np.ndarray:
            raise Exception("must be numpy array")
        if field[i].shape != nx:
            raise Exception("dimensions mismatch")

    df = []
    
    for d_numer in range(dim):
        df.append([])
        for d_denom in range(dim):
            df[d_numer].append([])
            krdelta = [0]*dim
            krdelta[d_denom] = 1
            df[d_numer][d_denom] = (field[d_numer])[[slice(krdelta[dd],NX[dd]) for dd in range(dim)]] - (field[d_numer])[[slice(0,NX[dd]-krdelta[dd]) for dd in range(dim)]]

    return df


def gidx(idxs, nx, dim):
    # Global index for the matrix in linear system of equations
    #   gidx = ...  +  iz * ny * nx * (DIM+1)  +  ix * ny * (DIM+1)  +  iy * (DIM+1)  +  ieq
    # So in general,  np.sum([idxs[i] * np.prod(NX[(i+1):DIM]) * (DIM+1) for i in range(DIM)]) + ieq
    if len(idxs) != dim:
        raise Exception("num of idxs != dimensions")
    return np.sum([idxs[i] * np.prod(nx[(i+1):dim]) * (dim+1) for i in range(dim)],dtype=np.int)


def idxsum(a, b):
    # just a sum of two vectors a and b, but returned as tuple
    # so that it can be used as an index for lists
    if len(a) != len(b):
        raise Exception("len a != len b")
    ret = tuple([a[i] + b[i] for i in range(len(a))])
    if True in [ret[i] < 0 for i in range(len(a))]:
        pprint("Warning!! idxsum returning values < 0")
    return ret


def krd(i, d, val=1, inv=False):
    ### "Kroenecker delta" array
    if not inv:
        krdelta = [0]*d
        krdelta[i] = val
    else:
        krdelta = [val]*d
        krdelta[i] = 0
    return krdelta

def trac2grid(tracs, tracvals, grid, gridfield, NX, DIM):
    # IDW
    comm = MPI.COMM_WORLD
    iproc = comm.Get_rank()
    nproc = comm.Get_size()
    
    proc_gridfield = np.zeros(gridfield.shape)

    for idx in itertools.product(*(range(0,i) for i in NX[0:DIM])):
        if idx[0] % nproc != iproc:
            continue
        distsq = np.sum(np.array([tracs[:,d] - grid[d][idx] for d in range(DIM)])**2, axis=0)
        distweight = 1.0 / distsq**2
        proc_gridfield[idx] = np.sum(tracvals * distweight) / np.sum(distweight)

    comm.Allreduce(proc_gridfield, gridfield, op=MPI.SUM)

def grid2trac(grid, gridfield, tracs, tracvals, NX, DIM, nfields=1):
    comm = MPI.COMM_WORLD
    iproc = comm.Get_rank()
    nproc = comm.Get_size()
    
    proc_tracvals = np.zeros(tracvals.shape)
    
    for itrac in range(iproc, tracs.shape[0], nproc):
        distsq = np.sum(np.array([tracs[itrac,d] - grid[d] for d in range(DIM)])**2, axis=0)
        
        distweight = 1.0 / distsq**2
            
        if nfields > 1:
            for ifield in range(nfields):
                proc_tracvals[itrac,ifield] = np.sum(gridfield[ifield] * distweight) / np.sum(distweight)
        else:
            proc_tracvals[itrac] = np.sum(gridfield * distweight) / np.sum(distweight)
    
    comm.Allreduce(proc_tracvals, tracvals, op=MPI.SUM)

def RK(grid, gridvel, tracs, h, NX, DIM, order=2):
    if order != 2:
        raise Exception("Sorry, don't know how to do that")

    comm = MPI.COMM_WORLD
    iproc = comm.Get_rank()
    nproc = comm.Get_size()

    pprint("Doing tracer advection")

    trac_vel = np.zeros((tracs.shape[0], DIM))
    tracs_half_h = np.zeros((tracs.shape[0], DIM))
    tracs_full_h = np.zeros((tracs.shape[0], DIM))
    tracvel_half_h = np.zeros((tracs.shape[0], DIM))
    
#    proc_trac_vel = np.zeros((tracs.shape[0], DIM))
#    proc_tracs_half_h = np.zeros((tracs.shape[0], DIM))
#    proc_tracs_full_h = np.zeros((tracs.shape[0], DIM))
#    proc_tracvel_half_h = np.zeros((tracs.shape[0], DIM))

    grid2trac(grid, gridvel, tracs, trac_vel, NX, DIM, nfields=DIM)
    
    for d in range(DIM):
#        for i in range(0, tracs.shape[0], nproc):
#            proc_tracs_half_h[i,d] = tracs[i,d] + 0.5 * h * trac_vel[i,d]
        tracs_half_h[:,d] = tracs[:,d] + 0.5 * h * trac_vel[:,d]
#    comm.Allreduce(proc_tracs_half_h, tracs_half_h, op=MPI.SUM)
    
    grid2trac(grid, gridvel, tracs_half_h, tracvel_half_h, NX, DIM, nfields=DIM)
    
    for d in range(DIM):
#        for i in range(0, tracs.shape[0], nproc):
#            proc_tracs_full_h[i,d] = tracs[i,d] + h * tracvel_half_h[i,d]
        tracs_full_h[:,d] = tracs[:,d] + h * tracvel_half_h[:,d]
#    comm.Allreduce(proc_tracs_full_h, tracs_full_h, op=MPI.SUM)
    
    return trac_vel, tracs_full_h


### START MAIN

comm = MPI.COMM_WORLD
iproc = comm.Get_rank()
nproc = comm.Get_size()

### Run control
DEBUGLEVEL = 0
SOLVEMETHOD = 'spsolve' # 'spsolve' or 'bicgstab' or 'inv'


### Geometry setup
AXES = ['x','z','y']   # one of these must be 'z'
DIM = 2
NX = [32,32,5]
TRACDENS = [64,64]
L = [660e3,660e3,1.]
if not 'z' in AXES[0:DIM]:
    raise Exception("one of the dimensions must be 'z'")
VERTICALDIM = np.where(np.array(AXES)=='z')[0][0]


### Constants
G = -9.81
SECINYR = 60.*60.*24.*365.25
TSTEP = SECINYR*2e5
MAXTIME = SECINYR*10.1e6

### Set up body forces
fb = [0.0 for d in range(DIM)]
fb[AXES.index('z')] = G

### And physical constants
mu = 1e20

### Form the (rectilinear) grid
axisgrids = [np.linspace(0,L[d],num=NX[d]) for d in range(DIM)]
grid = np.meshgrid(*(axisgrids[d] for d in range(DIM)), indexing='ij')

### Form velocity and pressure field arrays
vel = [np.zeros_like(grid[d]) for d in range(DIM)]
pres = np.zeros_like(grid[0])

rho = np.zeros_like(grid[0])

### Populate tracers
tracs = np.random.rand(*(np.prod(TRACDENS[0:DIM]), DIM))
for d in range(DIM):
    tracs[:, d] = tracs[:, d] * L[d]

### Insert densities to tracers
trac_rho = np.ones(np.prod(TRACDENS[0:DIM])) * 3300.
idxx = (tracs[:, AXES.index('x')] < 400e3) & (tracs[:, AXES.index('x')] > 300e3)
idxz = tracs[:, AXES.index('z')] < 100e3
trac_rho[idxx & idxz] = 3350.

current_time = 0
current_tstep = 0
while current_time < MAXTIME:
    pprint("=== tstep: ", current_tstep, " ===")
    pprint("Interpolating trac2grid")
    trac2grid(tracs, trac_rho, grid, rho, NX, DIM)

    ### Form sparse matrix, linear system of equations
    dof = (DIM+1) * np.prod(NX[0:DIM])  # DIM+1 <- one eq per dimension + continuity eq
    A = scipy.sparse.lil_matrix((dof, dof))
    b = np.zeros(dof)

    DONUMSCALING = True
    if DONUMSCALING:
        Kcont = 2.0 * mu / np.sum([L[d]/NX[d] for d in range(DIM)])
        Kbond = 4.0 * mu / np.sum([L[d]/NX[d] for d in range(DIM)])**2.0
    else:
        Kcont = 1.0
        Kbond = 1.0

    pprint ("Building matrix")
    for idx in itertools.product(*(range(1,(i-1)) for i in NX[0:DIM])):  # i.e. nested for loops for each dimension "i,j,k,..."
        # idx gets values [0,0,0] ; [0,0,1] ; [0,0,2] ; ... ; [1,0,0] ; [1,0,1] ; etc. up to [NX[0]-1,NX[1]-1,NX[2]-1]
        
        for ieq in range(DIM):
            # stokes dir 0,1,2,...
            # NB we assume ieq=0 -> stokes in dir 0
            #              ieq=1 -> stokes in dir 1, etc.
            mat_row = gidx(idx, NX, DIM) + ieq

            midcoordsum = 0.0 # sum of terms at i,j,k,...
            for d in range(DIM):
                # "idxsum(idx, krd(d, DIM, -1)" : add to index (idx) in dimension d value -1

                # stokes
                cidx = gidx(idxsum(idx, krd(d, DIM, -1)), NX, DIM) + ieq
                A[mat_row, cidx] = mu * 2.0 / ( (grid[d][idxsum(idx,krd(d, DIM, 0))] - grid[d][idxsum(idx,krd(d, DIM, -1))]) * (grid[d][idxsum(idx,krd(d, DIM, 1))] - grid[d][idxsum(idx,krd(d, DIM, -1))]) )
                cidx = gidx(idxsum(idx, krd(d, DIM,  1)), NX, DIM) + ieq
                A[mat_row, cidx] = mu * 2.0 / ( (grid[d][idxsum(idx,krd(d, DIM, 1))] - grid[d][idxsum(idx,krd(d, DIM,  0))]) * (grid[d][idxsum(idx,krd(d, DIM, 1))] - grid[d][idxsum(idx,krd(d, DIM, -1))]) )

                midcoordsum += -1.0 / ( (grid[d][idxsum(idx,krd(d, DIM, 1))]-grid[d][idxsum(idx,krd(d, DIM,  0))]) * (grid[d][idxsum(idx,krd(d, DIM, 1))]-grid[d][idxsum(idx,krd(d, DIM, -1))]) )
                midcoordsum += -1.0 / ( (grid[d][idxsum(idx,krd(d, DIM, 0))]-grid[d][idxsum(idx,krd(d, DIM, -1))]) * (grid[d][idxsum(idx,krd(d, DIM, 1))]-grid[d][idxsum(idx,krd(d, DIM, -1))]) )

            # stokes, u_i_j_k
            cidx = gidx(idx, NX, DIM) + ieq
            A[mat_row, cidx] = mu * 2.0 * midcoordsum

            # p_i+1_j : -1 / (x_i+1_j - x_i-1_j)
            # NB ieq in here (grid[ieq] and krd(ieq,...)) and in next pressure term is used to
            # determine the direction of the stokes eq
            cidx = gidx(idxsum(idx, krd(ieq, DIM, 1)), NX, DIM) + DIM  # DIM: stokes eq directions + 1 (pressure term)
            A[mat_row, cidx] = -Kcont / (grid[ieq][idxsum(idx,krd(ieq, DIM, 1))]-grid[ieq][idxsum(idx,krd(ieq, DIM, -1))])

            # p_i-1_j :  1 / (x_i+1_j - x_i-1_j)
            cidx = gidx(idxsum(idx, krd(ieq, DIM, -1)), NX, DIM) + DIM
            A[mat_row, cidx] =  Kcont / (grid[ieq][idxsum(idx,krd(ieq, DIM, 1))]-grid[ieq][idxsum(idx,krd(ieq, DIM, -1))])

            # body force
            # again, ieq is used to determine the direction of the stokes equation
            b[mat_row] = fb[ieq] * rho[idx]

        # " ieq == dimensions + 1" (i.e. after stokes eqs), continuity
        ieq = DIM
        mat_row = gidx(idx, NX, DIM) + ieq
        for d in range(DIM):
            cidx = gidx(idxsum(idx, krd(d, DIM, 1)), NX, DIM) + d
            A[mat_row, cidx] = Kcont / (grid[d][idxsum(idx,krd(d, DIM, 1))]-grid[d][idxsum(idx,krd(d, DIM, -1))])
            cidx = gidx(idxsum(idx, krd(d, DIM, -1)), NX, DIM) + d
            A[mat_row, cidx] = -Kcont / (grid[d][idxsum(idx,krd(d, DIM, 1))]-grid[d][idxsum(idx,krd(d, DIM, -1))])

        b[mat_row] = 0.0

    # one pressure value
    ieq = DIM
    mat_row = gidx([1]*DIM, NX, DIM) + ieq
    cidx = mat_row
    #A[mat_row, :] = 0.0
    A[mat_row, cidx] += 1.0
    #b[mat_row] = 0.0
    b[mat_row] += 1.0

    # Boundary conditions
    pprint("Setting boundary conditions")
    for idx in itertools.product(*(range(0,i) for i in NX[0:DIM])):
        isbnd = [0]*2*DIM
        nbnd = 0
        for d in range(DIM):
            if idx[d] == 0:
                isbnd[2*d] = 1
                nbnd += 1
            elif idx[d] == NX[d]-1:
                isbnd[2*d+1] = 1
                nbnd += 1

        if nbnd == 0:
            continue

        for ieq in range(DIM):
            # fixed vel
            mat_row = gidx(idx, NX, DIM) + ieq
            cidx = mat_row
            A[mat_row, cidx] = 1.0 * Kbond
            b[mat_row] = 0.0 * Kbond

    #        ## free-slip    **** UNFINISHED **** TODO ****
    #        mat_row = gidx(idx, NX, DIM) + ieq
    #        if nbnd == DIM:
    #            # corner, fix all
    #            cidx = mat_row
    #            A[mat_row, cidx] = 1.0
    #            b[mat_row] = 0.0
    #        elif isbnd[ieq] > 0:
    #            cidx = mat_row
    #            A[mat_row, cidx] = 1.0
    #            b[mat_row] = 0.0
    #        else:
    #            cidx = mat_row
    #            A[mat_row, cidx] = 1.0
    #            cidx = gidx(sumidx(idx, krd(ieq, DIM, 1)), NX, DIM) + ieq
    #            A[mat_row, cidx] = -1.0

        mat_row = gidx(idx, NX, DIM) + DIM

        for d in range(DIM):
            if nbnd == DIM and DIM > 1:
                # corner point
                # in corners
                # P_i_j_k - (1/(DIM-1))*P_i_j-1_k - (1/(DIM-1))*P_i_j_k-1 = 0
                # i.e. horizontal symmetry for pressure
                if d == VERTICALDIM:
                    cidx = gidx(idxsum(idx, krd(d, DIM, 0)), NX, DIM) + DIM
                    A[mat_row, cidx] = Kbond * 1.0
                else:
                    if isbnd[2*d] == 1:
                        cidx = gidx(idxsum(idx, krd(d, DIM, 1)), NX, DIM) + DIM
                    elif isbnd[2*d+1] == 1:
                        cidx = gidx(idxsum(idx, krd(d, DIM, -1)), NX, DIM) + DIM
                    A[mat_row, cidx] = -Kbond*1.0/(DIM-1.0)
            elif isbnd[2*d] == 1:
                # left bnd in dir d
                cidx = gidx(idxsum(idx, krd(d, DIM, 1)), NX, DIM) + d
                A[mat_row, cidx] = Kbond / (2.0*(grid[d][idxsum(idx,krd(d, DIM, 1))]-grid[d][idxsum(idx,krd(d, DIM, 0))]))
                #cidx = gidx(idxsum(idx, krd(d, DIM, 0)), NX, DIM) + d
                #A[mat_row, cidx] = 1.0
            elif isbnd[2*d+1] == 1:
                # right bnd in dir d
                cidx = gidx(idxsum(idx, krd(d, DIM, -1)), NX, DIM) + d
                A[mat_row, cidx] = -Kbond / (2.0*(grid[d][idx]-grid[d][idxsum(idx,krd(d, DIM, -1))]))
                #cidx = gidx(idxsum(idx, krd(d, DIM, 0)), NX, DIM) + d
                #A[mat_row, cidx] = 1.0
            else:
                # not at bnd in dir d
                cidx = gidx(idxsum(idx, krd(d, DIM, 1)), NX, DIM) + d
                A[mat_row, cidx] = Kbond / (grid[d][idxsum(idx,krd(d, DIM, 1))]-grid[d][idxsum(idx,krd(d, DIM, -1))])
                cidx = gidx(idxsum(idx, krd(d, DIM, -1)), NX, DIM) + d
                A[mat_row, cidx] = -Kbond / (grid[d][idxsum(idx,krd(d, DIM, 1))]-grid[d][idxsum(idx,krd(d, DIM, -1))])
                #cidx = gidx(idxsum(idx, krd(d, DIM, 0)), NX, DIM) + d
                #A[mat_row, cidx] = 1.0
        b[mat_row] = 0.0 * Kbond


    pprint ("Solving stokes")
    if SOLVEMETHOD == 'spsolve':
        t0 = time.time()
        x = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A),b)
        t1 = time.time()
        err = np.nan
    elif SOLVEMETHOD == 'bicgstab':
        t0 = time.time()
        x = scipy.sparse.linalg.bicgstab(A,b)
        t1 = time.time()
        err = x[1]
        x = x[0]
    elif SOLVEMETHOD == 'inv':
        #Ax=b => A^-1 A x = x = A^-1 b
        t0 = time.time()
        Ainv = np.linalg.inv(A.todense())
        t1 = time.time()
        x = np.array(np.dot(Ainv, b))[0]
    else:
        raise Exception("Unknown method '" + SOLVEMETHOD + "'")
    pprint("Solved in time: ", t1-t0)

    newvel = [[]] * DIM
    for d in range(DIM):
        newvel[d] = x[range(d,dof,DIM+1)].reshape(*tuple(NX[0:DIM]))
    newpres = x[range(DIM,dof,DIM+1)].reshape(NX[0:DIM])


    #print ("Interpolating grid2trac")
    #trac_vel = np.zeros((np.prod(TRACDENS[0:DIM]),DIM))
    #for d in range(DIM):
    #    tracveld = np.zeros(trac_vel.shape[0])
    #    grid2trac(grid, newvel[d], tracs, tracveld, NX, DIM)
    #    trac_vel[:,d] = tracveld

    trac_vel, tracs_new = RK(grid, newvel, tracs, TSTEP, NX, DIM)
    rho_new = np.zeros_like(rho)
    trac2grid(tracs_new, trac_rho, grid, rho_new, NX, DIM)

    if iproc == 0:
        plt.close('all')
        fig = plt.figure()
        ax = fig.add_subplot(221) #, projection='3d')
        #ax.quiver(grid[0], grid[1], newvel[0], newvel[1])
        ax.quiver(tracs[0:trac_vel.shape[0]:10,0], tracs[0:trac_vel.shape[0]:10,1], trac_vel[0:trac_vel.shape[0]:10,0], trac_vel[0:trac_vel.shape[0]:10,1])
        ax = fig.add_subplot(222)
        CS=ax.contourf(grid[0], grid[1], rho_new)
        plt.colorbar(CS)

        # check:
        # to calculate divergence of the velocity field
        # du/dx + dv/dy
        velDeriv = derivVector(newvel, grid) # all derivatives in all directions
        dudx = velDeriv[0][0]  # Note, du/dx and dv/dy here are
        dvdy = velDeriv[1][1]  # defined at different locations
        dudx = midpScalar(dudx)[1] # midpoint in y dir for x-derivatives
        dvdy = midpScalar(dvdy)[0] # midpoint in x dir for y-derivativex
                            # Now dudx and dvdy are defined at same locations
        midp_grid = midpVector(grid)
        midp_x = midpScalar(midp_grid[0][0])[1]
        midp_y = midpScalar(midp_grid[1][0])[1]

        divfield = dudx + dvdy

        ax = fig.add_subplot(223)
        CS=ax.contourf(midp_x, midp_y, divfield)
        plt.colorbar(CS)


        ax = fig.add_subplot(224)
        CS=ax.contourf(grid[0], grid[1], rho)
        plt.colorbar(CS)

        #plt.show()
        plt.savefig("fig_"+str(current_tstep))

    # Copy fields for next time step
    tracs = np.copy(tracs_new)
    rho = np.copy(rho_new)

    current_time += TSTEP
    current_tstep += 1

### DISCRETIZED EQUATIONS, 2nd try
# Stokes flow
# mu ( 2 [ (u_i+1_j - u_i_j) / (x_i+1_j - x_i_j)(x_i+1_j - x_i-1_j) - (u_i_j - u_i-1_j) / (x_i_j - x_i-1_j)(x_i+1_j - x_i-1_j) ]   + 
#      2 [ (u_i_j+1 - u_i_j) / (y_i_j+1 - y_i_j)(y_i_j+1 - y_i_j-1) - (u_i_j - u_i_j-1) / (y_i_j - y_i_j-1)(y_i_j+1 - y_i_j-1) ] ) +
# -(p_i+1_j - p_i-1_j) / (x_i+1_j - x_i-1_j)   + 
# f_x_i_j
# = 0
# Coefficients:
# u_i-1_j_k : 2 mu / (x_i_j - x_i-1_j)(x_i+1_j - x_i-1_j)
# u_i+1_j_k : 2 mu / (x_i+1_j - x_i_j)(x_i+1_j - x_i-1_j)
# u_i_j-1_k : 2 mu / (y_i_j - y_i_j-1)(y_i_j+1 - y_i_j-1)
# u_i_j+1_k : 2 mu / (y_i_j+1 - y_i_j)(y_i_j+1 - y_i_j-1)
# u_i_j_k-1 : ...
# u_i_j_k+1 : ...
# u_i_j   : -2 mu / (x_i+1_j - x_i_j)(x_i+1_j - x_i-1_j) + -2 mu / (x_i_j - x_i-1_j)(x_i+1_j - x_i-1_j) + -2 mu y_i_j)(y_i_j+1 - y_i_j-1) + -2 mu / (y_i_j - y_i_j-1)(y_i_j+1 - y_i_j-1)
# p_i+1_j : -1 / (x_i+1_j - x_i-1_j)
# p_i-1_j :  1 / (x_i+1_j - x_i-1_j)
# RHS: body force



### DISCRETIZED EQUATIONS, 1st try
# Stokes flow
# mu (2(u_i+1_j - 2u_i_j + u_i-1_j) / (x_i+1_j - x_i-1_j)^2 + 2(u_i_j+1 - 2u_i_j + u_i_j-1) / (y_i_j+1 - y_i_j-1)^2) -
# (p_i+1_j - p_i-1_j) / (x_i+1_j - x_i-1_j) + f_x_i_j = 0
#
# coefficients, vel u equation:
# for location i,j:
# u_i-1_j_k : mu 2 / (x_i+1_j - x_i-1_j)^2
# u_i+1_j_k : mu 2 / (x_i+1_j - x_i-1_j)^2
# u_i_j-1_k : mu 2 / (y_i_j+1 - y_i_j-1)^2
# u_i_j+1_k : mu 2 / (y_i_j+1 - y_i_j-1)^2
# u_i_j_k-1 : mu 2 / (z_i_j_k+1 - z_i_j_k-1)^2  (??)
# u_i_j_k+1 : mu 2 / (z_i_j_k+1 - z_i_j_k-1)^2  (??)
# u_i_j   : mu 2 (-2 / (x_i+1_j - x_i-1_j)^2 - 2 / (y_i_j+1 - y_i_j-1)^2 - 2 / (z_i_j_k+1 - z_i_j_k-1)^2))
# p_i+1_j : -1 / (x_i+1_j - x_i-1_j)
# p_i-1_j :  1 / (x_i+1_j - x_i-1_j)
# RHS: body force

# coefficients, vel y equation:
# for location i,j:
# v_i-1_j : mu 2 / (x_i+1_j - x_i-1_j)^2
# v_i+1_j : mu 2 / (x_i+1_j - x_i-1_j)^2
# v_i_j-1 : mu 2 / (y_i_j+1 - y_i_j-1)^2)
# v_i_j+1 : mu 2 / (y_i_j+1 - y_i_j-1)^2)
# v_i_j   : mu 2 (-2 / (x_i+1_j - x_i-1_j)^2 - 2 / (y_i_j+1 - y_i_j-1)^2)
# p_i+1_j : -1 / (y_i_j+1 - y_i_j-1)
# p_i-1_j :  1 / (y_i_j+1 - y_i_j-1)
# RHS: body force

# coefficients, "pressure equation" (continuity):
#     du/dx + dv/dy = 0
#     (u_i+1_j - u_i-1_j) / (x_i+1_j - x_i-1_j) + (v_i_j+1 - v_i_j-1) / (y_i_j+1 - y_i_j-1) = 0
# u_i+1_j :  1 / (x_i+1_j - x_i-1_j)
# u_i-1_j : -1 / (x_i+1_j - x_i-1_j)
# v_i_j+1 :  1 / (y_i_j+1 - y_i_j-1)
# v_i_j-1 : -1 / (y_i_j+1 - y_i_j-1)
# 
# Boundary condition examples, fixed vel
# left bnd in x dir:
# (u_i+1_j) / (2*x_i+1_j) + (v_i_j+1 - v_i_j-1) / (y_i_j+1 - y_i_j-1) = 0
# u_i+1_j :  1 / (2*x_i+1_j)
# u_i-1_j :  0
# v_i_j+1 :  1 / (y_i_j+1 - y_i_j-1)
# v_i_j-1 : -1 / (y_i_j+1 - y_i_j-1)
#
# right bnd in x dir:
#     (0 - u_i-1_j) / (2*(x_i_j - x_i-1_j)) + (v_i_j+1 - v_i_j-1) / (y_i_j+1 - y_i_j-1) = 0
# u_i+1_j :  0
# u_i-1_j : -1 / (2*(x_i_j - x_i-1_j))
# v_i_j+1 :  1 / (y_i_j+1 - y_i_j-1)
# v_i_j-1 : -1 / (y_i_j+1 - y_i_j-1)








sys.exit()


### START TEST SNIPPETS

### Generate test data for pres and vel
if DIM == 3:
    vel = [grid[(d+1)%3] for d in range(DIM)]
if DIM == 2:
    pres = np.sin(grid[0]) + np.sin(grid[1])
    #vel[0] = np.sin(grid[0]) + np.sin(grid[1])
    #vel[1] = np.cos(grid[0]) + np.cos(grid[1])
    vel[0] = grid[1]
    vel[1] = grid[0]

    #plt.imshow(vel[0])
    #plt.show()
    #plt.imshow(vel[1])
    #plt.show()
elif DIM == 1:
    pres = np.sin(grid[0])
    vel[0] = np.sin(grid[0])

    #plt.plot(vel[0])
    #plt.show()



### Boundary conditions
#???


### Form dp arrays in all directions
dp = diffScalar(pres)

### Form d[zxy..] and d[uvw..] arrays in all directions
dx = diffVector(grid)
dv = diffVector(vel)


# to calculate divergence of the velocity field
# du/dx + dv/dy
velDeriv = derivVector(vel, grid) # all derivatives in all directions
dudx = velDeriv[0][0]  # Note, du/dx and dv/dy here are
dvdy = velDeriv[1][1]  # defined at different locations
dudx = midpScalar(dudx)[1] # midpoint in y dir for x-derivatives
dvdy = midpScalar(dvdy)[0] # midpoint in x dir for y-derivativex
                       # Now dudx and dvdy are defined at same locations
midp_grid = midpVector(grid)
midp_x = midpScalar(midp_grid[0][0])[1]
midp_y = midpScalar(midp_grid[1][0])[1]

divfield = dudx + dvdy


plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(grid[0], grid[1], grid[2], vel[0], vel[1], vel[2])

plt.show()

#plt.subplot(ax[1,0])
#cs = ax[1,0].contourf(midp_x, midp_y, dudx)
#fig.colorbar(cs)

#plt.subplot(ax[0,1])
#cs = ax[0,1].contourf(midp_x, midp_y, dvdy)
#fig.colorbar(cs)

#plt.subplot(ax[1,1])
#cs = ax[1,1].contourf(midp_x, midp_y, divfield)
#fig.colorbar(cs)
plt.show()

#fig, ax = plt.subplots(1,2)
#cs = ax[0].contourf(x_midp[0], x_midp[1], divfield)
#fig.colorbar(cs, ax=ax[0], format="%.2f")
#plt.show()

#if DIM == 2:
#    plt.imshow(dvdx[0][0])
#    plt.show()
#elif DIM == 1:
#    print(dvdx[0])
#    print(midpScalar(grid[0]))
#    plt.plot(midpScalar(grid[0])[0], dvdx[0][0])
#    plt.show()






