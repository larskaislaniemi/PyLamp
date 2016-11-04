#!/usr/bin/python3

DEBUG = 6


DIM = 2         # 2 or 3, currently only 2 implemented (partly...)


IZ  = 0         # indices for different axes, this should probably
                # be compatible with python array indexing (rows, cols, etc),
                # i.e. zxy
IX  = 1
IY  = 2


IP  = DIM


# Physical constants
G = [9.81, 0.0]

# Tracer constants
NFTRAC = 2      # number of tracer functions
TR_RHO = 0      # indices for tracer functions: density
TR_ETA = 1      # viscosity
