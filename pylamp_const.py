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
SECINYR = 60*60*24*365.25
SECINKYR = SECINYR * 1e3
SECINMYR = SECINYR * 1e6


# Tracer constants
NFTRAC = 8      # number of tracer functions
TR_RHO = 0      # indices for tracer functions: density
TR_ETA = 1      # viscosity
TR_MRK = 2      # passive marker
TR_TMP = 3      # temperature
TR_HCD = 4      # heat conductivity
TR_HCP = 5      # heat capacity
TR_IRH = 6      # inherent density
TR_ALP = 7      # coefficient of thermal expansion


# Numerical constants
EPS = 2**(-10)
