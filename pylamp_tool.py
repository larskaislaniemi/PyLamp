#!/usr/bin/python3

from mpi4py import MPI
import time

prevtime = None

def pprint(*args):
    global prevtime
    
    if prevtime is None:
        prevtime = time.clock()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank ==0:
        nowtime = time.clock()
        timelapsed = 1000.0 * (nowtime-prevtime)
        timelapsedstr = "{0:5d}".format(int(timelapsed))
        nowtimestr = "{0:6d}".format(int(1000.0*nowtime))
        prevtime = nowtime
        print("[" + nowtimestr + " | " + timelapsedstr + "]", *args)


