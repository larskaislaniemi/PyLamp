#!/usr/bin/python3

import time

prevtime = None

def pprint(*args):
    global prevtime
    
    if prevtime is None:
        prevtime = time.clock()

    nowtime = time.clock()
    timelapsed = 1000.0 * (nowtime-prevtime)
    timelapsedstr = "{0:5d}".format(int(timelapsed))
    nowtimestr = "{0:6d}".format(int(1000.0*nowtime))
    prevtime = nowtime
    print("[" + nowtimestr + " | " + timelapsedstr + "]", *args)


