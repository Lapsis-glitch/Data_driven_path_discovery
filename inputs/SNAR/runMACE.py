#!/usr/bin/env python

from sys import argv as sargv
from sys import exit 
import subprocess as sp
import time
import os

inputFilename = sargv[1]
open('rdy_namd','w').close()

fname = 'rdy_mace'
a=True
#t = time.time()
while a:
    if os.path.isfile(fname):
        a = False
        os.remove(fname)
    time.sleep(0.0005)

exit(0)
