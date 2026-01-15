#!/usr/bin/env python

from sys import argv as sargv
from sys import exit 
#from os.path import dirname
#import subprocess as sp
import time
import os

#inputFilename = sargv[1]
#directory = dirname(inputFilename)
#cmdline = "cd " + directory + "; "
open('rdy_namd','w').close()
fname = 'rdy_mace'
a=True
#t = time.time()
#print(os.getcwd())

while a:
    if os.path.isfile(fname):
        a = False
        os.remove(fname)
    time.sleep(0.005)

#print('MACE Done!\n')
exit(0)
