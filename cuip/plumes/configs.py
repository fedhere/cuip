import os
MAXPROCESSES = 5e6
OUTPUTDIR = "./outputs/"
LIM = -1 #25
WINDOW = 5
IMYSIZE = 2160
IMYSIZE = 4096

NPTS = 10

try:
    DST_WRITE = os.environ['DST_WRITE']
except KeyError:
    DST_WRITE = os.environ['HOME']
DT = 10 # time interval between images
