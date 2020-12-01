from simple_class import read_file
import numpy as np
import re
import glob
import os

header_dir = '../headers/'

headers = sorted(glob.glob(os.path.join(header_dir,'header_*')))

print(len(headers))

def get_redshift(header):
    header_dict = read_file(header)
    step = np.int(header.split('Step')[1])
    redshift = header_dict['              Redshift']
    coord_dist = header_dict['CoordinateDistanceHMpc']
    
    redshift = np.float(redshift)
    
    return redshift, step, coord_dist

redshifts = np.zeros(len(headers))
steps = np.zeros(len(headers),dtype=int)
coord_dist = np.zeros(len(headers))

for i in range(len(headers)):
    redshifts[i], steps[i], coord_dist[i] = get_redshift(headers[i])
    
np.save("../data_headers/redshifts.npy",redshifts)
np.save("../data_headers/steps.npy",steps)
np.save("../data_headers/coord_dist.npy",coord_dist)
print(redshifts)
print(steps)
print(coord_dist)
