#!/usr/bin/env python3
'''
This is a script for generating HOD mock catalogs.

Usage
-----
$ python ./run_hod.py --help
'''

import os
import glob
import time
import sys

import yaml
import numpy as np
import argparse

#from abacusnbody.hod.abacus_hod import AbacusHOD
sys.path.append('../..')
from abacus_hod.hod.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/buba.yaml'

def extract_redshift(fn):
    red = float(fn.split('z')[-1][:5])
    return red

def main(path2config):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    
    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'])
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']

    # for what redshifts are subsampled halos and particles available
    cat_lc_dir = os.path.join(sim_params['subsample_dir'], sim_params['sim_name'])
    sim_slices = sorted(glob.glob(os.path.join(cat_lc_dir,'z*')))
    redshifts = [extract_redshift(sim_slices[i]) for i in range(len(sim_slices))]
    print("redshifts = ",redshifts)
    
    # load parameters pertaining to the HOD of ELGs in TNG
    der_config = yaml.load(open('config/TNG_ELG_HOD.yaml'))
    der_dic = {}
    fid_dic = {}
    zs = np.zeros(3)
    pars = np.zeros(3)
    for key in der_config['z0']['ELG_params'].keys():
        for i in range(3):
            zs[i] = der_config[f'z{i}']['z']
            pars[i] = der_config[f'z{i}']['ELG_params'][key]
            if i == 0:
                fid_dic[key] = der_config[f'z{i}']['ELG_params'][key]
                z_fid = der_config[f'z{i}']['z']
        der_dic[key] = (pars[-1] - pars[0])/(zs[-1] - zs[0])

    # create dictionaries with all designs
    HOD_dicts = []
    for i in range(len(redshifts)):
        # this redshift
        redshift = redshifts[i]

        # use the same HOD model for lower redshifts
        if redshift <= 0.725:
            HOD_dicts.append(fid_dic)
            continue

        # record dictionary
        HOD_dict = {}
        for key in fid_dic.keys():
            HOD_dict[key] = fid_dic[key] + der_dic[key]*(redshift - z_fid)
        print("z = ", redshift)    
        #print("HOD_dict = ", HOD_dict.items())
        print("-----------------------------")
        HOD_dicts.append(HOD_dict)

    for i in range(len(redshifts)):
        # this redshift
        redshift = redshifts[i]

        #if redshift != 0.5: continue
        
        # modify redshift in sim_params tuks
        sim_params['z_mock'] = redshift
        
        # create a new abacushod object
        newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

        # take the HOD dictionary
        for key in HOD_dicts[i].keys():
            newBall.tracers['ELG'][key] = HOD_dicts[i][key]

        # run the HOD on the current redshift
        start = time.time()
        mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk)
        print("Done redshift ", redshift, "took time ", time.time() - start)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":


    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())
    main(**args)
