#!/usr/bin/env python3
'''
This is the second script in the "lightcone halo" pipeline.  The goal of this script is to use the output
from build_mt.py (i.e. information about what halos intersect the lightcone and when) and save the relevant
information from the CompaSO halo info catalogs.

Usage
-----
$ ./save_cat.py --help
'''

import sys
import glob
from pathlib import Path
import time
import gc
import os

import asdf
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import argparse

from compaso_halo_catalog import CompaSOHaloCatalog
from tools.aid_asdf import save_asdf, reindex_particles
from tools.merger import simple_load, get_slab_halo, get_slab_origin_halo, extract_superslab, extract_superslab_minified

# TODO: copy all halo fields (just get rid of fields=fields_cat)

# these are probably just for testing; should be removed for production
DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_highbase_c000_ph100"  # AbacusSummit_base_c000_ph006
DEFAULTS['compaso_parent'] = Path("/mnt/gosling2/bigsims")
#DEFAULTS['compaso_parent'] = Path("/global/project/projectdirs/desi/cosmosim/Abacus")
DEFAULTS['catalog_parent'] = Path("/mnt/gosling1/boryanah/light_cone_catalog/")
#DEFAULTS['catalog_parent'] = Path("/global/cscratch1/sd/boryanah/light_cone_catalog/")
DEFAULTS['merger_parent'] = Path("/mnt/gosling2/bigsims/merger")
#DEFAULTS['merger_parent'] = Path("/global/project/projectdirs/desi/cosmosim/Abacus/merger")
DEFAULTS['z_start'] = 0.45  # 0.8 # 0.5
DEFAULTS['z_stop'] = 0.65  # 1.25 # 0.8 # 0.5
CONSTANTS = {'c': 299792.458}  # km/s, speed of light

def extract_redshift(fn):
    fn = str(fn)
    redshift = float(fn.split('z')[-1])
    return redshift
    
def main(sim_name, z_start, z_stop, compaso_parent, catalog_parent, merger_parent, resume=False, plot=False):

    # directory where the CompaSO halo catalogs are saved
    cat_dir = compaso_parent / sim_name / "halos"

    # fields to extract from the CompaSO catalogs
    fields_cat = ['id','npstartA','npoutA','N','x_L2com','v_L2com','sigmav3d_L2com']

    # obtain the redshifts of the CompaSO catalogs
    redshifts = glob.glob(os.path.join(cat_dir,"z*"))
    zs_cat = [extract_redshift(redshifts[i]) for i in range(len(redshifts))]
    print(zs_cat)
    
    # directory where we save the final outputs
    cat_lc_dir = catalog_parent / sim_name / "halos_light_cones"

    # directory where the merger tree files are kept
    merger_dir = merger_parent / sim_name
    
    # more accurate, slightly slower
    if not os.path.exists("data/zs_mt.npy"):
        # all merger tree snapshots and corresponding redshifts
        snaps_mt = sorted(merger_dir.glob("associations_z*.0.asdf"))
        zs_mt = get_zs_from_headers(snaps_mt)
        np.save("data/zs_mt.npy", zs_mt)
    zs_mt = np.load("data/zs_mt.npy")

    # names of the merger tree file for a given redshift
    merger_fns = list(merger_dir.glob("associations_z%4.3f.*.asdf"%zs_mt[0]))
    
    # number of chunks
    n_chunks = len(merger_fns)
    print("number of chunks = ",n_chunks)
    
    # all redshifts, steps and comoving distances of light cones files; high z to low z
    # remove presaving after testing done (or make sure presaved can be matched with simulation)
    if not os.path.exists("data_headers/coord_dist.npy") or not os.path.exists("data_headers/redshifts.npy"):
        zs_all, steps, chis_all = get_lc_info("all_headers")
        np.save("data_headers/redshifts.npy", zs_all)
        np.save("data_headers/coord_dist.npy", chis_all)
    zs_all = np.load("data_headers/redshifts.npy")
    chis_all = np.load("data_headers/coord_dist.npy")
    zs_all[-1] = float("%.1f" % zs_all[-1])  # LHG: I guess this is trying to match up to some filename or something?

    # get functions relating chi and z
    chi_of_z = interp1d(zs_all,chis_all)
    z_of_chi = interp1d(chis_all,zs_all)
    
    # initial redshift where we start building the trees
    ind_start = np.argmin(np.abs(zs_mt-z_start))
    ind_stop = np.argmin(np.abs(zs_mt-z_stop))

    # loop over each merger tree redshift
    for i in range(ind_start,ind_stop+1):
        
        # starting snapshot
        z_mt = zs_mt[i]
        z_cat = zs_cat[np.argmin(np.abs(z_mt-zs_cat))]
        print("Redshift = %.3f %.3f"%(z_mt,z_cat))

        # names of the merger tree file for this redshift
        merger_fns = list(merger_dir.glob("associations_z%4.3f.*.asdf"%z_mt))
        for counter in range(len(merger_fns)):
            merger_fns[counter] = str(merger_fns[counter])

        # slab indices and number of halos per slab
        slabs, N_halo_slabs = get_slab_halo(merger_fns, minified=False)
        print(slabs, N_halo_slabs)

        # names of the light cone merger tree file for this redshift
        merger_lc_fns = list((cat_lc_dir / ("z%.3f"%z_mt)).glob("Merger_lc*.asdf"))
        for counter in range(len(merger_lc_fns)):
            merger_lc_fns[counter] = str(merger_lc_fns[counter])

        # slab indices, origins and number of halos per slab
        slabs_lc, origins_lc, N_halo_slabs_lc = get_slab_origin_halo(merger_lc_fns, minified=False)
        print(slabs_lc, origins_lc, N_halo_slabs_lc)
            
        # loop over each chunk
        offset = 0
        for k in range(n_chunks):
            # loop over each observer origin
            for o in range(3):
        
                # load the light cone arrays
                with asdf.open(cat_lc_dir / ("z%.3f"%z_mt) / ("Merger_lc%d.%02d.asdf"%(o,k)), lazy_load=True, copy_arrays=True) as f:
                    merger_lc = f['data']

                # correct halo indices
                merger_lc['HaloIndex'][:] += offset

                # combine with information for other halos in same redshift catalog
                try:
                    Merger_lc = np.hstack((Merger_lc,merger_lc))
                except:
                    Merger_lc = merger_lc.copy()

                print("keys = ",Merger_lc.keys(),merger_lc.keys())
                    
                # print to check numbers
                N_halo_lc = len(merger_lc['HaloIndex'])
                N_halo_lc_total = len(Merger_lc['HaloIndex'][:])
                print(N_halo_lc, N_halo_lc_total)

            # offset all halos in given chunk
            offset += N_halo_slabs[k]
            
        # unpack the fields of the merger tree catalogs
        halo_ind_lc = Merger_lc['HaloIndex'][:]
        pos_interp_lc = Merger_lc['InterpolatedPosition'][:]
        vel_interp_lc = Merger_lc['InterpolatedVelocity'][:]
        chi_interp_lc = Merger_lc['InterpolatedComoving'][:]
        del Merger_lc
                
        # catalog directory 
        catdir = cat_dir / "z%.3f"%z_cat

        # load halo catalog, setting unpack to False for speed
        cat = CompaSOHaloCatalog(catdir, load_subsamples='A_halo_pid', fields=fields_cat, unpack_bits = False)
        
        # halo catalog
        halo_table = cat.halos[halo_ind_lc]
        header = cat.header
        N_halos = len(cat.halos)
        print("N_halos = ",N_halos)

        # load the pid, set unpack_bits to True if you want other information
        pid = cat.subsamples['pid']
        npstart = halo_table['npstartA']
        npout = halo_table['npoutA']
        pid_new, npstart_new, npout_new = reindex_particles(pid,npstart,npout)        
        pid_table = np.empty(len(pid_new),dtype=[('pid',pid_new.dtype)])
        pid_table['pid'] = pid_new
        halo_table['npstartA'] = npstart_new
        halo_table['npoutA'] = npout_new

        # isolate halos that did not have interpolation and get the velocity from the halo info files
        not_interp = (np.sum(np.abs(vel_interp_lc),axis=1) - 0.) < 1.e-6
        vel_interp_lc[not_interp] = halo_table['v_L2com'][not_interp]
        print("percentage not interpolated = ", 100.*np.sum(not_interp)/len(not_interp))

        # append new fields
        halo_table['index_halo'] = halo_ind_lc
        halo_table['pos_interp'] = pos_interp_lc
        halo_table['vel_interp'] = vel_interp_lc
        halo_table['redshift_interp'] = z_of_chi(chi_interp_lc)

        # save to files
        save_asdf(halo_table, "halo_info_lc", header, cat_lc_dir / "z%4.3f"%z_mt)
        save_asdf(pid_table, "pid_lc", header, cat_lc_dir / "z%4.3f"%z_mt)

        
        # delete things at the end
        del pid, pid_new, pid_table, npstart, npout, npstart_new, npout_new
        del halo_table
        del cat

        gc.collect()
            
#dict_keys(['HaloIndex', 'HaloMass', 'HaloVmax', 'IsAssociated', 'IsPotentialSplit', 'MainProgenitor', 'MainProgenitorFrac', 'MainProgenitorPrec', 'MainProgenitorPrecFrac', 'NumProgenitors', 'Position', 'Progenitors'])

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_start', help='Initial redshift where we start building the trees', default=DEFAULTS['z_start'])
    parser.add_argument('--z_stop', help='Final redshift (inclusive)', default=DEFAULTS['z_stop'])
    parser.add_argument('--compaso_parent', help='CompaSO directory', default=DEFAULTS['compaso_parent'])
    parser.add_argument('--catalog_parent', help='Light cone catalog directory', default=DEFAULTS['catalog_parent'])
    parser.add_argument('--merger_parent', help='Merger tree directory', default=DEFAULTS['merger_parent'])
    parser.add_argument('--resume', help='Resume the calculation from the checkpoint on disk', action='store_true')
    parser.add_argument('--plot', help='Want to show plots', action='store_true')
    
    args = vars(parser.parse_args())
    main(**args)
