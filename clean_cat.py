#!/usr/bin/env python3

import glob
import time
import os
import gc
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import argparse
from astropy.table import Table
from numba import njit
import asdf

from fast_cksum.cksum_io import CksumWriter
#from tools.aid_asdf import save_asdf

# these are probably just for testing; should be removed for production
DEFAULTS = {}
#DEFAULTS['sim_name'] = "AbacusSummit_highbase_c021_ph000"
#DEFAULTS['sim_name'] = "AbacusSummit_highbase_c000_ph100"
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
#DEFAULTS['sim_name'] = "AbacusSummit_huge_c000_ph201"
#DEFAULTS['catalog_parent'] = "/mnt/gosling1/boryanah/light_cone_catalog/"
DEFAULTS['catalog_parent'] = "/global/cscratch1/sd/boryanah/light_cone_catalog/"
DEFAULTS['z_start'] = 0.8#0.350
DEFAULTS['z_stop'] = 0.8#1.625

# save light cone catalog
def save_asdf(table, filename, header, cat_lc_dir):
    # cram into a dictionary
    data_dict = {}
    for field in table.keys():
        data_dict[field] = table[field]
        
    # create data tree structure
    data_tree = {
        "data": data_dict,
        "header": header,
    }
    
    # save the data and close file
    output_file = asdf.AsdfFile(data_tree)
    output_file.write_to(os.path.join(cat_lc_dir, filename+".asdf"))
    output_file.close()

@njit
def fast_avg(vel, npout):
    nstart = 0
    v_int = np.zeros((len(npout), 3), dtype=np.float32)
    for i in range(len(npout)):
        if npout[i] == 0: continue
        v = vel[nstart:nstart+npout[i]]

        s = np.array([0, 0, 0])
        for k in range(npout[i]):
            for j in range(3):
                s[j] += v[k][j]
        for j in range(3):
            s[j] /= (npout[i])

        v_int[i] = s
        nstart += npout[i]

    return v_int

def vrange(starts, stops):
    """Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:

        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    stops = np.asarray(stops)
    l = stops - starts # Lengths of each range.
    return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    
def compress_asdf(asdf_fn, table, header):
    # cram into a dictionary
    data_dict = {}
    for field in table.keys():
        data_dict[field] = table[field]

    # create data tree structure
    data_tree = {
        "data": data_dict,
        "header": header,
    }
    
    # set compression options here
    asdf.compression.set_compression_options(typesize="auto", shuffle="shuffle", asdf_block_size=12*1024**2, blocksize=3*1024**2, nthreads=4)
    with asdf.AsdfFile(data_tree) as af, CksumWriter(str(asdf_fn)) as fp: # where data_tree is the ASDF dict tree structure
        af.write_to(fp, all_array_compression="blsc")

def extract_redshift(fn):
    red = float(str(fn).split('z')[-1][:5])
    return red

def float_trunc(a, zerobits):
    """Set the least significant <zerobits> bits to zero in a numpy float32 or float64 array.
    Do this in-place. Also return the updated array.
    Maximum values of 'nzero': 51 for float64; 22 for float32.
    """
    at = a.dtype
    assert at == np.float64 or at == np.float32 or at == np.complex128 or at == np.complex64
    if at == np.float64 or at == np.complex128:
        assert zerobits <= 51
        mask = 0xffffffffffffffff - (1 << zerobits) + 1
        bits = a.view(np.uint64)
        bits &= mask
    elif at == np.float32 or at == np.complex64:
        assert zerobits <= 22
        mask = 0xffffffff - (1 << zerobits) + 1
        bits = a.view(np.uint32)
        bits &= mask
    return a



def main(sim_name, z_start, z_stop, catalog_parent, want_subsample_B=True):
    # location of the light cone catalogs
    catalog_parent = Path(catalog_parent)
    
    # directory where we have saved the final outputs from merger trees and halo catalogs
    cat_lc_dir = catalog_parent / sim_name / "halos_light_cones"

    # list all available redshifts
    sim_slices = sorted(cat_lc_dir.glob('z*'))
    redshifts = [extract_redshift(sim_slices[i]) for i in range(len(sim_slices))]
    print("redshifts = ",redshifts)

    # loop through all available redshifts
    for z_current in redshifts:
        print("current redshift = ", z_current)

        if (z_current < z_start) or (z_current > z_stop): continue
        
        # load the halo light cone catalog
        halo_fn = cat_lc_dir / ("z%4.3f"%z_current) / "halo_info_lc.asdf"
        with asdf.open(halo_fn, lazy_load=True, copy_arrays=True) as f:
            halo_header = f['header']
            table_halo = f['data']

        # simulation parameters
        Lbox = halo_header['BoxSize']
        print("Lbox = ", Lbox)

        # load the particles light cone catalog
        parts_fn = cat_lc_dir / ("z%4.3f"%z_current) / "pid_rv_lc.asdf"
        with asdf.open(parts_fn, lazy_load=True, copy_arrays=True) as f:
            parts_header = f['header']
            table_parts = f['data']

        # parse the halo positions, npstart, npoutA and halo ids
        halo_pos = table_halo['pos_interp']
        halo_x = halo_pos[:, 0]
        halo_y = halo_pos[:, 1]
        halo_z = halo_pos[:, 2]
        halo_index = table_halo['index_halo']
        halo_npstart = table_halo['npstartA']
        halo_npout = table_halo['npoutA']
        halo_npoutA = halo_npout.copy()
        if want_subsample_B:
            halo_npout += table_halo['npoutB']
        halo_origin = table_halo['origin']

        # pars the particle id's
        parts_pid = table_parts['pid']

        remove_edges = True
        if remove_edges:
            str_edges = ""
            # find halos that are near the edges
            offset = 10.
            x_min = -Lbox/2.+offset
            x_max = Lbox/2.-offset
            y_min = -Lbox/2.+offset
            y_max = 3./2*Lbox # what about when we cross the 1000. boundary
            z_min = -Lbox/2.+offset
            z_max = 3./2*Lbox

            # define mask that picks away from the edges
            halo_mask = (halo_x >= x_min) & (halo_x < x_max)
            halo_mask &= (halo_y >= y_min) & (halo_y < y_max)
            halo_mask &= (halo_z >= z_min) & (halo_z < z_max)

            print("spatial masking = ", np.sum(halo_mask)*100./len(halo_mask))
        else:
            str_edges = "_all"
            halo_mask = np.ones(len(halo_x), dtype=bool)
        

        # figure out how many origins for the given redshifts
        unique_origins = np.unique(halo_origin)
        print("unique origins = ", unique_origins)

        # start an empty boolean array which will have "True" for only unique halos
        halo_mask_extra = np.zeros(len(halo_x), dtype=bool)

        # add to the halo mask requirement that halos be unique (for a given origin)
        for origin in unique_origins:

            # boolean array making halos at this origin
            mask_origin = halo_origin == origin

            # halo indices for this origin
            halo_inds = np.arange(len(halo_mask), dtype=int)[mask_origin]

            # find unique halo indices (already for specific origins)
            _, inds = np.unique(halo_index[mask_origin], return_index=True)
            halo_mask_extra[halo_inds[inds]] = True

            # how many halos were left
            print("non-unique masking %d = "%origin, len(inds)*100./np.sum(mask_origin))

        

        # additionally remove halos that are repeated on the borders (0 & 1 and 0 & 2)
        origin_xyz_dic = {1: 2, 2: 1}

        for key in origin_xyz_dic.keys():
            # select halos in the original box and the copies as long as they on the boundary
            mask_origin = ((halo_origin == 0) | ((halo_origin == key) & (halo_pos[:, origin_xyz_dic[key]] < Lbox/2.+offset)))

            # halo indices for this origin
            halo_inds = np.arange(len(halo_mask), dtype=int)[mask_origin]

            # find unique halo indices (already for specific origins)
            _, inds = np.unique(halo_index[mask_origin], return_index=True)
            halo_mask_extra[halo_inds[inds]] = True

            # how many halos were left
            print("non-unique masking extra %d = "%key, len(inds)*100./np.sum(mask_origin))
            
        # add the extra mask coming from the uniqueness requirement
        halo_mask &= halo_mask_extra

        # repeat halo mask npout times to get a mask for the particles
        parts_mask = np.repeat(halo_mask, halo_npout)
        print("particle masking from halos = ", np.sum(parts_mask)*100./len(parts_mask))

        # halo indices of the particles
        halo_inds = np.arange(len(halo_mask), dtype=int)
        parts_halo_inds = np.repeat(halo_inds, halo_npout)

        # number of unique hosts of particles belonging to halos near edges or repeated
        num_uni_hosts = len(np.unique(parts_halo_inds[parts_mask]))
        print("unique parts hosts, filtered halos = ", num_uni_hosts, np.sum(halo_mask))
        assert num_uni_hosts <= np.sum(halo_mask), "number of unique particle hosts must be less than or equal to number of halos in the mask"

        # add to the particle mask, particles whose pid equals 0
        parts_mask_extra = parts_pid != 0
        parts_mask &= parts_mask_extra

        print("pid == 0 masking = ", np.sum(parts_mask_extra)*100./len(parts_mask))

        # filter out the host halo indices of the particles left after removing halos near edges, non-unique halos and particles that were not matched
        parts_halo_inds = parts_halo_inds[parts_mask]

        # requires more thought cause it changes the npouts
        uni_halo_inds, inds, counts = np.unique(parts_halo_inds, return_index=True, return_counts=True)
        print("how many halos' lives did you ruin? = ", num_uni_hosts - len(inds))
        table_halo['npstartA'][:] = -999
        table_halo['npoutA'][:] = 0# todo I think that this will have A and B # could perhaps use the old npstartA here for counts
        table_halo['npstartA'][uni_halo_inds] = inds
        table_halo['npoutA'][uni_halo_inds] = counts

        # apply the mask to the particles and to the halos
        for key in table_parts.keys():
            table_parts[key] = table_parts[key][parts_mask]
        for key in table_halo.keys():
            table_halo[key] = table_halo[key][halo_mask]

        # simple checks
        assert np.sum(table_halo['npoutA']) == len(table_parts['pid']), "different number of particles and npout expectation"
        assert np.sum(table_parts['pid'] == 0) == 0, "still some particles with pid == 0"
        unique_origins = np.unique(table_halo['origin'])
        print("unique origins = ", unique_origins)
        for origin in unique_origins:
            assert len(np.unique(table_halo['index_halo'][origin == table_halo['origin']])) == np.sum(origin == table_halo['origin']), "still some non-unique halos left"

        # complicated checks
        parts_pos = table_parts['pos']
        parts_vel = table_parts['vel']
        halo_pos = table_halo['pos_interp']
        parts_halo_pos = np.repeat(halo_pos, table_halo['npoutA'], axis=0)
        parts_dist = parts_halo_pos - parts_pos
        parts_dist = np.sqrt(np.sum(parts_dist**2, axis=1))
        print("min dist = ", np.min(parts_dist))
        print("max dist = ", np.max(parts_dist))
        print(np.array(parts_pos[np.max(parts_dist) == parts_dist]))
        print(np.array(parts_halo_pos[np.max(parts_dist) == parts_dist]))

        # adding average velocity and position from subsample A (and B)
        halo_pos_avg = fast_avg(parts_pos, table_halo['npoutA'])
        halo_vel_avg = fast_avg(parts_vel, table_halo['npoutA'])
        table_halo['pos_avg'] = halo_pos_avg
        table_halo['vel_avg'] = halo_vel_avg


        # scaling down to only use the A subsample
        halo_npoutA = halo_npoutA[halo_mask]
        mask_lost = halo_npoutA > table_halo['npoutA']
        print("halos that now have fewer particles left than the initial subsample A = ", np.sum(mask_lost))
        halo_npoutA[mask_lost] = table_halo['npoutA'][mask_lost]
        starts = table_halo['npstartA'].astype(int)
        stops = starts + halo_npoutA.astype(int)
        parts_inds = vrange(starts, stops)

        # record the particles and the halos
        table_halo['npoutA'] = halo_npoutA
        for key in table_parts.keys():
            table_parts[key] = table_parts[key][parts_inds]

        '''
        # save asdf without compression (perhaps name can be lc_halo_info.asdf and lc_pid_rc.asdf
        save_asdf(table_parts, "lc"+str_edges+"_pid_rv", parts_header, cat_lc_dir / ("z%4.3f"%z_current))
        save_asdf(table_halo, "lc"+str_edges+"_halo_info", halo_header, cat_lc_dir / ("z%4.3f"%z_current))
        '''

        # knock out last few digits: 4 bits of the pos, the lowest 12 bits of vel
        table_parts['pos'] = float_trunc(table_parts['pos'], 4)
        table_parts['vel'] = float_trunc(table_parts['vel'], 12)
        table_parts['redshift'] = float_trunc(table_parts['redshift'], 12)

        # condense the asdf file
        halo_fn_new = cat_lc_dir / ("z%4.3f"%z_current) / ("lc"+str_edges+"_halo_info.asdf")
        compress_asdf(str(halo_fn_new), table_halo, halo_header)
        parts_fn_new = cat_lc_dir / ("z%4.3f"%z_current) / ("lc"+str_edges+"_pid_rv.asdf")
        compress_asdf(str(parts_fn_new), table_parts, parts_header)


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_start', help='Initial redshift where we start building the trees', type=float, default=DEFAULTS['z_start'])
    parser.add_argument('--z_stop', help='Final redshift (inclusive)', type=float, default=DEFAULTS['z_stop'])
    parser.add_argument('--catalog_parent', help='Light cone catalog directory', default=(DEFAULTS['catalog_parent']))
    parser.add_argument('--want_subsample_B', help='If this option is called, will only work with subsample A and exclude B', action='store_false')
    
    args = vars(parser.parse_args())
    main(**args)
