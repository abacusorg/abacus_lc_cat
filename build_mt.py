#!/usr/bin/env python3
'''
This is the first script in the "lightcone halo" pipeline.  The goal of this script is to use merger
tree information to flag halos that intersect the lightcone and make a unique determination of which
halo catalog epoch from which to draw the halo.

Usage
-----
$ ./build_mt.py --help
'''

import sys
import glob
import time
import gc
import os
from pathlib import Path

import asdf
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import argparse
import numba as nb
from astropy.table import Table
from astropy.io import ascii

from tools.InputFile import InputFile
from tools.merger import simple_load, get_halos_per_slab, extract_superslab, extract_superslab_minified
from tools.aid_asdf import save_asdf
from tools.read_headers import get_lc_info

# these are probably just for testing; should be removed for production
DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_highbase_c000_ph100"  # AbacusSummit_base_c000_ph006
DEFAULTS['merger_parent'] = Path("/mnt/gosling2/bigsims/merger")
#DEFAULTS['merger_parent'] = Path("/global/project/projectdirs/desi/cosmosim/Abacus/merger")
DEFAULTS['catalog_parent'] = Path("/mnt/gosling1/boryanah/light_cone_catalog/")
#DEFAULTS['catalog_parent'] = Path("/global/cscratch1/sd/boryanah/light_cone_catalog/")
DEFAULTS['z_start'] = 0.65  # 0.8 # 0.5
DEFAULTS['z_stop'] = 0.72  # 1.25 # 0.8 # 0.5
CONSTANTS = {'c': 299792.458}  # km/s, speed of light

def reorder_by_slab(fns,minified):
    '''
    Reorder filenames in terms of their slab number
    '''
    if minified:
        return sorted(fns, key=extract_superslab_minified)
    else:
        return sorted(fns, key=extract_superslab)

def get_one_header(merger_dir):
    '''
    Get an example header by looking at one association
    file in a merger directory
    '''

    # choose one of the merger tree files
    fn = list(merger_dir.glob('associations*.asdf'))[0]
    with asdf.open(fn) as af:
        header = af['header']
    return header
    
def get_zs_from_headers(snap_names):
    '''
    Read redshifts from merger tree files
    '''
    
    zs = np.zeros(len(snap_names))
    for i in range(len(snap_names)):
        snap_name = snap_names[i]
        with asdf.open(snap_name) as f:
            zs[i] = np.float(f["header"]["Redshift"])
    return zs

@nb.njit
def dist(pos1, pos2, L=None):
    '''
    Calculate L2 norm distance between a set of points
    and either a reference point or another set of points.
    Optionally includes periodicity.
    
    Parameters
    ----------
    pos1: ndarray of shape (N,m)
        A set of points
    pos2: ndarray of shape (N,m) or (m,) or (1,m)
        A single point or set of points
    L: float, optional
        The box size. Will do a periodic wrap if given.
    
    Returns
    -------
    dist: ndarray of shape (N,)
        The distances between pos1 and pos2
    '''
    
    # read dimension of data
    N, nd = pos1.shape
    
    # allow pos2 to be a single point
    pos2 = np.atleast_2d(pos2)
    assert pos2.shape[-1] == nd
    broadcast = len(pos2) == 1
        
    dist = np.empty(N, dtype=pos1.dtype)
    
    i2 = 0
    for i in range(N):
        delta = 0.
        for j in range(nd):
            dx = pos1[i][j] - pos2[i2][j]
            if L is not None:
                if dx >= L/2:
                    dx -= L
                elif dx < -L/2:
                    dx += L
            delta += dx*dx
        dist[i] = np.sqrt(delta)
        if not broadcast:
            i2 += 1
    return dist

def unpack_inds(halo_ids):
    '''
    Unpack indices in Sownak's format of Nslice*1e12 
    + superSlabNum*1e9 + halo_position_superSlab
    '''
    
    # obtain slab number and index within slab
    id_factor = int(1e12)
    slab_factor = int(1e9)
    index = (halo_ids % slab_factor).astype(int)
    slab_number = ((halo_ids % id_factor - index) // slab_factor).astype(int)
    return slab_number, index

def correct_inds(halo_ids, N_halos_slabs, slabs, inds_fn):
    '''
    Reorder indices for given halo index array with 
    corresponding n halos and slabs for its time epoch
    '''

    # number of halos in the loaded chunks
    N_halos_load = np.array([N_halos_slabs[i] for i in inds_fn])
    
    # unpack slab and index for each halo
    slab_ids, ids = unpack_inds(halo_ids)

    # total number of halos in the slabs that we have loaded
    N_halos = np.sum(N_halos_load)
    offsets = np.zeros(len(inds_fn), dtype=int)
    offsets[1:] = np.cumsum(N_halos_load)[:-1]
    
    # determine if unpacking halos for only one file (Merger_this['HaloIndex']) -- no need to offset 
    if len(inds_fn) == 1: return ids

    '''
    # an attempt to speed up code but might be slower than currently done
    # TODO: there's a bug, ask Lehman (not necessary to fix)
    # determine if indices are contiguous in terms of their chunk number (np.unique will return slab_unique sorted)
    slab_unique, slab_first_ids = np.unique(slab_ids, return_index=True)
    contiguous = True
    for i in range(len(slab_first_ids)):
        if slab_first_ids[i] != offsets[np.argsort(inds_fn)][i] or slab_unique[i] != np.sort(inds_fn)[i]:
            contiguous = False
    if contiguous:
        for i in range(len(slab_first_ids)):
            ids[N_halos_load[i]*i:N_halos_load[i]*(i+1)] += offsets[i]
        return ids
    ''' 
        
    # select the halos belonging to given slab
    for i, ind_fn in enumerate(inds_fn):
        select = np.where(slab_ids == slabs[ind_fn])[0]
        ids[select] += offsets[i]

    return ids

def get_mt_info(fns_load, fields, minified):
    '''
    Load merger tree and progenitors information
    '''
    
    data = simple_load(fns_load, fields=fields)
    merger = data['merger']
    
    # get number of halos in each slab and number of slabs
    # TODO: simple_load() could return this, but there's a use for this function standalone elsewhere
    # TODO: LHG is unsure if the order returned by get_halos_per_slab() matches that returned by simple_load(), and if that matters
    halos_per_slab = get_halos_per_slab(fns, minified)

    # if loading all progenitors
    if "Progenitors" in fields:
        num_progs = merger["NumProgenitors"]
        # get an array with the starting indices of the progenitors array
        start_progs = np.empty(len(merger), dtype=int)
        start_progs[0] = 0
        start_progs[1:] = num_progs.cumsum()[:-1]
        merger.add_column(start_progs, name='StartProgenitors', copy=False)

    return data, halos_per_slab

def solve_crossing(r1, r2, pos1, pos2, chi1, chi2, Lbox, origin):
    '''
    Solve when the crossing of the light cones occurs and the
    interpolated position and velocity
    '''
    
    # identify where the distance between this object and its main progenitor is larger than half the boxsize (or really even 4 Mpc/h since that is Sownak's boundary)
    delta_pos = np.abs(pos2 - pos1)
    delta_pos = np.where(delta_pos > 0.5 * Lbox, (delta_pos - Lbox), delta_pos)
    delta_sign = np.sign(pos1 - pos2)

    # move the halos so that you keep things continuous
    pos1 = pos2 + delta_sign * delta_pos
    r1 = dist(pos1, origin)
    r2 = dist(pos2, origin)

    # solve for chi_star, where chi = eta_0-eta
    # equation is r1+(chi1-chi)/(chi1-chi2)*(r2-r1) = chi, with solution:
    chi_star = (r1 * (chi1 - chi2) + chi1 * (r2 - r1)) / ((chi1 - chi2) + (r2 - r1))

    # get interpolated positions of the halos
    v_avg = (pos2 - pos1) / (chi1 - chi2)  # og
    pos_star = pos1 + v_avg * (chi1 - chi_star[:, None])


    # enforce boundary conditions by periodic wrapping
    pos_star[pos_star >= Lbox/2.] = pos_star[pos_star >= Lbox/2.] - Lbox
    pos_star[pos_star < -Lbox/2.] = pos_star[pos_star < -Lbox/2.] + Lbox
    
    # interpolated velocity [km/s]
    vel_star = v_avg * CONSTANTS['c']  # vel1+a_avg*(chi1-chi_star)

    # mark True if closer to chi2 (this snapshot)
    bool_star = np.abs(chi1 - chi_star) > np.abs(chi2 - chi_star)

    # condition to check whether halo in this light cone band
    # assert np.sum((chi_star > chi1) | (chi_star < chi2)) == 0, "Solution is out of bounds"
    
    return chi_star, pos_star, vel_star, bool_star

def offset_pos(pos,ind_origin,all_origins):
    '''
    Offset the interpolated positions to create continuous light cones
    '''

    # location of initial observer
    first_observer = all_origins[0]
    current_observer = all_origins[ind_origin]
    offset = (first_observer-current_observer)
    pos += offset
    return pos

def main(sim_name, z_start, z_stop, merger_parent, catalog_parent, resume=False, plot=False):
    '''
    Main function.
    The algorithm: for each merger tree epoch, for 
    each superslab, for each light cone origin,
    compute the intersection of the light cone with
    each halo, using the interpolated position
    to the previous merger epoch (and possibly a 
    velocity correction).  If the intersection is
    between the current and previous merger epochs, 
    then record the closer one as that halo's
    epoch and mark its progenitors as ineligible.
    Will need one padding superslab in the previous
    merger epoch.  Can process in a rolling fashion.
    '''
    
    merger_dir = merger_parent / sim_name
    header = get_one_header(merger_dir)
    
    # simulation parameters
    Lbox = header['BoxSize']
    # location of the LC origins in Mpc/h
    origins = np.array(header['LightConeOrigins']).reshape(-1,3)

    # just for testing with highbase. remove!
    origins /= 2.
    
    # directory where we save the final outputs
    cat_lc_dir = catalog_parent / sim_name / "halos_light_cones/"
    os.makedirs(cat_lc_dir, exist_ok=True)

    # directory where we save the current state if we want to resume
    os.makedirs(cat_lc_dir / "tmp", exist_ok=True)
    with open(cat_lc_dir / "tmp" / "tmp.log", "a") as f:
        f.writelines(["# Starting light cone catalog construction in simulation %s \n"%sim_name])
    
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
    chi_of_z = interp1d(zs_all, chis_all)
    z_of_chi = interp1d(chis_all, zs_all)

    # more accurate, slightly slower
    if not os.path.exists("data/zs_mt.npy"):
        # all merger tree snapshots and corresponding redshifts
        snaps_mt = sorted(merger_dir.glob("associations_z*.0.asdf"))
        zs_mt = get_zs_from_headers(snaps_mt)
        np.save("data/zs_mt.npy", zs_mt)
    zs_mt = np.load("data/zs_mt.npy")

    # number of chunks
    n_chunks = len(list(merger_dir.glob("associations_z%4.3f.*.asdf"%zs_mt[0])))
    print("number of chunks = ",n_chunks)

    # starting and finishing redshift indices indices
    ind_start = np.argmin(np.abs(zs_mt - z_start))
    ind_stop = np.argmin(np.abs(zs_mt - z_stop))

    if resume:
        # if user wants to resume from previous state, create padded array for marking whether chunk has been loaded
        resume_flags = np.ones((n_chunks, origins.shape[1]), dtype=bool)
        
        # previous redshift, distance between shells
        infile = InputFile(cat_lc_dir / "tmp" / "tmp.log")
        z_this_tmp = infile.z_prev
        delta_chi_old = infile.delta_chi
        chunk = infile.super_slab
        assert (np.abs(n_chunks-1 - chunk) < 1.0e-6), "Your recorded state did not complete all chunks, can't resume from old"
        assert (np.abs(zs_mt[ind_start] - z_this_tmp) < 1.0e-6), "Your recorded state is not for the correct redshift, can't resume from old"
        with open(cat_lc_dir / "tmp" / "tmp.log", "a") as f:
            f.writelines(["# Resuming from redshift z = %4.3f \n"%z_this_tmp])
    else:
        # delete the exisiting temporary files
        tmp_files = list((cat_lc_dir / "tmp").glob("*"))
        for i in range(len(tmp_files)):
            os.unlink(str(tmp_files[i]))
        resume_flags = np.zeros((n_chunks, origins.shape[0]), dtype=bool)

    # fields to extract from the merger trees
    # fields_mt = ['HaloIndex','HaloMass','Position','MainProgenitor','Progenitors','NumProgenitors']
    # lighter version
    fields_mt = ['HaloIndex', 'Position', 'MainProgenitor']

    # redshift of closest point on wall between original and copied box
    z1 = z_of_chi(0.5 * Lbox - origins[0][0])
    # redshift of closest point where all three boxes touch
    # z2 = z_of_chi((0.5*Lbox-origin[0])*np.sqrt(2))
    # furthest point where all three boxes touch;
    z3 = z_of_chi((0.5 * Lbox - origins[0][0]) * np.sqrt(3))

    # initialize difference between the conformal time of last two shells
    delta_chi_old = 0.0
    
    for i in range(ind_start, ind_stop + 1):

        # this snapshot redshift and the previous
        z_this = zs_mt[i]
        z_prev = zs_mt[i + 1]
        print("redshift of this and the previous snapshot = ", z_this, z_prev)

        # coordinate distance of the light cone at this redshift and the previous
        assert z_this >= np.min(zs_all), "You need to set starting redshift to the smallest value of the merger tree"
        chi_this = chi_of_z(z_this)
        chi_prev = chi_of_z(z_prev)
        delta_chi = chi_prev - chi_this
        print("comoving distance between this and previous snapshot = ", delta_chi)

        # read merger trees file names at this and previous snapshot from minified version 
        # LHG: do we need to support both minified and non-minified separately? I thought all the data was in the minifted format now.
        fns_this = list(merger_dir.glob(f'associations_z{z_this:4.3f}.*.asdf.minified'))
        fns_prev = list(merger_dir.glob(f'associations_z{z_prev:4.3f}.*.asdf.minified'))
        minified = True

        # if minified files not available, load the regular files
        if len(list(fns_this)) == 0 or len(list(fns_prev)) == 0:
            fns_this = list(merger_dir.glob(f'associations_z{z_this:4.3f}.*.asdf'))
            fns_prev = list(merger_dir.glob(f'associations_z{z_prev:4.3f}.*.asdf'))
            minified = False

        # turn file names into strings
        fns_this = [str(f) for f in fns_this]
        fns_prev = [str(f) for f in fns_prev]

        # number of merger tree files
        print("number of files = ", len(fns_this), len(fns_prev))
        assert n_chunks == len(fns_this) and n_chunks == len(fns_prev), "Incomplete merger tree files"

        # reorder file names by super slab number
        fns_this = reorder_by_slab(fns_this,minified)
        fns_prev = reorder_by_slab(fns_prev,minified)
        
        # maybe we want to support resuming from arbitrary superslab
        first_ss = 0
        
        # We're going to be loading slabs in a rolling fashion:
        # reading the "high" slab at the leading edge, discarding the trailing "low" slab
        # and moving the mid to low. But first we need to read all three to prime the queue
        mt_prev = {}  # indexed by slab num
        mt_prev[(first_ss-1)%n_chunks] = get_mt_info(fns_prev[(first_ss-1)%n_chunks], fields=fields_mt, minified=minified)
        mt_prev[first_ss] = get_mt_info(fns_prev[first_ss], fields=fields_mt, minified=minified)

        # for each chunk
        for k in range(first_ss,n_chunks):
            # starting and finishing superslab chunks
            klow = (k-1)%n_chunks
            khigh = (k+1)%n_chunks
            
            # Slide down by one
            if (klow-1)%n_chunks in mt_prev:
                del mt_prev[(klow-1)%n_chunks]
            mt_prev[khigh] = get_mt_info(fns_prev[khigh], fields=fields_mt, minified=minified)

            print(f"Loaded chunk {k} in this redshift, and {tuple(mt_prev)} in previous")
            # get merger tree data for this snapshot and for the previous one
            mt_data_this, halos_per_slab_this = get_mt_info(fns_this[k], fields=fields_mt, minified=minified)
            
            # ======== LHG: haven't edited below here

            # number of halos in this step and previous step; this depends on the number of files requested
            N_halos_this = np.sum(N_halos_slabs_this[inds_fn_this])
            N_halos_prev = np.sum(N_halos_slabs_prev[inds_fn_prev])
            print("N_halos_this = ", N_halos_this)
            print("N_halos_prev = ", N_halos_prev)
            
            # mask where no merger tree info is available (because we don'to need to solve for eta star for those)
            noinfo_this = Merger_this['MainProgenitor'] <= 0
            info_this = Merger_this['MainProgenitor'] > 0
            
            # print percentage where no information is available or halo not eligible
            print("percentage no info = ", np.sum(noinfo_this) / len(noinfo_this) * 100.0)

            # no info is denoted by 0 or -999 (or regular if ineligible), but -999 messes with unpacking, so we set it to 0
            Merger_this['MainProgenitor'][noinfo_this] = 0

            # rework the main progenitor and halo indices to return in proper order
            Merger_this['HaloIndex'] = correct_inds(
                Merger_this['HaloIndex'],
                N_halos_slabs_this,
                slabs_this,
                inds_fn_this,
            )
            Merger_this['MainProgenitor'] = correct_inds(
                Merger_this['MainProgenitor'],
                N_halos_slabs_prev,
                slabs_prev,
                inds_fn_prev,
            )
            Merger_prev['HaloIndex'] = correct_inds(
                Merger_prev['HaloIndex'],
                N_halos_slabs_prev,
                slabs_prev,
                inds_fn_prev,
            )
            
            # loop over all origins
            for o in range(len(origins)):

                # location of the observer
                origin = origins[o]
                
                # comoving distance to observer
                Merger_this['ComovingDistance'] = dist(Merger_this['Position'], origin)
                Merger_prev['ComovingDistance'] = dist(Merger_prev['Position'], origin)
                
                # merger tree data of main progenitor halos corresponding to the halos in current snapshot
                Merger_prev_main_this = Merger_prev[Merger_this['MainProgenitor']].copy()
                
                # if eligible, can be selected for light cone redshift catalog;
                if i != ind_start or resume_flags[k, o]:
                    # dealing with the fact that these files may not exist for all origins and all chunks
                    if os.path.exists(cat_lc_dir / "tmp" / ("eligibility_prev_z%4.3f_lc%d.%02d.npy"%(z_this, o, k))):
                        eligibility_this = np.load(cat_lc_dir / "tmp" / ("eligibility_prev_z%4.3f_lc%d.%02d.npy"%(z_this, o, k)))
                    else:
                        eligibility_this = np.ones(N_halos_this, dtype=bool)
                else:
                    eligibility_this = np.ones(N_halos_this, dtype=bool)
                
                # for a newly opened redshift, everyone is eligible to be part of the light cone catalog
                eligibility_prev = np.ones(N_halos_prev, dtype=bool)

                # mask for eligible halos for light cone origin with and without information
                mask_noinfo_this = noinfo_this & eligibility_this
                mask_info_this = info_this & eligibility_this

                # halos that have merger tree information
                Merger_this_info = Merger_this[mask_info_this].copy()
                Merger_prev_main_this_info = Merger_prev_main_this[mask_info_this]
                
                # halos that don't have merger tree information
                Merger_this_noinfo = Merger_this[mask_noinfo_this].copy()
                
                # select objects that are crossing the light cones
                # TODO: revise conservative choice if stranded between two ( & \) less conservative ( | \ )
                mask_lc_this_info = (
                    ((Merger_this_info['ComovingDistance'] > chi_this) & (Merger_this_info['ComovingDistance'] <= chi_prev))
                )
                #| ((Merger_prev_main_this_info['ComovingDistance'] > chi_this) & (Merger_prev_main_this_info['ComovingDistance'] <= chi_prev))

                mask_lc_this_noinfo = (
                    (Merger_this_noinfo['ComovingDistance'] > chi_this - delta_chi_old / 2.0)
                    & (Merger_this_noinfo['ComovingDistance'] <= chi_this + delta_chi / 2.0)
                )

                # spare the computer the effort and avert empty array errors
                # TODO: perhaps revise, as sometimes we might have no halos in
                # noinfo but some in info and vice versa
                if np.sum(mask_lc_this_info) == 0 or np.sum(mask_lc_this_noinfo) == 0: continue

                # percentage of objects that are part of this or previous snapshot
                print(
                    "percentage of halos in light cone %d with and without progenitor info = "%o,
                    np.sum(mask_lc_this_info) / len(mask_lc_this_info) * 100.0,
                    np.sum(mask_lc_this_noinfo) / len(mask_lc_this_noinfo) * 100.0,
                )

                # select halos with mt info that have had a light cone crossing
                Merger_this_info_lc = Merger_this_info[mask_lc_this_info]
                Merger_prev_main_this_info_lc = Merger_prev_main_this_info[mask_lc_this_info]

                if plot:
                    x_min = -Lbox/2.+k*(Lbox/n_chunks)
                    x_max = x_min+(Lbox/n_chunks)

                    x = Merger_this_info_lc['Position'][:,0]
                    choice = (x > x_min) & (x < x_max)
                    
                    y = Merger_this_info_lc['Position'][choice,1]
                    z = Merger_this_info_lc['Position'][choice,2]
                    
                    plt.figure(1)
                    plt.scatter(y, z, color='dodgerblue', s=0.1, label='current objects')

                    plt.legend()
                    plt.axis('equal')

                    x = Merger_prev_main_this_info_lc['Position'][:,0]
                    
                    choice = (x > x_min) & (x < x_max)

                    y = Merger_prev_main_this_info_lc['Position'][choice,1]
                    z = Merger_prev_main_this_info_lc['Position'][choice,2]
                    
                    plt.figure(2)
                    plt.scatter(y, z, color='orangered', s=0.1, label='main progenitor')

                    plt.legend()
                    plt.axis('equal')
                    plt.show()

                # select halos without mt info that have had a light cone crossing
                Merger_this_noinfo_lc = Merger_this_noinfo[mask_lc_this_noinfo]

                # add columns for new interpolated position, velocity and comoving distance
                Merger_this_info_lc.add_column('InterpolatedPosition',copy=False)
                Merger_this_info_lc.add_column('InterpolatedVelocity',copy=False)
                Merger_this_info_lc.add_column('InterpolatedComoving',copy=False)

                # get chi star where lc crosses halo trajectory; bool is False where closer to previous
                (
                    Merger_this_info_lc['InterpolatedComoving'],
                    Merger_this_info_lc['InterpolatedPosition'],
                    Merger_this_info_lc['InterpolatedVelocity'],
                    bool_star_this_info_lc,
                ) = solve_crossing(
                    Merger_prev_main_this_info_lc['ComovingDistance'],
                    Merger_this_info_lc['ComovingDistance'],
                    Merger_prev_main_this_info_lc['Position'],
                    Merger_this_info_lc['Position'],
                    chi_prev,
                    chi_this,
                    Lbox,
                    origin
                )

                # number of objects in this light cone
                N_this_star_lc = np.sum(bool_star_this_info_lc)
                N_this_noinfo_lc = np.sum(mask_lc_this_noinfo)

                if i != ind_start or resume_flags[k, o]:
                    # cheap way to deal with the fact that sometimes we won't have information about all light cone origins for certain chunks and epochs
                    if os.path.exists(cat_lc_dir / "tmp" / ("Merger_next_z%4.3f_lc%d.%02d.asdf"%(z_this,o,k))):
                        # load leftover halos from previously loaded redshift
                        with asdf.open(cat_lc_dir / "tmp" / ("Merger_next_z%4.3f_lc%d.%02d.asdf"%(z_this,o,k))) as f:
                            Merger_next = f['data']

                        # adding contributions from the previously loaded redshift
                        N_next_lc = len(Merger_next['HaloIndex'])
                    else:
                        N_next_lc = 0
                else:
                    N_next_lc = 0

                # total number of halos belonging to this light cone superslab and origin
                N_lc = N_this_star_lc + N_this_noinfo_lc + N_next_lc
                
                print("in this snapshot: interpolated, no info, next, total = ", N_this_star_lc * 100.0 / N_lc, N_this_noinfo_lc * 100.0 / N_lc, N_next_lc * 100.0 / N_lc, N_lc)
                
                # save those arrays
                Merger_lc = Table(
                    {'HaloIndex':np.zeros(N_lc, dtype=Merger_this_info_lc['HaloIndex'].dtype),
                     'InterpolatedVelocity': np.zeros(N_lc, dtype=(np.float32,3)),
                     'InterpolatedPosition': np.zeros(N_lc, dtype=(np.float32,3)),
                     'InterpolatedComoving': np.zeros(N_lc, dtype=np.float32)
                    }
                )

                # record interpolated position and velocity for those with info belonging to current redshift
                Merger_lc['InterpolatedPosition'][:N_this_star_lc] = Merger_this_info_lc['InterpolatedPosition'][bool_star_this_info_lc]
                Merger_lc['InterpolatedVelocity'][:N_this_star_lc] = Merger_this_info_lc['InterpolatedVelocity'][bool_star_this_info_lc]
                Merger_lc['InterpolatedComoving'][:N_this_star_lc] = Merger_this_info_lc['InterpolatedComoving'][bool_star_this_info_lc]
                Merger_lc['HaloIndex'][:N_this_star_lc] = Merger_this_info_lc['HaloIndex'][bool_star_this_info_lc]

                # record interpolated position and velocity of the halos in the light cone without progenitor information
                Merger_lc['InterpolatedPosition'][N_this_star_lc:N_this_star_lc+N_this_noinfo_lc] = Merger_this_noinfo_lc['Position']
                Merger_lc['InterpolatedVelocity'][N_this_star_lc:N_this_star_lc+N_this_noinfo_lc] = np.zeros_like(Merger_this_noinfo_lc['Position'])
                Merger_lc['InterpolatedComoving'][N_this_star_lc:N_this_star_lc+N_this_noinfo_lc] = np.ones(Merger_this_noinfo_lc['Position'].shape[0])*chi_this
                Merger_lc['HaloIndex'][N_this_star_lc:N_this_star_lc+N_this_noinfo_lc] = Merger_this_noinfo_lc['HaloIndex']
                del Merger_this_noinfo_lc

                # record information from previously loaded redshift that was postponed
                if i != ind_start or resume_flags[k, o]:
                    if N_next_lc != 0:
                        Merger_lc['InterpolatedPosition'][-N_next_lc:] = Merger_next['InterpolatedPosition']['data'][:]
                        Merger_lc['InterpolatedVelocity'][-N_next_lc:] = Merger_next['InterpolatedVelocity']['data'][:]
                        Merger_lc['InterpolatedComoving'][-N_next_lc:] = Merger_next['InterpolatedComoving']['data'][:]
                        Merger_lc['HaloIndex'][-N_next_lc:] = Merger_next['HaloIndex']['data'][:]
                        del Merger_next
                    resume_flags[k, o] = False

                
                # offset position to make light cone continuous
                Merger_lc['InterpolatedPosition'] = offset_pos(Merger_lc['InterpolatedPosition'],ind_origin=o,all_origins=origins)
                
                # create directory for this redshift
                os.makedirs(cat_lc_dir / ("z%.3f"%z_this), exist_ok=True)

                # write table with interpolated information
                save_asdf(Merger_lc, ("Merger_lc%d.%02d"%(o,k)), header, cat_lc_dir / ("z%.3f"%z_this))

                # TODO: Need to make sure no bugs with eligibility, ask Lehman
                # version 1: only the main progenitor is marked ineligible
                # if halo belongs to this redshift catalog or the previous redshift catalog;
                eligibility_prev[Merger_prev_main_this_info_lc['HaloIndex']] = False

                
                # version 2: all progenitors of halos belonging to this redshift catalog are marked ineligible 
                # run version 1 AND 2 to mark ineligible Merger_next objects to avoid multiple entries
                # optimize with numba if possible (ask Lehman)
                # Note that some progenitor indices are zeros;
                # For best result perhaps combine Progs with MainProgs 
                if "Progenitors" in fields_mt:
                    nums = Merger_this_info_lc['NumProgenitors'][bool_star_this_info_lc]
                    starts = Merger_this_info_lc['StartProgenitors'][bool_star_this_info_lc]
                    # for testing purposes (remove in final version)
                    main_progs = Merger_this_info_lc['HaloIndex'][bool_star_this_info_lc]
                    # loop around halos that were marked belonging to this redshift catalog
                    for j in range(N_this_star_lc):
                        # select all progenitors
                        start = starts[j]
                        num = nums[j]
                        prog_inds = Progs_this[start : start + num]

                        # remove progenitors with no info
                        prog_inds = progs_inds[prog_inds > 0]
                        if len(prog_inds) == 0: continue

                        # correct halo indices
                        prog_inds = correct_inds(prog_inds, N_halos_slabs_prev, slabs_prev, inds_fn_prev)
                        halo_inds = Merger_prev['HaloIndex'][prog_inds]

                        # test output; remove in final version
                        if j < 100: print(halo_inds, Merger_prev[main_progs[j]])

                        # mark ineligible
                        eligibility_prev[halo_inds] = False

                # information to keep for next redshift considered
                N_next = np.sum(~bool_star_this_info_lc)
                Merger_next = Table(
                    {'HaloIndex': np.zeros(N_next, dtype=Merger_lc['HaloIndex'].dtype),
                     'InterpolatedVelocity': np.zeros(N_next, dtype=(np.float32,3)),
                     'InterpolatedPosition': np.zeros(N_next, dtype=(np.float32,3)),
                     'InterpolatedComoving': np.zeros(N_next, dtype=np.float32)
                    }
                )
                Merger_next['HaloIndex'][:] = Merger_prev_main_this_info_lc['HaloIndex'][~bool_star_this_info_lc]
                Merger_next['InterpolatedVelocity'][:] = Merger_this_info_lc['InterpolatedVelocity'][~bool_star_this_info_lc]
                Merger_next['InterpolatedPosition'][:] = Merger_this_info_lc['InterpolatedPosition'][~bool_star_this_info_lc]
                Merger_next['InterpolatedComoving'][:] = Merger_this_info_lc['InterpolatedComoving'][~bool_star_this_info_lc]
                del Merger_this_info_lc, Merger_prev_main_this_info_lc
                
                if plot:
                    # select the halos in the light cones
                    pos_choice = Merger_lc['InterpolatedPosition']

                    # selecting thin slab
                    pos_x_min = -Lbox/2.+k*(Lbox/n_chunks)
                    pos_x_max = x_min+(Lbox/n_chunks)

                    ijk = 0
                    choice = (pos_choice[:, ijk] >= pos_x_min) & (pos_choice[:, ijk] < pos_x_max)

                    circle_this = plt.Circle(
                        (origins[0][1], origins[0][2]), radius=chi_this, color="g", fill=False
                    )
                    circle_prev = plt.Circle(
                        (origins[0][1], origins[0][2]), radius=chi_prev, color="r", fill=False
                    )

                    # clear things for fresh plot
                    ax = plt.gca()
                    ax.cla()

                    # plot particles
                    ax.scatter(pos_choice[choice, 1], pos_choice[choice, 2], s=0.1, alpha=1., color="dodgerblue")

                    # circles for in and prev
                    ax.add_artist(circle_this)
                    ax.add_artist(circle_prev)
                    plt.xlabel([-1000, 3000])
                    plt.ylabel([-1000, 3000])
                    plt.axis("equal")
                    plt.show()

                gc.collect()
                
                # split the eligibility array over three files for the three chunks it's made up of
                offset = 0
                for idx in inds_fn_prev:
                    eligibility_prev_idx = eligibility_prev[offset:offset+N_halos_slabs_prev[idx]]
                    # combine current information with previously existing
                    if os.path.exists(cat_lc_dir / "tmp" / ("eligibility_prev_z%4.3f_lc%d.%02d.npy"%(z_prev, o, idx))):
                        eligibility_prev_old = np.load(cat_lc_dir / "tmp" / ("eligibility_prev_z%4.3f_lc%d.%02d.npy"%(z_prev, o, idx)))
                        eligibility_prev_idx = eligibility_prev_old & eligibility_prev_idx
                        print("Exists!")
                    else:
                        print("Doesn't exist")
                    np.save(cat_lc_dir / "tmp" / ("eligibility_prev_z%4.3f_lc%d.%02d.npy"%(z_prev, o, idx)), eligibility_prev_idx)
                    offset += N_halos_slabs_prev[idx]

                # write as table the information about halos that are part of next loaded redshift
                save_asdf(Merger_next, ("Merger_next_z%4.3f_lc%d.%02d"%(z_prev, o, k)), header, cat_lc_dir / "tmp")

                # save redshift of catalog that is next to load and difference in comoving between this and prev
                # TODO: save as txt file that gets appended to and then read the last line
                with open(cat_lc_dir / "tmp" / "tmp.log", "a") as f:
                    f.writelines(["# Next iteration: \n", "z_prev = %.8f \n"%z_prev, "delta_chi = %.8f \n"%delta_chi, "light_cone = %d \n"%o, "super_slab = %d \n"%k])
                
            del Merger_this, Merger_prev

        # update values for difference in comoving distance
        delta_chi_old = delta_chi

# dict_keys(['HaloIndex', 'HaloMass', 'HaloVmax', 'IsAssociated', 'IsPotentialSplit', 'MainProgenitor', 'MainProgenitorFrac', 'MainProgenitorPrec', 'MainProgenitorPrecFrac', 'NumProgenitors', 'Position', 'Progenitors'])
    

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_start', help='Initial redshift where we start building the trees', default=DEFAULTS['z_start'])
    parser.add_argument('--z_stop', help='Final redshift (inclusive)', default=DEFAULTS['z_stop'])
    parser.add_argument('--merger_parent', help='Merger tree directory', default=DEFAULTS['merger_parent'])
    parser.add_argument('--catalog_parent', help='Light cone catalog directory', default=DEFAULTS['catalog_parent'])
    parser.add_argument('--resume', help='Resume the calculation from the checkpoint on disk', action='store_true')
    parser.add_argument('--plot', help='Want to show plots', action='store_true')
    
    args = vars(parser.parse_args())
    main(**args)
