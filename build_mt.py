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

from tools.merger import simple_load, get_slab_halo, extract_superslab, extract_superslab_minified

# these are probably just for testing; should be removed for production
DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_highbase_c000_ph100"  # AbacusSummit_base_c000_ph006
DEFAULTS['z_start'] = 0.3  # 0.8#0.5
DEFAULTS['z_stop'] = 0.65  # 1.25#0.8#0.5


# reorder in terms of their slab number
def reorder_by_slab(fns,minified):
    if minified:
        return sorted(fns, key=extract_superslab_minified)
    else:
        return sorted(fns, key=extract_superslab)


# read redshifts from merger tree files
def get_zs_from_headers(snap_names):
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
    N, nd = pos1.shape
    
    # Allow pos2 to be a single point
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


# unpack indices in Sownak's format of Nslice*1e12 + superSlabNum*1e9 + halo_position_superSlab
def unpack_inds(halo_ids):
    id_factor = int(1e12)
    slab_factor = int(1e9)
    index = (halo_ids % slab_factor).astype(int)
    slab_number = ((halo_ids % id_factor - index) // slab_factor).astype(int)
    return slab_number, index


# reorder indices: for given halo index array with corresponding n halos and slabs for its time epoch
def correct_inds(halo_ids, N_halos_slabs, slabs, start=0, stop=None, copies=1):
    # unpack indices
    slab_ids, ids = unpack_inds(halo_ids)

    # total number of halos in the slabs that we have loaded
    N_halos = np.sum(N_halos_slabs[start:stop])

    # set up offset array for all files
    offsets_all = np.zeros(len(slabs), dtype=int)
    offsets_all[1:] = np.cumsum(N_halos_slabs)[:-1]

    # select the halos belonging to given slab
    # offset = 0
    for i in range(start, stop):
        select = np.where(slab_ids == slabs[i])[0]
        ids[select] += offsets_all[i]
        # offset += N_halos_slabs[i]

    # add additional offset from multiple copies
    if copies == 2:
        # ids[:N_halos] += 0*N_halos
        ids[N_halos : 2 * N_halos] += 1 * N_halos
    elif copies == 3:
        ids[N_halos : 2 * N_halos] += 1 * N_halos
        ids[2 * N_halos : 3 * N_halos] += 2 * N_halos

    return ids


# load merger tree and progenitors information
def get_mt_info(fns, fields, origin, minified, start=0, stop=None, copies=1):

    # if we are loading all progenitors and not just main
    if "Progenitors" in fields:
        merger_tree, progs = simple_load(fns[start:stop], fields=fields)
    else:
        merger_tree = simple_load(fns[start:stop], fields=fields)

    # tuks get rid of this when actually working with different halo origins
    # if a far redshift, need 2 copies only
    if copies == 2:
        merger_tree0 = merger_tree
        merger_tree1 = merger_tree.copy()
        merger_tree1["Position"] += np.array([0, 0, Lbox])
        merger_tree2 = merger_tree.copy()
        merger_tree2["Position"] += np.array([0, Lbox, 0])
        merger_tree = np.hstack((merger_tree1, merger_tree2))

    # if in intermediate redshift range, need 3 copies
    elif copies == 3:
        merger_tree0 = merger_tree
        merger_tree1 = merger_tree.copy()
        merger_tree1["Position"] += np.array([0, 0, Lbox])
        merger_tree2 = merger_tree.copy()
        merger_tree2["Position"] += np.array([0, Lbox, 0])
        merger_tree = np.hstack((merger_tree0, merger_tree1, merger_tree2))

    # get number of halos in each slab and number of slabs
    N_halos_slabs, slabs = get_slab_halo(fns, minified)

    
    # load positions in Mpc/h, index of the main progenitors, index of halo
    pos = merger_tree["Position"]
    main_prog = merger_tree["MainProgenitor"]
    halo_ind = merger_tree["HaloIndex"]
    
    
    # compute comoving distance between observer and every halo
    com_dist = dist(pos, origin)
    # com_dist = dist(pos,origin,L=Lbox)  # periodic version

    # if loading all progenitors
    if "Progenitors" in fields:
        num_progs = merger_tree["NumProgenitors"]
        # get an array with the starting indices of the progenitors array
        start_progs = np.zeros(merger_tree.shape, dtype=int)
        start_progs[1:] = num_progs.cumsum()[:-1]

        return (
            com_dist,
            main_prog,
            halo_ind,
            pos,
            start_progs,
            num_progs,
            progs,
            N_halos_slabs,
            slabs,
        )

    return com_dist, main_prog, halo_ind, pos, N_halos_slabs, slabs


# solve when the crossing of the light cones occurs and the interpolated position and velocity
def solve_crossing(r1, r2, pos1, pos2, chi1, chi2, Lbox, origin):
    # identify where the distance between this object and its main progenitor is larger than half the boxsize (or really even 4 Mpc/h since that is Sownak's boundary)
    delta_pos = np.abs(pos2 - pos1)
    delta_pos = np.where(delta_pos > 0.5 * Lbox, (delta_pos - Lbox), delta_pos)
    delta_sign = np.sign(pos1 - pos2)

    # move the halos so that you keep things continuous
    pos1 = pos2 + delta_sign * delta_pos
    r1 = dist(pos1, origin)
    r2 = dist(pos2, origin)

    # solve for eta_star, where chi = eta_0-eta
    # equation is r1+(chi1-chi)/(chi1-chi2)*(r2-r1) = chi
    # with solution chi_star = (r1(chi1-chi2)+chi1(r2-r1))/((chi1-chi2)+(r2-r1))
    chi_star = (r1 * (chi1 - chi2) + chi1 * (r2 - r1)) / ((chi1 - chi2) + (r2 - r1))

    # get interpolated positions of the halos
    v_avg = (pos2 - pos1) / (chi1 - chi2)  # og
    pos_star = pos1 + v_avg * (chi1 - chi_star[:, None])

    # interpolated velocity [km/s]
    vel_star = v_avg * c  # vel1+a_avg*(chi1-chi_star)

    # mark True if closer to chi2 (this snapshot)
    bool_star = np.abs(chi1 - chi_star) > np.abs(chi2 - chi_star)

    # condition to check whether halo in this light cone band
    # assert np.sum((chi_star > chi1) | (chi_star < chi2)) == 0, "Solution is out of bounds"

    
    return chi_star, pos_star, vel_star, bool_star


def get_one_header(merger_dir):
    '''Get an example header by looking at one association file in a merger directory'''
    fn = list(merger_dir.glob('associations*.asdf'))[0]
    with asdf.open(fn) as af:
        header = af['header']
    return header


def main(sim_name, z_start, z_stop, resume=False, plot=False):
    # speed of light
    global c
    c = 299792.458  # km/s
    
    merger_parent = Path("/mnt/gosling2/bigsims/merger")
    merger_dir = merger_parent / sim_name
    header = get_one_header(merger_dir)
    
    # simulation parameters
    Lbox = header['BoxSize']
    # location of the LC origins in Mpc/h
    origins = np.array(header['LightConeOrigins']).reshape(-1,3)
    origin = origins[0]

    # directory where we save the final outputs
    cat_lc_dir = Path("/mnt/gosling1/boryanah/light_cone_catalog/") / sim_name / "halos_light_cones/"
    os.makedirs(cat_lc_dir, exist_ok=True)

    # directory where we save the current state if we want to resume
    os.makedirs(cat_lc_dir / "tmp", exist_ok=True)

    # all redshifts, steps and comoving distances of light cones files; high z to low z
    # LHG: these have to be recomputed per-sim. How is chi being determined? etaK from header?
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


    # fields are we extracting from the merger trees
    # fields_mt = ['HaloIndex','HaloMass','Position','MainProgenitor','Progenitors','NumProgenitors']
    # lighter version
    fields_mt = ["HaloIndex", "Position", "MainProgenitor"]

    # redshift of closest point on wall between original and copied box
    z1 = z_of_chi(0.5 * Lbox - origin[0])
    # redshift of closest point where all three boxes touch
    # z2 = z_of_chi((0.5*Lbox-origin[0])*np.sqrt(2))
    # furthest point where all three boxes touch; TODO: I think that's what we need
    z2 = z_of_chi((0.5 * Lbox - origin[0]) * np.sqrt(3))

    # corresponding indices
    ind_start = np.argmin(np.abs(zs_mt - z_start))
    ind_stop = np.argmin(np.abs(zs_mt - z_stop))

    # initialize difference between the conformal time of last two shells
    delta_chi_old = 0.0

    # loop over each merger tree redshift
    # LHG
    # The algorithm: for each merger tree epoch, for each superslab, for each light cone origin,
    # compute the intersection of the light cone with each halo, using the interpolated position
    # to the previous merger epoch (and possibly a velocity correction).  If the intersection is
    # between the current and previous merger epochs, then record the closer one as that halo's
    # epoch and mark its progenitors as ineligible.
    # Will need one padding superslab in the previous merger epoch.  Can process in a rolling fashion.
    for i in range(ind_start, ind_stop + 1):

        # this snapshot and previous
        z_this = zs_mt[i]
        z_prev = zs_mt[i + 1]
        print("redshift of this and previous snapshot = ", z_this, z_prev)

        # up to z1, we work with original box
        if z_this < z1:
            copies_this = 1
        # after z2, we work with 2 copies of the box
        elif z_this > z2:
            copies_this = 2
        # between z1 and z2, we work with 3 copies of the box
        else:
            copies_this = 3

        # up to z1, we work with original box
        if z_prev < z1:
            copies_prev = 1
        # after z2, we work with 2 copies of the box
        elif z_prev > z2:
            copies_prev = 2
        # between z1 and z2, we work with 3 copies of the box; could be improved needs testing
        else:
            copies_prev = 3

        print(
            "copies of the box needed for this and previous snapshot = ",
            copies_this,
            copies_prev,
        )

        # number of copies should be same and equal to max of the two; repetitive
        copies_this = max(copies_this, copies_prev)
        copies_prev = max(copies_this, copies_prev)

        # previous redshift, distance between shells and copies
        if resume:
            z_this_tmp, delta_chi_old, copies_old = np.load(cat_lc_dir / "tmp" / "z_prev_delta_copies.npy")
            assert (
                np.abs(z_this - z_this_tmp) < 1.0e-6
            ), "Your recorded state is not for the correct redshift, can't resume from old"

        # what is the coordinate distance of the light cone at that redshift and the previous
        assert z_this >= np.min(
            zs_all
        ), "You need to set starting redshift to the smallest value of the merger tree"
        chi_this = chi_of_z(z_this)
        chi_prev = chi_of_z(z_prev)
        delta_chi = chi_prev - chi_this
        print("comoving distance between this and previous snapshot = ", delta_chi)

        # read merger trees file names at this and previous snapshot
        fns_this = merger_dir.glob(f'associations_z{z_this:4.3f}.*.asdf.minified')
        fns_prev = merger_dir.glob(f'associations_z{z_prev:4.3f}.*.asdf.minified')
        fns_this = list(fns_this)
        fns_prev = list(fns_prev)
        minified = True
        
        if len(list(fns_this)) == 0 or len(list(fns_prev)) == 0:
            fns_this = merger_dir.glob(f'associations_z{z_this:4.3f}.*.asdf')
            fns_prev = merger_dir.glob(f'associations_z{z_prev:4.3f}.*.asdf')
            fns_this = list(fns_this)
            fns_prev = list(fns_prev)
            minified = False

        for counter in range(len(fns_this)):
            fns_this[counter] = str(fns_this[counter])
            fns_prev[counter] = str(fns_prev[counter])


        print("number of files = ", len(list(fns_this)), len(list(fns_prev)))
        # number of chunks
        n_chunks = len(list(fns_this))
        assert n_chunks == len(list(fns_prev)), "Incomplete merger tree files"

        # reorder file names by super slab number
        fns_this = reorder_by_slab(fns_this,minified)
        fns_prev = reorder_by_slab(fns_prev,minified)
        
        # starting and finishing superslab chunks; it is best to use all
        start_this = 0
        stop_this = n_chunks
        start_prev = 0
        stop_prev = n_chunks

        # tuks perhaps repeat for multiple origins
        # get comoving distance and other merger tree data for this snapshot and for the previous one
        if "Progenitors" in fields_mt:
            (
                Merger_this
                Progs_this,
                N_halos_slabs_this,
                slabs_this,
            ) = get_mt_info(
                fns_this,
                fields=fields_mt,
                origin=origin,
                minified=minified,
                start=start_this,
                stop=stop_this,
                copies=copies_this,
            )
            (
                Merger_prev,
                Progs_prev,
                N_halos_slabs_prev,
                slabs_prev
            ) = get_mt_info(
                fns_prev,
                fields=fields_mt,
                origin=origin,
                minified=minified,
                start=start_prev,
                stop=stop_prev,
                copies=copies_prev,
            )
        else:
            (
                Merger_this,
                N_halos_slabs_this,
                slabs_this,
            ) = get_mt_info(
                fns_this,
                fields=fields_mt,
                origin=origin,
                minified=minified,
                start=start_this,
                stop=stop_this,
                copies=copies_this,
            )
            (
                Merger_prev,
                N_halos_slabs_prev,
                slabs_prev,
            ) = get_mt_info(
                fns_prev,
                fields=fields_mt,
                origin=origin,
                minified=minified,
                start=start_prev,
                stop=stop_prev,
                copies=copies_prev,
            )

        # number of halos in this step and previous step; this depends on the number of copies and files requested
        N_halos_this = Merger_this.shape[0]
        N_halos_prev = Merger_prev.shape[0]
        print("N_halos_this = ", N_halos_this)
        print("N_halos_prev = ", N_halos_prev)

        # if eligible, can be selected for light cone redshift catalog;
        if i != ind_start or resume:
            # load last state if resuming
            if resume:
                eligibility_this = np.load(cat_lc_dir / "tmp" / "eligibility_prev.npy")
            # needs more copies if transitioning from 1 to 3 and 3 to 2 intersections
            # tuks write out for 3 light cones
            if copies_old == 1 and copies_this == 3:
                eligibility_this = np.hstack(
                    (eligibility_this, eligibility_this, eligibility_this)
                )
            elif copies_old == 3 and copies_this == 2:
                len_copy = int(len(eligibility_this) // 3)
                # overlap can only be in copy 1 or copy 3 (draw it)
                eligibility_this = np.hstack(
                    (eligibility_this[:len_copy], eligibility_this[-len_copy:])
                )
        # all start as eligible
        else:
            eligibility_this = np.ones(N_halos_this, dtype=bool)

        # for a newly opened redshift, everyone is eligible to be part of the light cone catalog
        eligibility_prev = np.ones(N_halos_prev, dtype=bool)

        # mask where no merger tree info is available or halos that are not eligible (because we don'to need to solve for eta star for those)
        mask_noinfo_this = (main_prog_this <= 0) | (~eligibility_this)
        mask_info_this = (
            ~mask_noinfo_this
        )  # todo: revise eligibility etc

        # print percentage where no information is available or halo not eligible
        print(
            "percentage no info or ineligible = ",
            np.sum(mask_noinfo_this) / len(mask_noinfo_this) * 100.0,
        )

        # no info is denoted by 0 or -999 (or regular if ineligible), but -999 messes with unpacking, so we set it to 0
        main_prog_this[mask_noinfo_this] = 0

        # rework the main progenitor and halo indices to return in proper order
        main_prog_this = correct_inds(
            main_prog_this,
            N_halos_slabs_prev,
            slabs_prev,
            start=start_prev,
            stop=stop_prev,
            copies=copies_prev,
        )
        halo_ind_this = correct_inds(
            halo_ind_this,
            N_halos_slabs_this,
            slabs_this,
            start=start_this,
            stop=stop_this,
            copies=copies_this,
        )
        halo_ind_prev = correct_inds(
            halo_ind_prev,
            N_halos_slabs_prev,
            slabs_prev,
            start=start_prev,
            stop=stop_prev,
            copies=copies_prev,
        )

        # we only use this when loading incomplete merger trees because we're missing data
        if stop_this != n_chunks:
            mask_noinfo_this[main_prog_this > N_halos_prev] = False
            main_prog_this[main_prog_this > N_halos_prev] = 0

        # positions and comoving distances of main progenitor halos corresponding to the halos in current snapshot
        pos_prev_main_this = pos_prev[main_prog_this]
        com_dist_prev_main_this = com_dist_prev[main_prog_this]
        halo_ind_prev_main_this = halo_ind_prev[main_prog_this]

        # halos that have merger tree information
        pos_this_info = pos_this[mask_info_this]
        com_dist_this_info = com_dist_this[mask_info_this]
        halo_ind_this_info = halo_ind_this[mask_info_this]
        pos_prev_main_this_info = pos_prev_main_this[mask_info_this]
        com_dist_prev_main_this_info = com_dist_prev_main_this[mask_info_this]
        halo_ind_prev_main_this_info = halo_ind_prev_main_this[mask_info_this]
        eligibility_this_info = eligibility_this[mask_info_this]
        if "Progenitors" in fields_mt:
            start_progs_this_info = start_progs_this[mask_info_this]
            num_progs_this_info = num_progs_this[mask_info_this]

        # halos that don't have merger tree information
        pos_this_noinfo = pos_this[mask_noinfo_this]
        com_dist_this_noinfo = com_dist_this[mask_noinfo_this]
        halo_ind_this_noinfo = halo_ind_this[mask_noinfo_this]
        eligibility_this_noinfo = eligibility_this[mask_noinfo_this]
        # pos_prev_main_this_noinfo = pos_prev_main_this[mask_noinfo_this]
        # com_dist_prev_main_this_noinfo = com_dist_prev_main_this[mask_noinfo_this]

        # select objects that are crossing the light cones
        # TODO: revise conservative choice if stranded between two ( & \) less conservative ( | \ )
        mask_lc_this_info = (
            ((com_dist_this_info > chi_this) & (com_dist_this_info <= chi_prev))
        ) & (
            eligibility_this_info
        )  # | \
        # ((com_dist_prev_main_this_info > chi_this) & (com_dist_prev_main_this_info <= chi_prev))) & (eligibility_this_info)

        mask_lc_this_noinfo = (
            (com_dist_this_noinfo >= chi_this - delta_chi_old / 2.0)
            & (com_dist_this_noinfo < chi_this + delta_chi / 2.0)
        ) & (eligibility_this_noinfo)
        # why not just remove those at the beginning

        # percentage of objects that are part of this or previous snapshot
        print(
            "percentage of halos in light cone with and without progenitor info = ",
            np.sum(mask_lc_this_info) / len(mask_lc_this_info) * 100.0,
            np.sum(mask_lc_this_noinfo) / len(mask_lc_this_noinfo) * 100.0,
        )

        # select halos with mt info that have had a light cone crossing
        pos_this_info_lc = pos_this_info[mask_lc_this_info]
        com_dist_this_info_lc = com_dist_this_info[mask_lc_this_info]
        pos_prev_main_this_info_lc = pos_prev_main_this_info[mask_lc_this_info]
        com_dist_prev_main_this_info_lc = com_dist_prev_main_this_info[mask_lc_this_info]
        halo_ind_prev_main_this_info_lc = halo_ind_prev_main_this_info[mask_lc_this_info]
        halo_ind_this_info_lc = halo_ind_this_info[mask_lc_this_info]
        eligibility_this_info_lc = eligibility_this_info[mask_lc_this_info]
        if "Progenitors" in fields_mt:
            start_progs_this_info_lc = start_progs_this_info[mask_lc_this_info]
            num_progs_this_info_lc = num_progs_this_info[mask_lc_this_info]

        if plot:
            x_min = -500.
            x_max = x_min+10.

            x = pos_this_info_lc[:,0]
            choice = (x > x_min) & (x < x_max)

            y = pos_this_info_lc[choice,1]
            z = pos_this_info_lc[choice,2]

            plt.figure(1)
            plt.scatter(y,z,color='dodgerblue',s=0.1,label='current objects')

            plt.legend()
            plt.axis('equal')
            plt.savefig("this.png")
        
            x = pos_prev_main_this_info_lc[:,0]
            choice = (x > x_min) & (x < x_max)
        
            y = pos_prev_main_this_info_lc[choice,1]
            z = pos_prev_main_this_info_lc[choice,2]

            plt.figure(2)
            plt.scatter(y,z,color='orangered',s=0.1,label='main progenitor')
            
            plt.legend()
            plt.axis('equal')
            plt.savefig("prev.png")
            plt.show()
        
        # select halos without mt info that have had a light cone crossing
        pos_this_noinfo_lc = pos_this_noinfo[mask_lc_this_noinfo]
        halo_ind_this_noinfo_lc = halo_ind_this_noinfo[mask_lc_this_noinfo]
        # com_dist_this_noinfo_lc = com_dist_this_noinfo[mask_lc_this_noinfo]
        # pos_prev_main_this_noinfo_lc = pos_prev_main_this_noinfo[mask_lc_this_noinfo]
        # com_dist_prev_main_this_noinfo_lc = com_dist_prev_main_this_noinfo[mask_lc_this_noinfo]
        # eligibility_this_noinfo_lc = eligibility_this_noinfo[mask_lc_this_noinfo]

        # save the position and (dummy) velocity of the halos in the light cone without progenitor information
        pos_star_this_noinfo_lc = pos_this_noinfo_lc
        vel_star_this_noinfo_lc = pos_this_noinfo_lc * 0.0
        chi_star_this_noinfo_lc = np.ones(pos_this_noinfo_lc.shape[0]) * chi_this

        # REMOVE ME record to test later
        objs = [
            com_dist_prev_main_this_info_lc,
            com_dist_this_info_lc,
            pos_prev_main_this_info_lc,
            pos_this_info_lc,
            chi_prev,
            chi_this,
        ]
        for k in range(len(objs)):
            np.save("%d.npy" % k, objs[k])

        # get chi star where lc crosses halo trajectory; bool is False where closer to previous
        (
            chi_star_this_info_lc,
            pos_star_this_info_lc,
            vel_star_this_info_lc,
            bool_star_this_info_lc,
        ) = solve_crossing(
            com_dist_prev_main_this_info_lc,
            com_dist_this_info_lc,
            pos_prev_main_this_info_lc,
            pos_this_info_lc,
            chi_prev,
            chi_this,
            Lbox,
            origin
        )

        # add ineligible halos if any from last iteration of the loop to those crossed in previous
        # marked ineligible (False) only if same halo has already been assigned to a light cone
        bool_elig_star_this_info_lc = (bool_star_this_info_lc) & (eligibility_this_info_lc)

        # number of objects in light cone
        N_this_star_lc = np.sum(bool_elig_star_this_info_lc)
        N_this_noinfo_lc = np.sum(mask_lc_this_noinfo)
        N_lc = N_this_star_lc + N_this_noinfo_lc

        print(
            "in this snapshot: interpolated, no info, total = ",
            N_this_star_lc * 100.0 / N_lc,
            N_this_noinfo_lc * 100.0 / N_lc,
            N_lc,
        )

        # start new arrays for final output (assuming it is in this snapshot and not in previous)
        pos_interp_lc = np.zeros((N_lc, 3))
        vel_interp_lc = np.zeros((N_lc, 3))
        chi_interp_lc = np.zeros(N_lc, dtype=np.float)
        halo_ind_lc = np.zeros(N_lc, dtype=int)

        # record interpolated position and velocity
        pos_interp_lc[:N_this_star_lc] = pos_star_this_info_lc[bool_elig_star_this_info_lc]
        vel_interp_lc[:N_this_star_lc] = vel_star_this_info_lc[bool_elig_star_this_info_lc]
        halo_ind_lc[:N_this_star_lc] = halo_ind_this_info_lc[bool_elig_star_this_info_lc]
        chi_interp_lc[:N_this_star_lc] = chi_star_this_info_lc[bool_elig_star_this_info_lc]
        pos_interp_lc[-N_this_noinfo_lc:] = pos_star_this_noinfo_lc
        vel_interp_lc[-N_this_noinfo_lc:] = vel_star_this_noinfo_lc
        halo_ind_lc[-N_this_noinfo_lc:] = halo_ind_this_noinfo_lc
        chi_interp_lc[-N_this_noinfo_lc:] = chi_star_this_noinfo_lc

        # create directory for this redshift
        os.makedirs(cat_lc_dir / ("z%.3f"%z_this), exist_ok=True)

        # adding contributions from the previous
        if i != ind_start or resume:
            if resume:
                halo_ind_next = np.load(cat_lc_dir / "tmp" / "halo_ind_next.npy")
                pos_star_next = np.load(cat_lc_dir / "tmp" / "pos_star_next.npy")
                vel_star_next = np.load(cat_lc_dir / "tmp" / "vel_star_next.npy")
                chi_star_next = np.load(cat_lc_dir / "tmp" / "chi_star_next.npy")
                resume = False
            
            N_lc += len(halo_ind_next)  # todo improve
            pos_interp_lc = np.vstack((pos_interp_lc, pos_star_next))
            vel_interp_lc = np.vstack((vel_interp_lc, vel_star_next))
            chi_interp_lc = np.hstack((chi_interp_lc, chi_star_next))
            halo_ind_lc = np.hstack((halo_ind_lc, halo_ind_next))

        # save those arrays
        table_lc = np.empty(
            N_lc,
            dtype=[
                ("halo_ind", halo_ind_lc.dtype),
                ("pos_interp", (pos_interp_lc.dtype, 3)),
                ("vel_interp", (vel_interp_lc.dtype, 3)),
                ("chi_interp", chi_interp_lc.dtype),
            ],
        )
        table_lc["halo_ind"] = halo_ind_lc
        table_lc["pos_interp"] = pos_interp_lc
        table_lc["vel_interp"] = vel_interp_lc
        table_lc["chi_interp"] = chi_interp_lc

        np.save(cat_lc_dir / ("z%.3f"%z_this) / "table_lc.npy", table_lc)

        # mark eligibility
        # version 1: only the main progenitor is marked ineligible;
        # record objects assigned to prev and mark ineligible; ~bool is closer to prev
        halo_ind_next = halo_ind_prev_main_this_info_lc[~bool_star_this_info_lc]
        eligibility_prev[halo_ind_next] = False
        # get rid of objects assigned to this snapshot since closer to it
        halo_ind_assign = halo_ind_prev_main_this_info_lc[bool_star_this_info_lc]
        eligibility_prev[halo_ind_assign] = False
        # get rid of objects that have previously been assigned
        halo_ind_inelig = halo_ind_prev_main_this_info_lc[~eligibility_this_info_lc]
        eligibility_prev[halo_ind_inelig] = False

        # version 2: all progenitors are marked ineligible
        # slower, but works (perhaps optimize with numba). Confusing part is why halo_inds has zeros
        # todo: slight issue is that we mask only for main prog - can perhaps combine progs with main prog (main prog may not be in progs)
        if "Progenitors" in fields_mt:
            for j in range(len(start_progs_this_info_lc[~bool_star_this_info_lc])):
                start = (start_progs_this_info_lc[~bool_star_this_info_lc])[j]
                num = (num_progs_this_info_lc[~bool_star_this_info_lc])[j]
                prog_inds = Progs_this[start : start + num]
                prog_inds = correct_inds(prog_inds, N_halos_slabs_prev, slabs_prev)
                halo_inds = halo_ind_prev[prog_inds]
                # if j < 100: print(halo_inds, halo_ind_next[j])
                eligibility_prev[halo_inds] = False

        # information to keep for next redshift considered; should have dimensions equal to sum elig prev
        chi_star_next = chi_star_this_info_lc[~bool_star_this_info_lc]
        vel_star_next = vel_star_this_info_lc[~bool_star_this_info_lc]
        pos_star_next = pos_star_this_info_lc[~bool_star_this_info_lc]

        if plot:
            # select the halos in the light cones
            try:
                pos_choice = pos_this[halo_ind_lc]
            except:
                # not too sure
                pos_choice = pos_this[halo_ind_lc % len_copy]

            # selecting thin slab
            pos_x_min = -490.0
            pos_x_max = -480.0

            ijk = 0
            choice = (pos_choice[:, ijk] >= pos_x_min) & (pos_choice[:, ijk] < pos_x_max)
            # choice_lc = (pos_this_info[:,ijk] >= pos_x_min) & (pos_this_info[:,ijk] < pos_x_max)

            circle_this = plt.Circle(
                (origin[1], origin[2]), radius=chi_this, color="g", fill=False
            )
            circle_prev = plt.Circle(
                (origin[1], origin[2]), radius=chi_prev, color="r", fill=False
            )

            ax = plt.gca()
            ax.cla()  # clear things for fresh plot

            # ax.scatter(pos_this_info[choice_lc,1],pos_this_info[choice_lc,2],s=0.01,alpha=0.5,color='orange')
            ax.scatter(
                pos_choice[choice, 1],
                pos_choice[choice, 2],
                s=0.01,
                alpha=0.5,
                color="dodgerblue",
            )

            # circles for in and prev
            ax.add_artist(circle_this)
            ax.add_artist(circle_prev)
            plt.xlabel([-1000, 3000])
            plt.ylabel([-1000, 3000])
            plt.axis("equal")
            plt.show()

        del (
            com_dist_this,
            main_prog_this,
            halo_ind_this,
            pos_this,
            N_halos_slabs_this,
            slabs_this,
        )
        del (
            com_dist_prev,
            main_prog_prev,
            halo_ind_prev,
            pos_prev,
            N_halos_slabs_prev,
            slabs_prev,
        )
        gc.collect()

        # update values for difference in comoving distance, eligibility and number of copies
        delta_chi_old = delta_chi
        eligibility_this = eligibility_prev
        copies_old = copies_prev

        # save the current state so you can resume
        np.save(cat_lc_dir / "tmp" / "eligibility_prev.npy", eligibility_prev)
        np.save(cat_lc_dir / "tmp" / "halo_ind_next.npy", halo_ind_next)
        np.save(cat_lc_dir / "tmp" / "pos_star_next.npy", pos_star_next)
        np.save(cat_lc_dir / "tmp" / "vel_star_next.npy", vel_star_next)
        np.save(cat_lc_dir / "tmp" / "chi_star_next.npy", chi_star_next)
        np.save(cat_lc_dir / "tmp" / "z_prev_delta_copies.npy",
            np.array([z_prev, delta_chi, copies_prev]),
        )

    # dict_keys(['HaloIndex', 'HaloMass', 'HaloVmax', 'IsAssociated', 'IsPotentialSplit', 'MainProgenitor', 'MainProgenitorFrac', 'MainProgenitorPrec', 'MainProgenitorPrecFrac', 'NumProgenitors', 'Position', 'Progenitors'])
    

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_start', help='Initial redshift where we start building the trees', default=DEFAULTS['z_start'])
    parser.add_argument('--z_stop', help='Final redshift (inclusive)', default=DEFAULTS['z_stop'])
    parser.add_argument('--resume', help='Resume the calculation from the checkpoint on disk', action='store_true')
    parser.add_argument('--plot', help='Want to show plots', action='store_true')
    
    args = vars(parser.parse_args())
    main(**args)
