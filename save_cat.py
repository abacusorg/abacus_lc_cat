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
from astropy.table import Table

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog, user_dt, clean_dt_progen
from tools.aid_asdf import save_asdf, reindex_pid, reindex_pid_pos_vel, reindex_pid_pos_vel_AB, reindex_pid_AB
from tools.merger import simple_load, get_halos_per_slab, get_zs_from_headers, get_halos_per_slab_origin, extract_superslab, extract_superslab_minified, unpack_inds

# these are probably just for testing; should be removed for production
DEFAULTS = {}
#DEFAULTS['sim_name'] = "AbacusSummit_highbase_c021_ph000"
#DEFAULTS['sim_name'] = "AbacusSummit_highbase_c000_ph100"
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
#DEFAULTS['sim_name'] = "AbacusSummit_huge_c000_ph201"
#DEFAULTS['compaso_parent'] = "/mnt/gosling2/bigsims"
DEFAULTS['compaso_parent'] = "/global/project/projectdirs/desi/cosmosim/Abacus"
#DEFAULTS['catalog_parent'] = "/mnt/gosling1/boryanah/light_cone_catalog/"
DEFAULTS['catalog_parent'] = "/global/cscratch1/sd/boryanah/light_cone_catalog/"
#DEFAULTS['merger_parent'] = "/mnt/gosling2/bigsims/merger"
DEFAULTS['merger_parent'] = "/global/project/projectdirs/desi/cosmosim/Abacus/merger"
DEFAULTS['z_start'] = 0.45#0.350
DEFAULTS['z_stop'] = 0.725#1.625
CONSTANTS = {'c': 299792.458}  # km/s, speed of light

def extract_redshift(fn):
    fn = str(fn)
    redshift = float(fn.split('z')[-1])
    return redshift

def correct_all_inds(halo_ids, N_halo_slabs, slabs, n_superslabs):
    # unpack indices into slabs and inds
    slab_ids, ids = unpack_inds(halo_ids)
        
    # total number of halos in the slabs that we have loaded
    offsets = np.zeros(n_superslabs, dtype=int)
    offsets[1:] = np.cumsum(N_halo_slabs)[:-1]

    # select the halos belonging to given slab
    for counter in range(n_superslabs):
        select = np.where(slab_ids == slabs[counter])[0]
        ids[select] += offsets[counter]
    return ids

def main(sim_name, z_start, z_stop, compaso_parent, catalog_parent, merger_parent, save_pos=False, purge=False, complete=False, want_subsample_B=True):

    compaso_parent = Path(compaso_parent)
    catalog_parent = Path(catalog_parent)
    merger_parent = Path(merger_parent)

    # directory where the CompaSO halo catalogs are saved
    cat_dir = compaso_parent / sim_name / "halos"
    
    # obtain the redshifts of the CompaSO catalogs
    redshifts = glob.glob(os.path.join(cat_dir,"z*"))
    zs_cat = [extract_redshift(redshifts[i]) for i in range(len(redshifts))]
    
    # directory where we save the final outputs
    cat_lc_dir = catalog_parent / sim_name / "halos_light_cones"

    # directory where the merger tree files are kept
    merger_dir = merger_parent / sim_name
    
    # if merger tree redshift information has been saved, load it (if not, save it)
    if not os.path.exists(Path("data_mt") / sim_name / "zs_mt.npy"):
        # all merger tree snapshots and corresponding redshifts
        snaps_mt = sorted(merger_dir.glob("associations_z*.0.asdf"))
        zs_mt = get_zs_from_headers(snaps_mt)
        os.makedirs(Path("data_mt") / sim_name, exist_ok=True)
        np.save(Path("data_mt") / sim_name / "zs_mt.npy", zs_mt)
    zs_mt = np.load(Path("data_mt") / sim_name / "zs_mt.npy")

    # names of the merger tree file for a given redshift
    merger_fns = list(merger_dir.glob("associations_z%4.3f.*.asdf"%zs_mt[0]))
    
    # number of superslabs
    n_superslabs = len(merger_fns)
    print("number of superslabs = ",n_superslabs)
    
    # all redshifts, steps and comoving distances of light cones files; high z to low z
    # remove presaving after testing done (or make sure presaved can be matched with simulation)
    if not os.path.exists(Path("data_headers") / sim_name / "coord_dist.npy") or not os.path.exists(Path("data_headers") / sim_name / "redshifts.npy"):
        zs_all, steps, chis_all = get_lc_info("all_headers")
        os.makedirs(Path("data_headers") / sim_name, exist_ok=True)
        np.save(Path("data_headers") / sim_name / "redshifts.npy", zs_all)
        np.save(Path("data_headers") / sim_name / "coord_dist.npy", chis_all)
    zs_all = np.load(Path("data_headers") / sim_name / "redshifts.npy")
    chis_all = np.load(Path("data_headers") / sim_name / "coord_dist.npy")
    zs_all[-1] = float("%.1f" % zs_all[-1])  # LHG: I guess this is trying to match up to some filename or something?

    # fields to extract from the CompaSO catalogs
    with asdf.open(str(cat_dir / ("z%.3f"%zs_cat[0]) / 'halo_info' / 'halo_info_000.asdf')) as f:
        print(f.keys())
        fields_cat = f['data'].keys()
    # just for testing; remove for final version
    if want_subsample_B:
        fields_cat = ['id', 'npstartA', 'npoutA', 'npstartB', 'npoutB', 'N', 'x_L2com', 'v_L2com', 'sigmav3d_L2com']
        subsample_str = 'AB'
    else:
        fields_cat = ['id', 'npstartA', 'npoutA', 'N', 'x_L2com', 'v_L2com', 'sigmav3d_L2com']
        subsample_str = 'A'
    fields_cat_mp = ['haloindex', 'haloindex_mainprog', 'v_L2com_mainprog', 'N_mainprog']

    # get functions relating chi and z
    chi_of_z = interp1d(zs_all, chis_all)
    z_of_chi = interp1d(chis_all, zs_all)
    
    # initial redshift where we start building the trees
    ind_start = np.argmin(np.abs(zs_mt-z_start))
    ind_stop = np.argmin(np.abs(zs_mt-z_stop))

    # directory where we save the current state
    os.makedirs(cat_lc_dir / "tmp", exist_ok=True)
    if purge:
        # delete the exisiting temporary files
        tmp_files = list((cat_lc_dir / "tmp").glob("haloindex_*"))
        for i in range(len(tmp_files)):
            os.unlink(str(tmp_files[i]))
            
    # loop over each merger tree redshift
    for i in range(ind_start,ind_stop+1):
        
        # starting snapshot
        z_mt = zs_mt[i]
        z_mt_mp = zs_mt[i+1]
        z_cat = zs_cat[np.argmin(np.abs(z_mt-zs_cat))]
        print("Redshift = %.3f %.3f"%(z_mt,z_cat))

        # convert the redshifts into comoving distance
        chi_mt = chi_of_z(z_mt)
        chi_mt_mp = chi_of_z(z_mt_mp)
        
        # catalog directory
        catdir = cat_dir / ("z%.3f"%z_cat)
        
        # names of the merger tree file for this redshift
        merger_fns = list(merger_dir.glob("associations_z%4.3f.*.asdf"%z_mt))
        for counter in range(len(merger_fns)):
            merger_fns[counter] = str(merger_fns[counter])

        # slab indices and number of halos per slab
        N_halo_slabs, slabs = get_halos_per_slab(merger_fns, minified=False)
        N_halo_total = np.sum(N_halo_slabs)
        
        # names of the light cone merger tree file for this redshift
        merger_lc_fns = list((cat_lc_dir / ("z%.3f"%z_mt)).glob("Merger_lc*.asdf"))
        for counter in range(len(merger_lc_fns)):
            merger_lc_fns[counter] = str(merger_lc_fns[counter])

        # slab indices, origins and number of halos per slab
        N_halo_slabs_lc, slabs_lc, origins_lc = get_halos_per_slab_origin(merger_lc_fns, minified=False)

        # total number of halos in this light cone redshift
        N_lc = np.sum(N_halo_slabs_lc)
        print("total number of lc halos = ", N_lc)
        if N_lc == 0: continue

        # create a new dictionary with translations of merger names 
        key_dic = {'HaloIndex': ['index_halo', np.int64],
                   'InterpolatedPosition': ['pos_interp', (np.float32,3)],
                   'InterpolatedVelocity': ['vel_interp', (np.float32,3)],
                   'InterpolatedComoving': ['redshift_interp', np.float32],
                   'LightConeOrigin': ['origin', np.int8],
        }
        
        # Merger_lc should have all fields (compaso + mainprog + interpolated)        
        cols = {fields_cat[i]: np.zeros(N_lc, dtype=(user_dt[fields_cat[i]])) for i in range(len(fields_cat))}
        fields = []
        for i in range(len(fields_cat)):
            fields.append(fields_cat[i])
        for i in range(len(fields_cat_mp)):
            # I think we don't need the main prog info
            #cols[fields_cat_mp[i]] = np.zeros(N_lc, dtype=(clean_dt_progen[fields_cat_mp[i]]))
            fields.append(fields_cat_mp[i])
        for key in key_dic.keys():
            cols[key_dic[key][0]] = np.zeros(N_lc, dtype=key_dic[key][1])
        Merger_lc = Table(cols, copy=False)

        # if we want to complete to z = 0, then turn on complete for z = 0.1 (we don't have shells past that)
        if complete and np.abs(z_mt - 0.1) < 1.e-3:
            save_z0 = True
        else:
            save_z0 = False
        
        # initialize index for filling halo information
        start = 0; file_no = 0

        # offset for correcting halo indices
        offset = 0

        # counts particles
        count = 0
        
        # loop over each superslab
        for k in range(n_superslabs):
            # assert superslab number is correct
            assert slabs[k] == k, "the superslabs are not matching"
            
            # origins for which information is available
            origins_k = origins_lc[slabs_lc == k]

            if len(origins_k) == 0:
                # offset all halos in given superslab
                offset += N_halo_slabs[k]
                continue

            
            # list of halo indices
            halo_info_list = []
            for i in [0, 1, -1]:
                halo_info_list.append(str(catdir / 'halo_info' / ('halo_info_%03d.asdf'%((k+i)%n_superslabs))))

            print(halo_info_list)
            # load the CompaSO catalogs
            if (save_pos or save_z0):
                try:
                    cat = CompaSOHaloCatalog(halo_info_list, load_subsamples=f'{subsample_str:s}_halo_all', fields=fields, unpack_bits=False)
                    loaded_pos = True
                except:
                    cat = CompaSOHaloCatalog(halo_info_list, load_subsamples=f'{subsample_str:s}_halo_pid', fields=fields, unpack_bits=False)
                    loaded_pos = False
            else:
                cat = CompaSOHaloCatalog(halo_info_list, load_subsamples=f'{subsample_str:s}_halo_pid', fields=fields, unpack_bits=False)
                loaded_pos = False
            
            # loop over each observer origin
            for o in origins_k:

                # number of halos in this file
                num = N_halo_slabs_lc[file_no]
                file_no += 1

                print("origin, superslab, N_halo_slabs_lc", o, k, num)
                # skip if none
                if num == 0: continue
                
                # load the light cone arrays
                with asdf.open(cat_lc_dir / ("z%.3f"%z_mt) / ("Merger_lc%d.%02d.asdf"%(o,k)), lazy_load=True, copy_arrays=True) as f:
                    merger_lc = f['data']

                # the files should be congruent
                N_halo_lc = len(merger_lc['HaloIndex'])
                assert N_halo_lc == num, "file order is messed up"
                
                # translate information from this file to the complete array
                for key in merger_lc.keys():
                    Merger_lc[key_dic[key][0]][start:start+num] = merger_lc[key][:]
                    
                # adding information about which lightcone the halo belongs to
                Merger_lc['origin'][start:start+num] = np.repeat(o, num).astype(np.int8)
                
                # halo index and velocity
                halo_ind_lc = Merger_lc['index_halo'][start:start+num]
                halo_ind_lc = correct_all_inds(halo_ind_lc, N_halo_slabs, slabs, n_superslabs)
                halo_ind_lc = (halo_ind_lc - offset)%N_halo_total
                vel_interp_lc = Merger_lc['vel_interp'][start:start+num]

                # correct halo indices
                correction = N_halo_slabs[k] + N_halo_slabs[(k+1)%n_superslabs] + N_halo_slabs[(k-1)%n_superslabs] - N_halo_total
                halo_ind_lc[halo_ind_lc > N_halo_total - N_halo_slabs[(k-1)%n_superslabs]] += correction
                
                # halo table
                halo_table = cat.halos[halo_ind_lc]
                header = cat.header
                N_halos = len(cat.halos)
                print("N_halos = ", N_halos)
                assert N_halos == N_halo_total+correction, "mismatch between halo number in compaso catalog and in merger tree"

                
                # load eligibility information if it exists
                if os.path.exists(cat_lc_dir / "tmp" / ("haloindex_z%4.3f_lc%d.%02d.npy"%(z_mt, o, k))):
                    haloindex_ineligible = np.load(cat_lc_dir / "tmp" / ("haloindex_z%4.3f_lc%d.%02d.npy"%(z_mt, o, k)))

                    # find the halos in halo_table that have been marked ineligible and get rid of them
                    mask_ineligible = np.in1d(halo_table['haloindex'], haloindex_ineligible)
                    # decided this is bad cause of the particle indexing or rather the halo indexing that uses num and then the total number of particles
                    #halo_table = halo_table[mask_ineligible]
                    halo_table['N'][mask_ineligible] = 0
                    halo_table['npstartA'][mask_ineligible] = -999 # note unsigned integer
                    halo_table['npoutA'][mask_ineligible] = 0
                    if want_subsample_B:
                        halo_table['npstartB'][mask_ineligible] = -999 # note unsigned integer
                        halo_table['npoutB'][mask_ineligible] = 0
                    print("percentage surviving halos after eligibility = ", 100.*(1-np.sum(mask_ineligible)/len(mask_ineligible)))

                    
                # load the particle ids
                pid = cat.subsamples['pid']
                if (save_pos or save_z0) and loaded_pos:
                    pos = cat.subsamples['pos']
                    vel = cat.subsamples['vel']

                # reindex npstart and npout for the new catalogs
                npstartA = halo_table['npstartA']
                npoutA = halo_table['npoutA']
                # select the pids in this halo light cone, and index into them starting from 0
                if want_subsample_B:
                    npstartB = halo_table['npstartB']
                    npoutB = halo_table['npoutB']

                    if (save_pos or save_z0) and loaded_pos:
                        pid_new, pos_new, vel_new, npstart_new, npout_new, npout_new_B = reindex_pid_pos_vel_AB(pid, pos, vel, npstartA, npoutA, npstartB, npoutB)
                        del pid, pos, vel
                    else:
                        pid_new, npstart_new, npout_new, npout_new_B = reindex_pid_AB(pid, npstartA, npoutA, npstartB, npoutB)
                        del pid
                    del npstartA, npoutA, npstartB, npoutB
                else:
                    if (save_pos or save_z0) and loaded_pos:
                        pid_new, pos_new, vel_new, npstart_new, npout_new = reindex_pid_pos_vel(pid, pos, vel, npstartA, npoutA)
                        del pid, pos, vel
                    else:
                        pid_new, npstart_new, npout_new = reindex_pid(pid, npstartA, npoutA)
                        del pid
                    del npstartA, npoutA

                # assert that indexing is right
                if want_subsample_B:
                    assert np.sum(npout_new+npout_new_B) == len(pid_new), "mismatching indexing"
                else:
                    assert np.sum(npout_new) == len(pid_new), "mismatching indexing"

                # offset for this superslab and origin
                Merger_lc['npstartA'][start:start+num] = npstart_new + count
                Merger_lc['npoutA'][start:start+num] = npout_new
                if want_subsample_B:
                    Merger_lc['npoutB'][start:start+num] = npout_new_B
                    del npout_new_B
                del npstart_new, npout_new

                # increment number of particles in superslab and origin
                count += len(pid_new)
                
                # create particle array
                if (save_pos or save_z0) and loaded_pos:
                    pid_table = Table({'pid': np.zeros(len(pid_new), pid_new.dtype), 'pos': np.zeros((len(pid_new), 3), pos_new.dtype), 'vel': np.zeros((len(pid_new), 3), vel_new.dtype)})
                    pid_table['pid'] = pid_new
                    pid_table['pos'] = pos_new
                    pid_table['vel'] = vel_new
                    del pid_new, pos_new, vel_new
                else:
                    pid_table = Table({'pid': np.zeros(len(pid_new), pid_new.dtype)})
                    pid_table['pid'] = pid_new
                    del pid_new

                # save the particles
                save_asdf(pid_table, "pid_lc%d.%02d"%(o,k), header, cat_lc_dir / ("z%4.3f"%z_mt))
                del pid_table
                                
                # for halos that did not have interpolation and get the velocity from the halo info files
                not_interp = (np.sum(np.abs(vel_interp_lc), axis=1) - 0.) < 1.e-6
                print("percentage not interpolated = ", 100.*np.sum(not_interp)/len(not_interp))
                vel_interp_lc[not_interp] = halo_table['v_L2com'][not_interp]
                
                # halos with merger tree info (0 for merged or smol, -999 for no info)
                mask_info = halo_table['haloindex_mainprog'] > 0
                print("percentage without merger tree info = ", 100.*(1. - np.sum(mask_info)/len(mask_info)))
                print("percentage of removed halos = ", np.sum(halo_table['N'] == 0) * 100./len(mask_info))
                # I think that it may be possible that because in later redshifts (not z_start of build_mt),
                # we have halos from past times, so it is possible that at some point some halo had merger tree
                # info and then it got lost somewhere
                assert np.sum(~mask_info) + np.sum(not_interp) + np.sum(halo_table['N'] == 0), "Different number of halos with merger tree info and halos that have been interpolated"
                del not_interp
                
                # interpolated velocity v = v1 + (v2-v1)/(chi1-chi2)*(chi-chi2) because -d(chi) = d(eta)
                a_avg = (halo_table['v_L2com'] - halo_table['v_L2com_mainprog'])/(chi_mt_mp - chi_mt)
                v_star = halo_table['v_L2com_mainprog'] + a_avg * (chi_mt_mp - merger_lc['InterpolatedComoving'][:, None])
                vel_interp_lc[mask_info] = v_star[mask_info]
                del a_avg, v_star

                # save the velocity information
                Merger_lc['vel_interp'][start:start+num] = vel_interp_lc
                del vel_interp_lc
                
                # interpolated mass m = m1 + (m2-m1)/(chi1-chi2)*(chi-chi2)
                try:
                    # TESTING
                    mdot = (halo_table['N'].astype(float) - halo_table['N_mainprog'][:, 0].astype(float))/(chi_mt_mp - chi_mt)
                    m_star = halo_table['N_mainprog'][:, 0].astype(float) + mdot * (chi_mt_mp - merger_lc['InterpolatedComoving'])
                    # getting rid of negative masses which occur for halos with mass today = 0 or halos that come from the previous redshift (i.e. 1/2 to 1 and not 1 to 3/2)
                    m_star[m_star < 0.] = 0.
                    m_star = np.round(m_star).astype(halo_table['N'].dtype)
                    halo_table['N'][mask_info] = m_star[mask_info]
                    del m_star, mdot
                except:
                    print("failed for some reason")
                    pass
                
                # copy the rest of the halo fields
                for key in fields_cat:
                    # from the CompaSO fields, those have already been reindexed
                    if key == 'npstartA' or key == 'npoutA': continue
                    if key == 'npstartB' or key == 'npoutB': continue
                    Merger_lc[key][start:start+num] = halo_table[key][:]
                
                # save information about halos that were used in this catalog and have merger tree information
                np.save(cat_lc_dir / "tmp" / ("haloindex_z%4.3f_lc%d.%02d.npy"%(z_mt_mp, o, k)), halo_table['haloindex_mainprog'][mask_info])
                del halo_table, mask_info
                
                # add halos in this file
                start += num
                
            # offset all halos in given superslab
            offset += N_halo_slabs[k]
            del cat

        assert len(Merger_lc['redshift_interp']) == start, "Are you missing some halos?"
        #chi_interp_lc[chi_interp_lc < np.min(chis_all)] = np.min(chis_all)
        Merger_lc['redshift_interp'] = z_of_chi(Merger_lc['redshift_interp'])
        
        # save to files
        save_asdf(Merger_lc, "halo_info_lc", header, cat_lc_dir / ("z%4.3f"%z_mt))
        del Merger_lc

        # loop over each superslab
        file_no = 0
        offset = 0
        for k in range(n_superslabs):
            # origins for which information is available
            origins_k = origins_lc[slabs_lc == k]

            # loop over each observer origin
            for o in origins_k:
                
                with asdf.open(cat_lc_dir / ("z%4.3f"%z_mt) / ("pid_lc%d.%02d.asdf"%(o,k)), lazy_load=True, copy_arrays=True) as f:
                    pid_lc = f['data']['pid'][:]
                    if (save_pos or save_z0) and loaded_pos:
                        pos_lc = f['data']['pos'][:]
                        vel_lc = f['data']['vel'][:]
                if file_no == 0:
                    if (save_pos or save_z0) and loaded_pos:
                        pid_table = Table({'pid': np.zeros(count, pid_lc.dtype), 'pos': np.zeros((count, 3), pos_lc.dtype), 'vel': np.zeros((count, 3), vel_lc.dtype)})
                    else:
                        pid_table = Table({'pid': np.zeros(count, pid_lc.dtype)})

                pid_table['pid'][offset:offset+len(pid_lc)] = pid_lc
                if (save_pos or save_z0) and loaded_pos:
                    pid_table['pos'][offset:offset+len(pid_lc)] = pos_lc
                    pid_table['vel'][offset:offset+len(pid_lc)] = vel_lc
                file_no += 1
                offset += len(pid_lc)
        assert offset == count, "Missing particles somewhere"
        save_asdf(pid_table, "pid_lc", header, cat_lc_dir / ("z%4.3f"%z_mt))

        gc.collect()
        
#dict_keys(['HaloIndex', 'HaloMass', 'HaloVmax', 'IsAssociated', 'IsPotentialSplit', 'MainProgenitor', 'MainProgenitorFrac', 'MainProgenitorPrec', 'MainProgenitorPrecFrac', 'NumProgenitors', 'Position', 'Progenitors'])

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_start', help='Initial redshift where we start building the trees', type=float, default=DEFAULTS['z_start'])
    parser.add_argument('--z_stop', help='Final redshift (inclusive)', type=float, default=DEFAULTS['z_stop'])
    parser.add_argument('--compaso_parent', help='CompaSO directory', default=(DEFAULTS['compaso_parent']))
    parser.add_argument('--catalog_parent', help='Light cone catalog directory', default=(DEFAULTS['catalog_parent']))
    parser.add_argument('--merger_parent', help='Merger tree directory', default=(DEFAULTS['merger_parent']))
    parser.add_argument('--save_pos', help='Want to save positions', action='store_true')
    parser.add_argument('--purge', help='Purge the temporary files', action='store_true')
    parser.add_argument('--complete', help='Save the positions and velocities of particles at z = 0.1 to interpolate to z = 0', action='store_true')
    parser.add_argument('--want_subsample_B', help='If this option is called, will only work with subsample A and exclude B', action='store_false')
    
    args = vars(parser.parse_args())
    main(**args)
