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

from tools.aid_asdf import save_asdf, load_mt_pid, load_mt_cond_edge, load_mt_dist, load_mt_pid_pos_vel, load_mt_npout, load_mt_npout_B, load_lc_pid_rv, load_mt_origin, reindex_pid_pos_vel
from bitpacked import unpack_rvint, unpack_pids
from tools.merger import get_zs_from_headers, get_one_header
from tools.read_headers import get_lc_info
from tools.match_searchsorted import match, match_srt, match_halo_pids_to_lc_rvint
from tools.InputFile import InputFile

# these are probably just for testing; should be removed for production
DEFAULTS = {}
#DEFAULTS['sim_name'] = "AbacusSummit_highbase_c021_ph000"
#DEFAULTS['sim_name'] = "AbacusSummit_highbase_c000_ph100"
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
#DEFAULTS['sim_name'] = "AbacusSummit_huge_c000_ph201"
#DEFAULTS['light_cone_parent'] = "/mnt/gosling2/bigsims/"
DEFAULTS['light_cone_parent'] = "/global/project/projectdirs/desi/cosmosim/Abacus"
#DEFAULTS['catalog_parent'] = "/mnt/gosling1/boryanah/light_cone_catalog/"
DEFAULTS['catalog_parent'] = "/global/cscratch1/sd/boryanah/light_cone_catalog/"
#DEFAULTS['merger_parent'] = "/mnt/gosling2/bigsims/merger/"
DEFAULTS['merger_parent'] = "/global/project/projectdirs/desi/cosmosim/Abacus/merger"
DEFAULTS['z_lowest'] = 0.350
DEFAULTS['z_highest'] = 1.064# 0.991 # TESTING # 1.625

def get_mt_fns(z_th, zs_mt, chis_mt, cat_lc_dir):
    """
    Return the mt catalog names straddling the given redshift
    """
    for k in range(len(zs_mt-1)):
        squish = (zs_mt[k] <= z_th) & (z_th <= zs_mt[k+1])
        if squish == True: break 
    z_low = zs_mt[k]
    z_high = zs_mt[k+1]
    chi_low = chis_mt[k]
    chi_high = chis_mt[k+1]
    fn_low = cat_lc_dir / ("z%.3f/pid_lc.asdf"%(z_low))
    fn_high = cat_lc_dir / ("z%.3f/pid_lc.asdf"%(z_high))
    halo_fn_low = cat_lc_dir / ("z%.3f/halo_info_lc.asdf"%(z_low))
    halo_fn_high = cat_lc_dir / ("z%.3f/halo_info_lc.asdf"%(z_high))

    mt_fns = [fn_high, fn_low]
    mt_zs = [z_high, z_low]
    mt_chis = [chi_high, chi_low]
    halo_mt_fns = [halo_fn_high, halo_fn_low]

    return mt_fns, mt_zs, mt_chis, halo_mt_fns

def extract_steps(fn):
    """
    Return the step number as integer from a light cone file
    """
    split_fn = fn.split('Step')[1]
    step = np.int(split_fn.split('.asdf')[0])
    return step

def main(sim_name, z_lowest, z_highest, light_cone_parent, catalog_parent, merger_parent, resume=False, want_subsample_B=True):
    """
    Main algorithm: for each step in the light cone files, figure out the two closest halo light cone catalogs and load relevant information
    from these into the "currently_loaded" lists (if the step is far from the midpoint between the two, load only a single redshift catalog). 
    Then figure out which are the step files associated with the current step (1 to 3) and load the redshift catalogs corresponding to this
    step (1 or 2) from the "currently_loaded" lists. Then consider all combinations of light cone origins and redshift catalog origins 
    (the largest overlap will be for 0 and 0, 1 and 1, 2 and 2, but there will be a small number of halos on the boundary between the 
    original box and the two copies, so this is an effort to find particles that have migrated across the border). To speed up the process
    of matching the halo particles to the light cone particles, we have included another condition that selects only those particles in the
    halo light cone catalog that are a distance from the observer of only +/- 10 Mpc/h around the mean comoving distance of the current step.
    """
    # turn light cone, halo catalog and merger tree paths into Path objects
    light_cone_parent = Path(light_cone_parent)
    catalog_parent = Path(catalog_parent)
    merger_parent = Path(merger_parent)

    # directory where the merger tree files are kept
    merger_dir = merger_parent / sim_name
    header = get_one_header(merger_dir)

    # physical location of the observer (original box origin)
    observer_origin = (np.array(header['LightConeOrigins']).reshape(-1,3))[0]
    print("observer origin = ", observer_origin)
    
    # simulation parameters
    Lbox = header['BoxSize']
    PPD = header['ppd']
    
    # directory where we have saved the final outputs from merger trees and halo catalogs
    cat_lc_dir = catalog_parent / sim_name / "halos_light_cones"

    # directory where light cones are saved
    lc_dir = light_cone_parent / sim_name / "lightcones"
    
    # all redshifts, steps and comoving distances of light cones files; high z to low z
    # remove presaving after testing done
    if not os.path.exists(Path("data_headers") / sim_name / "coord_dist.npy") or not os.path.exists(Path("data_headers") / sim_name / "redshifts.npy") or not os.path.exists(Path("data_headers") / sim_name / "steps.npy"):
        zs_all, steps, chis_all = get_lc_info("all_headers")
        os.makedirs(Path("data_headers") / sim_name, exist_ok=True)
        np.save(Path("data_headers") / sim_name / "redshifts.npy", zs_all)
        np.save(Path("data_headers") / sim_name / "steps.npy", steps_all)
        np.save(Path("data_headers") / sim_name / "coord_dist.npy", chis_all)
    zs_all = np.load(Path("data_headers") / sim_name / "redshifts.npy")
    steps_all = np.load(Path("data_headers") / sim_name / "steps.npy")
    chis_all = np.load(Path("data_headers") / sim_name / "coord_dist.npy")
    zs_all[-1] = float("%.1f" % zs_all[-1])

    # if merger tree redshift information has been saved, load it (if not, save it)
    if not os.path.exists(Path("data_mt") / sim_name / "zs_mt.npy"):
        # all merger tree snapshots and corresponding redshifts
        snaps_mt = sorted(merger_dir.glob("associations_z*.0.asdf"))
        zs_mt = get_zs_from_headers(snaps_mt)
        os.makedirs(Path("data_mt") / sim_name, exist_ok=True)
        np.save(Path("data_mt") / sim_name / "zs_mt.npy", zs_mt)
    zs_mt = np.load(Path("data_mt") / sim_name / "zs_mt.npy")

    # correct out of bounds error for interpolation 
    zs_mt = zs_mt[(zs_mt <= zs_all.max()) & (zs_mt >= zs_all.min())]

    # time step of furthest and closest shell in the light cone files
    step_min = np.min(steps_all)
    step_max = np.max(steps_all)
    
    # get functions relating chi and z
    chi_of_z = interp1d(zs_all,chis_all)
    z_of_chi = interp1d(chis_all, zs_all)
    
    # conformal distance of the mtree catalogs
    chis_mt = chi_of_z(zs_mt)

    # Read light cone file names
    lc_rv_fns = sorted(glob.glob(os.path.join(lc_dir, 'rv/LightCone*')))
    lc_pid_fns = sorted(glob.glob(os.path.join(lc_dir, 'pid/LightCone*')))
    
    # select the final and initial step for computing the convergence map
    step_start = steps_all[np.argmin(np.abs(zs_all-z_highest))]
    step_stop = steps_all[np.argmin(np.abs(zs_all-z_lowest))]
    print("step_start = ",step_start)
    print("step_stop = ",step_stop)

    # these are the time steps associated with each of the light cone files
    step_fns = np.zeros(len(lc_pid_fns),dtype=int)
    for i in range(len(lc_pid_fns)):
        step_fns[i] = extract_steps(lc_pid_fns[i])

    # directory where we save the current state if we want to resume
    os.makedirs(cat_lc_dir / "tmp", exist_ok=True)
    if resume:
        # check if resuming from an old state
        infile = InputFile(cat_lc_dir / "tmp" / "match.log")
        z_last = infile.z_last
        assert (np.abs(z_last-z_highest) <= 2.e-1), "Your recorded state is not for the currently requested redshift, can't resume from old. Last recorded state is z = %.3f"%z_last
    else:
        z_last = -1
        if os.path.exists(cat_lc_dir / "tmp" / "match.log"):
            os.unlink(cat_lc_dir / "tmp" / "match.log")
        
    # initialize previously loaded mt file name
    currently_loaded_zs = []
    currently_loaded_headers = []
    currently_loaded_npouts = []
    currently_loaded_origins = []
    currently_loaded_pos = []
    currently_loaded_dist = []
    currently_loaded_pids = []
    currently_loaded_tables = []

    # loop through all selected steps
    for step in range(step_start,step_stop+1):
        
        # adjust the indexing using j
        j = step-step_min
        step_this = steps_all[j]
        z_this = zs_all[j]
        chi_this = chis_all[j]
        assert step_this == step, "You've messed up the step counts"
        print("light cones step, redshift = ", step_this, z_this)
        
        # get the two redshifts it's straddling, their file names (of particles and halos), and their comoving values
        mt_fns, mt_zs, mt_chis, halo_mt_fns = get_mt_fns(z_this, zs_mt, chis_mt, cat_lc_dir)

        # get the mean chi
        mt_chi_mean = np.mean(mt_chis)

        # how many shells are we including on both sides, including mid point (total of 2 * buffer_no + 1)
        buffer_no = 1 # 2 # 1 should be enough and it spares time

        # is this the redshift that's closest to the bridge between two redshifts?
        mid_bool = (np.argmin(np.abs(mt_chi_mean-chis_all)) <= j+buffer_no) & (np.argmin(np.abs(mt_chi_mean-chis_all)) >= j-buffer_no)

        # TESTING
        #mid_bool = True
        
        # if not in between two redshifts, we just need one catalog -- the one it is closest to; else keep both
        if not mid_bool:
            mt_fns = [mt_fns[np.argmin(np.abs(mt_chis-chi_this))]]
            halo_mt_fns = [halo_mt_fns[np.argmin(np.abs(mt_chis-chi_this))]] 
            mt_zs = [mt_zs[np.argmin(np.abs(mt_chis-chi_this))]] 
        print("using redshifts = ",mt_zs)

        # if we have loaded two zs but are only using one, that means that we are past the mid-point and can record the first one
        if len(currently_loaded_zs) > len(mt_zs):
            print("We will be dismissing z = ", mt_zs[0])
            dismiss = True
        else:
            dismiss = False
            
        # load the two (or one) straddled merger tree files and store them into lists of currently loaded things; record one of them if it's time
        for i in range(len(mt_fns)):

            # discard the old redshift catalog and record its data
            if dismiss:
                # check whether we are resuming and whether this is the redshift last written into the log file
                if resume and np.abs(currently_loaded_zs[0] - z_last) < 1.e-6:
                    print("This redshift (z = %.3f) has already been recorded, skipping"%z_last)
                else:
                    # save the information about that redshift into asdf file
                    save_asdf(currently_loaded_tables[0], "pid_rv_lc", currently_loaded_headers[0], cat_lc_dir / ("z%4.3f"%currently_loaded_zs[0]))
                    print("saved catalog = ", currently_loaded_zs[0])
                    
                    # record the write-out into the log file
                    with open(cat_lc_dir / "tmp" / "match.log", "a") as f:
                        f.writelines(["# Last saved redshift: \n", "z_last = %.8f \n"%currently_loaded_zs[0]])
                
                # discard this first entry (aka the one being written out) from the lists of currently loaded things
                currently_loaded_zs = currently_loaded_zs[1:]
                currently_loaded_headers = currently_loaded_headers[1:]
                currently_loaded_pids = currently_loaded_pids[1:]
                currently_loaded_origins = currently_loaded_origins[1:]
                currently_loaded_pos = currently_loaded_pos[1:]
                currently_loaded_dist = currently_loaded_dist[1:]
                currently_loaded_npouts = currently_loaded_npouts[1:]
                currently_loaded_tables = currently_loaded_tables[1:]
                gc.collect()

            # check if catalog is already loaded and don't load if so
            if mt_zs[i] in currently_loaded_zs: print("skipped loading catalog ", mt_zs[i]); continue
            
            # load new merger tree catalog
            mt_pid, header = load_mt_pid(mt_fns[i], Lbox, PPD)
            halo_mt_npout = load_mt_npout(halo_mt_fns[i])
            if want_subsample_B:
                halo_mt_npout += load_mt_npout_B(halo_mt_fns[i])
            halo_mt_origin = load_mt_origin(halo_mt_fns[i])
            halo_mt_cond_edge = load_mt_cond_edge(halo_mt_fns[i], Lbox)
            halo_mt_dist = load_mt_dist(halo_mt_fns[i], observer_origin)
            mt_origins = np.repeat(halo_mt_origin, halo_mt_npout)
            mt_cond_edge = np.repeat(halo_mt_cond_edge, halo_mt_npout, axis=0)
            mt_dist = np.repeat(halo_mt_dist, halo_mt_npout, axis=0)
            del halo_mt_origin, halo_mt_npout, halo_mt_cond_edge, halo_mt_dist
            # remove npouts unless applying Lehman's idea
            gc.collect()
            
            # start the light cones table for this redshift
            lc_table_final = Table(
                {'pid': np.zeros(len(mt_pid), dtype=mt_pid.dtype),
                 'pos': np.zeros(len(mt_pid), dtype=(np.float32,3)),
                 'vel': np.zeros(len(mt_pid), dtype=(np.float32,3)),
                 #'redshift': np.zeros(len(mt_pid), dtype=np.float16), # TESTING ask Lehman why float16 not allowed
                }
            )
            
            # append the newly loaded catalog
            currently_loaded_zs.append(mt_zs[i])
            currently_loaded_headers.append(header)
            currently_loaded_pids.append(mt_pid)
            currently_loaded_pos.append(mt_cond_edge)
            currently_loaded_dist.append(mt_dist)
            currently_loaded_origins.append(mt_origins)
            currently_loaded_tables.append(lc_table_final)
            # Useful for Lehman's
            #currently_loaded_npouts.append(halo_mt_npout)
        print("currently loaded redshifts = ",currently_loaded_zs)    
        
        # find all light cone file names that correspond to this time step
        choice_fns = np.where(step_fns == step_this)[0]
        
        # number of light cones at this step
        num_lc = len(choice_fns)
        assert (num_lc <= 3) & (num_lc > 0), "There can be at most three files in the light cones corresponding to a given step"
        
        # loop through those one to three light cone files
        for choice_fn in choice_fns:
            print("light cones file = ",lc_pid_fns[choice_fn])

            # load particles in light cone
            lc_pid, lc_rv = load_lc_pid_rv(lc_pid_fns[choice_fn], lc_rv_fns[choice_fn], Lbox, PPD)

            # sorting to speed up the matching
            i_sort_lc_pid = np.argsort(lc_pid)
            lc_pid = lc_pid[i_sort_lc_pid]
            lc_rv = lc_rv[i_sort_lc_pid]
            del i_sort_lc_pid
            gc.collect()
            
            # what are the offsets for each of the origins
            if 'LightCone1' in lc_pid_fns[choice_fn]:
                offset_lc = np.array([0., 0., Lbox], dtype=np.float32)
                origin = 1
            elif 'LightCone2' in lc_pid_fns[choice_fn]:
                offset_lc = np.array([0., Lbox, 0.], dtype=np.float32)
                origin = 2
            else:
                offset_lc = np.array([0., 0., 0.], dtype=np.float32)
                origin = 0
            
            # loop over the one or two closest catalogs 
            for i in range(len(mt_fns)):
                #if i == 0: continue # TESTING

                # define variables for each of the currently loaded lists
                which_mt = np.where(mt_zs[i] == currently_loaded_zs)[0]
                mt_pid = currently_loaded_pids[which_mt[0]]
                mt_cond_edge = currently_loaded_pos[which_mt[0]]
                mt_dist = currently_loaded_dist[which_mt[0]]
                mt_origins = currently_loaded_origins[which_mt[0]]
                header = currently_loaded_headers[which_mt[0]]
                lc_table_final = currently_loaded_tables[which_mt[0]]
                mt_z = currently_loaded_zs[which_mt[0]]
                # useful for Lehman's
                #halo_mt_npout = currently_loaded_npouts[which_mt[0]]
                
                # which origins are available for this merger tree file
                origins = np.unique(mt_origins)

                # add to main function if it works
                # TESTING adding another condition to reduce the number of particles considered (spatial position of the halos in relation to the particles in the light cone)
                cond_dist = (mt_dist < chi_this + 10.) & (mt_dist > chi_this - 10.)
                #cond_dist = np.ones_like(mt_dist, dtype=bool) # TESTING!!!!!!!!!!!!!!!!
                del mt_dist
                gc.collect()
                if np.sum(cond_dist) == 0:
                    continue
                
                # loop through each of the available origins
                for o in origins:
                    # consider boundary conditions (can probably be sped up if you say if origin 0 and o 1 didn't find anyone, don't check o 0 and o 1, 2
                    if o == origin:
                        condition = mt_origins == o
                    elif origin == 0 and o == 1:
                        condition = (mt_origins == o) & (mt_cond_edge[:, 0])
                    elif origin == 0 and o == 2:
                        condition = (mt_origins == o) & (mt_cond_edge[:, 1])
                    elif origin == 1 and o == 0:
                        condition = (mt_origins == o) & (mt_cond_edge[:, 2])
                    elif origin == 2 and o == 0:
                        condition = (mt_origins == o) & (mt_cond_edge[:, 3])
                    elif origin == 1 and o == 2:
                        continue
                    elif origin == 2 and o == 1:
                        continue
                    condition &= cond_dist
                    if np.sum(condition) == 0:
                        print("skipped", origin, o)
                        continue
                    
                    print("origin and o, percentage of particles = ", origin, o, np.sum(condition)/len(condition))

                    # match the pids in the merger trees and the light cones selected by the above conditions
                    '''
                    # og
                    inds_mt_pid = np.arange(len(mt_pid))[condition]
                    mt_in_lc = match(mt_pid[inds_mt_pid], lc_pid, arr2_sorted=True) #, arr2_index=i_sort_lc_pid) # commented out to spare time
                    comm2 = mt_in_lc[mt_in_lc > -1]
                    comm1 = (np.arange(len(mt_pid), dtype=np.int32)[condition])[mt_in_lc > -1]
                    del condition
                    gc.collect()
                    del mt_in_lc
                    gc.collect()
                    '''
                    # TESTING
                    # match merger tree and light cone pids
                    print("starting")
                    t1 = time.time()
                    comm1, comm2 = match_srt(mt_pid[condition], lc_pid, condition)
                    del condition
                    gc.collect()
                    print("time = ", time.time()-t1)
                    
                    # select the intersected positions and velocities
                    pos_mt_lc, vel_mt_lc = unpack_rvint(lc_rv[comm2], boxsize=Lbox)
                    del comm2
                    gc.collect()
                    
                    # select the intersected pids 
                    pid_mt_lc = mt_pid[comm1]
                    
                    # print percentage of matched pids
                    print("at z = %.3f, matched = "%mt_z, len(comm1)*100./(len(mt_pid)))
                    # original version end
                
                    '''
                    # alternative Lehman implementation start
                    t1 = time.time()
                    comm1, nmatch, hrvint = match_halo_pids_to_lc_rvint(halo_mt_npout, mt_pid, lc_rv, lc_pid)
                    print("at z = %.3f, matched = "%mt_z,len(hrvint)*100./(len(mt_pid)))
                    print("time = ", time.time()-t1)
                
                    pos_mt_lc, vel_mt_lc = unpack_rvint(hrvint,Lbox)
                    pid_mt_lc = mt_pid[comm1]                
                    # alternative Lehman implementation end
                    '''

                    # offset particle positions depending on which light cone we are at
                    pos_mt_lc += offset_lc
                
                    # save the pid, position, velocity and redshift
                    lc_table_final['pid'][comm1] = pid_mt_lc
                    lc_table_final['pos'][comm1] = pos_mt_lc
                    lc_table_final['vel'][comm1] = vel_mt_lc
                    #lc_table_final['redshift'][comm1] = np.full_like(pid_mt_lc, z_this, dtype=np.float16) # TESTING ask Lehman why float16 not allowed
                    del pid_mt_lc, pos_mt_lc, vel_mt_lc, comm1
                    gc.collect()

                del mt_pid, mt_origins, mt_cond_edge, lc_table_final, cond_dist
                gc.collect()

            print("-------------------")
            del lc_pid, lc_rv
            gc.collect()

    # close the two that are currently open
    for i in range(len(currently_loaded_zs)):

        # save the information about that redshift into an asdf
        save_asdf(currently_loaded_tables[0], "pid_rv_lc", currently_loaded_headers[0], cat_lc_dir / ("z%4.3f"%currently_loaded_zs[0]))
        print("saved catalog = ", currently_loaded_zs[0])

        # record to the log file
        with open(cat_lc_dir / "tmp" / "match.log", "a") as f:
            f.writelines(["# Last saved redshift: \n", "z_last = %.8f \n"%currently_loaded_zs[0]])
            
        # discard the first instance from the currently loaded lists of things
        currently_loaded_zs = currently_loaded_zs[1:]
        currently_loaded_headers = currently_loaded_headers[1:]
        currently_loaded_pids = currently_loaded_pids[1:]
        currently_loaded_origins = currently_loaded_origins[1:]
        currently_loaded_pos = currently_loaded_pos[1:]
        currently_loaded_npouts = currently_loaded_npouts[1:]
        currently_loaded_tables = currently_loaded_tables[1:]
        gc.collect()
        
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass
        
if __name__ == '__main__':
    # parser arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_lowest', help='Initial redshift where we start building the trees', type=float, default=DEFAULTS['z_lowest'])
    parser.add_argument('--z_highest', help='Final redshift (inclusive)', type=float, default=DEFAULTS['z_highest'])
    parser.add_argument('--light_cone_parent', help='Light cone output directory', default=(DEFAULTS['light_cone_parent']))
    parser.add_argument('--catalog_parent', help='Light cone catalog directory', default=(DEFAULTS['catalog_parent']))
    parser.add_argument('--merger_parent', help='Merger tree directory', default=(DEFAULTS['merger_parent']))
    parser.add_argument('--resume', help='Resume the calculation from the checkpoint on disk', action='store_true')
    parser.add_argument('--want_subsample_B', help='If this option is called, will only work with subsample A and exclude B', action='store_false')
    args = vars(parser.parse_args())
    main(**args)
