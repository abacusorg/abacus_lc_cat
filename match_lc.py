import glob
import asdf
import numpy as np
import sys
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
import gc
import os

from tools.aid_asdf import save_asdf, load_mt_pid, load_lc_pid_rv
from bitpacked import unpack_rvint, unpack_pids
    
# cosmological parameters
h = 0.6736
H0 = h*100.# km/s/Mpc
Om_m = 0.315192
c = 299792.458# km/s

# simulation parameters
Lbox = 2000. # Mpc/h
PPD = 6912


# initial redshift where we start building the trees
# aiming to get these: 0.575, 0.65, 0.725,
z_lowest = 0.576#0.5
z_highest = 0.725#1.25 # 1.25 # 2.45 # including

# location of the origin in Mpc/h
origin = np.array([-990.,-990.,-990.])

# simulation name
sim_name = "AbacusSummit_base_c000_ph006"

# all merger tree redshifts; from low z to high
zs_mt = np.load("data/zs_mt.npy")[:-1]
#print("mt redshifts = ",zs_mt)

# directory where we have saved the final outputs from merger trees and halo catalogs
cat_lc_dir = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"

# directory where light cones are saved
lc_dir = "/mnt/store2/bigsims/"+sim_name+"/lightcones/"

# all redshifts, steps and comoving distances of light cones files; high z to low z
zs_all = np.load("data_headers/redshifts.npy")
steps_all = np.load("data_headers/steps.npy")
chis_all = np.load("data_headers/coord_dist.npy")
zs_all[-1] = float('%.1f'%zs_all[-1])
#print("light cone zs = ",zs_all)

# time step of furthest and closest shell in the light cone files
step_min = np.min(steps_all)
step_max = np.max(steps_all)

# get functions relating chi and z
chi_of_z = interp1d(zs_all,chis_all)
z_of_chi = interp1d(chis_all,zs_all)

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

# return the mt catalog names straddling the given redshift
def get_mt_fns(z_th):
    for k in range(len(zs_mt-1)):
        squish = (zs_mt[k] <= z_th) & (z_th <= zs_mt[k+1])
        if squish == True: break 
    z_low = zs_mt[k]
    z_high = zs_mt[k+1]
    chi_low = chis_mt[k]
    chi_high = chis_mt[k+1]
    fn_low = cat_lc_dir+"z%.3f/pid_lc.asdf"%(z_low)
    fn_high = cat_lc_dir+"z%.3f/pid_lc.asdf"%(z_high)
    
    mt_fns = [fn_high, fn_low]
    mt_zs = [z_high, z_low]
    mt_chis = [chi_high, chi_low]
    
    return mt_fns, mt_zs, mt_chis

def extract_steps(fn):
    split_fn = fn.split('Step')[1]
    step = np.int(split_fn.split('.asdf')[0])
    return step

# these are the time steps associated with each of the light cone files
step_fns = np.zeros(len(lc_pid_fns),dtype=int)
for i in range(len(lc_pid_fns)):
    step_fns[i] = extract_steps(lc_pid_fns[i])

# initialize previously loaded mt file name
currently_loaded_zs = []
currently_loaded_headers = []
currently_loaded_pids = []
currently_loaded_tables = []
for step in range(step_start,step_stop+1):
    #TESTINGfor step in range(600,602):
    # this is because our arrays start correspond to step numbers: step_start, step_start+1, step_start+2 ... step_stop
    j = step-step_min
    step_this = steps_all[j]
    z_this = zs_all[j]
    chi_this = chis_all[j]
    
    assert step_this == step, "You've messed up the counts"
    print("light cones step, redshift = ",step_this, z_this)

    # get the two redshifts it's straddling and the mean chi
    mt_fns, mt_zs, mt_chis = get_mt_fns(z_this)

    # get the mean chi
    mt_chi_mean = np.mean(mt_chis)
    
    # how many shells are we including on both sides, including mid point (total of 2j+1)
    buffer_no = 2
    
    # is this the redshift that's closest to the bridge between two redshifts 
    mid_bool = (np.argmin(np.abs(mt_chi_mean-chis_all)) <= j+buffer_no) & (np.argmin(np.abs(mt_chi_mean-chis_all)) >= j-buffer_no)

    # if not in between two redshifts, we just need one catalog -- the one it is closest to
    if not mid_bool:
        mt_fns = [mt_fns[np.argmin(np.abs(mt_chis-chi_this))]] 
        mt_zs = [mt_zs[np.argmin(np.abs(mt_chis-chi_this))]] 

    # load this and prev
    for i in range(len(mt_fns)):
        # check if catalog already loaded
        if mt_zs[i] in currently_loaded_zs: print("skipped loading catalog ",mt_zs[i]); continue
               
        # discard the old redshift catalog and record its data
        if len(currently_loaded_zs) >= 2:

            # save the information about that redshift
            save_asdf(currently_loaded_tables[0],"pid_rv_lc",currently_loaded_headers[0],cat_lc_dir,currently_loaded_zs[0])
            print("saved catalog = ",currently_loaded_zs[0])
            
            # discard it from currently loaded
            currently_loaded_zs = currently_loaded_zs[1:]
            currently_loaded_headers = currently_loaded_headers[1:]
            currently_loaded_pids = currently_loaded_pids[1:]
            currently_loaded_tables = currently_loaded_tables[1:]

        # load new merger tree catalog
        mt_pid, header = load_mt_pid(mt_fns[i],Lbox,PPD)
        
        # start the light cones table for this redshift
        lc_table_final = np.empty(len(mt_pid),dtype=[('pid',mt_pid.dtype),('pos',(np.float32,3)),\
                                                     ('vel',(np.float32,3)),('redshift',np.float32)])
        
        # append the newly loaded catalog
        currently_loaded_zs.append(mt_zs[i])
        currently_loaded_headers.append(header)
        currently_loaded_pids.append(mt_pid)
        currently_loaded_tables.append(lc_table_final)

    print("currently loaded redshifts = ",currently_loaded_zs)    
    print("using redshifts = ",mt_zs)

    # find all light cone file names that correspond to this time step
    choice_fns = np.where(step_fns == step_this)[0]
    # number of light cones at this step
    num_lc = len(choice_fns)
    assert (num_lc <= 3) & (num_lc > 0), "There can be at most three files in the light cones corresponding to a given step"
    
    # loop through those one to three light cone files
    for i_choice, choice_fn in enumerate(choice_fns):
        print("light cones file = ",lc_pid_fns[choice_fn])

        # load particles in light cone
        lc_pid, lc_rv = load_lc_pid_rv(lc_pid_fns[choice_fn],lc_rv_fns[choice_fn],Lbox,PPD)
        
        if 'LightCone1' in lc_pid_fns[choice_fn]:
            offset_lc = np.array([0.,0.,Lbox])
        elif 'LightCone2' in lc_pid_fns[choice_fn]:
            offset_lc = np.array([0.,Lbox,0.])
        else: offset_lc = np.array([0.,0.,0.])

        # loop over the one or two closest catalogs 
        for i in range(len(mt_fns)):
            which_mt = np.where(mt_zs[i] == currently_loaded_zs)[0]
            mt_pid = currently_loaded_pids[which_mt[0]]
            header = currently_loaded_headers[which_mt[0]]
            lc_table_final = currently_loaded_tables[which_mt[0]]
            mt_z = currently_loaded_zs[which_mt[0]]
            
            # actual galaxies in light cone
            pid_mt_lc, comm1, comm2 = np.intersect1d(mt_pid,lc_pid,return_indices=True)
            # select the intersected positions
            pos_mt_lc, vel_mt_lc = unpack_rvint(lc_rv[comm2],Lbox)
            
            # print percentage of matched pids
            print("at z = %.3f, matched = "%mt_z,len(comm1)*100./(len(mt_pid)))
            
            # offset depending on which light cone we are at
            pos_mt_lc += offset_lc

            # save the pid, position, velocity and redshift
            lc_table_final['pid'][comm1] = pid_mt_lc
            lc_table_final['pos'][comm1] = pos_mt_lc
            lc_table_final['vel'][comm1] = vel_mt_lc
            lc_table_final['redshift'][comm1] = np.ones(len(pid_mt_lc))*z_this
            
        print("-------------------")


# close the two that are currently open
for i in range(len(currently_loaded_zs)):
    # save the information about that redshift
    save_asdf(currently_loaded_tables[0],"pid_rv_lc",currently_loaded_headers[0],cat_lc_dir,currently_loaded_zs[0])
    print("saved catalog = ",currently_loaded_zs[0])
            
    # discard it from currently loaded
    currently_loaded_zs = currently_loaded_zs[1:]
    currently_loaded_headers = currently_loaded_headers[1:]
    currently_loaded_pids = currently_loaded_pids[1:]
    currently_loaded_tables = currently_loaded_tables[1:]
quit()
# close the file that was open to write in # what is this tuks
save_asdf(lc_table_final,"pid_rv_lc",header,cat_lc_dir,mt_z)
print("closed redshift = ",mt_z)
del lc_table_final
