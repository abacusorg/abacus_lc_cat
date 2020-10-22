import glob
import asdf
import numpy as np
import sys
from util import simple_load, get_slab_halo, extract_superslab
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rfn
import time
import gc
import os

from compaso_halo_catalog import CompaSOHaloCatalog


# save to files
load_asdf(halo_table,"halo_info_lc",header,cat_lc_dir,z_in)
load_asdf(pid_table,"pid_lc",header,cat_lc_dir,z_in)

def load_asdf(table,filename,header,cat_lc_dir,z_in):
    # load the halo catalog/particle subsamples
    table = 

    
# cosmological parameters
h = 0.6736
H0 = h*100.# km/s/Mpc
Om_m = 0.315192
c = 299792.458# km/s

# simulation parameters
Lbox = 2000. # Mpc/h
PPD = 6912
NP = PPD**3

# location of the origin in Mpc/h
origin = np.array([-990.,-990.,-990.])

# simulation name
sim_name = "AbacusSummit_base_c000_ph006"
#sim_name = "AbacusSummit_highbase_c000_ph100"

# directory where we save the final outputs
cat_lc_dir = "/mnt/store/boryanah/"+sim_name+"/halos_light_cones/"

# all redshifts, steps and comoving distances of light cones files; high z to low z
zs_all = np.load("/home/boryanah/HOD_Abacus/light_cones/data_headers/redshifts.npy")
steps_all = np.load("/home/boryanah/HOD_Abacus/light_cones/data_headers/steps.npy")
chis_all = np.load("/home/boryanah/HOD_Abacus/light_cones/data_headers/coord_dist.npy")

# time step of furthest and closest shell
step_min = np.min(steps_all)
step_max = np.max(steps_all)

# get functions relating chi and z
chi_of_z = interp1d(zs_all,chis_all)
z_of_chi = interp1d(chis_all,zs_all)

# all merger tree snapshots and corresponding redshifts
snaps_mt = sorted(glob.glob(merger_dir+"associations_z*.0.asdf"))
#zs_mt = get_zs_from_snap_names(snaps_mt)
# more accurate, slightly slower
if not os.path.exists("data/zs_mt.npy"):
    zs_mt = get_zs_from_headers(snaps_mt)
    np.save("data/zs_mt.npy",zs_mt)
zs_mt = np.load("data/zs_mt.npy")

# fields are we extracting from the merger trees
fields_mt = ['HaloIndex','HaloMass','Position','MainProgenitor','Progenitors','NumProgenitors']

# fields are we extracting from the catalogs
fields_cat = ['id','npstartA','npoutA','N','x_L2com','v_L2com']

# redshift of closest point on wall between original and copied box
z1 = z_of_chi(0.5*Lbox-origin[0])
# redshift of closest point where all three boxes touch
z2 = z_of_chi((0.5*Lbox-origin[0])*np.sqrt(2))

# initial redshift where we start building the trees
z_start = 0.5#np.min(zs_mt)
ind_start = np.argmin(np.abs(zs_mt-z_start))

# load full halo catalog for all redshifts

# load HOD galaxies and potentially which light cones they belong to
pids_final = np.load("data_hod/pids_final.npy")
steps_final = np.load("data_hod/steps_final.npy")
print("total number of galaxies = ",len(pids_final))


# which lightcone
which_lc = '1'#'0'#'1'#'2'

# Key info
data_key = 'data'
BoxSize = 2000.
PPD = 6912
lc_dir = '/mnt/store/AbacusSummit/AbacusSummit_base_c000_ph006/lightcones'

# depending on the box you're loading where is the origin
if which_lc == '0': origin = np.array([-990.,-990.,-990.])
if which_lc == '1': origin = np.array([-990.,-990.,-2990.])
if which_lc == '2': origin = np.array([-990.,-2990.,-990.])


# Read file names
lc_rv_fns = sorted(glob(pjoin(lc_dir, 'rv/LightCone'+which_lc+'*')))
lc_pid_fns = sorted(glob(pjoin(lc_dir, 'pid/LightCone'+which_lc+'*')))

def extract_steps(fn):
    split_fn = fn.split('Step')[1]
    step = np.int(split_fn.split('.asdf')[0])
    return step

for i in range(len(lc_rv_fns)):
    print("file number out of = ",i,len(lc_rv_fns))
    
    step_lc = extract_steps(lc_rv_fns[i])
    print("step, redshift = ", step_lc, zs_all[np.argmin(np.abs(step_lc-steps_all))])

    # galaxies potential here (-1 because small step means, higher redshift and chi
    inds_hod = (steps_final == step_lc) | (steps_final-1 == step_lc)
    pids_hod = pids_final[inds_hod]

    # particles in light cone
    lc_pids = asdf.open(lc_pid_fns[i], lazy_load=True, copy_arrays=True)
    pid, lagr_pos, tagged, density = unpack_pids(lc_pids[data_key]['packedpid'][:],BoxSize,PPD)

    # actual galaxies in light cone
    pid_lc_hod, comm1, comm2 = np.intersect1d(pids_hod,pid,return_indices=True)

    # load their positions and velocities
    lc_rvs = asdf.open(lc_rv_fns[i], lazy_load=True, copy_arrays=True)
    xyz_hod, vel_hod = unpack_rvint(lc_rvs[data_key]['rvint'][comm2],BoxSize)

    # save positions and velocities
    try:
        xyz_final_hod = np.vstack((xyz_final_hod,xyz_hod))
        vel_final_hod = np.vstack((vel_final_hod,vel_hod))
    except:
        xyz_final_hod = xyz_hod
        vel_final_hod = vel_hod

print("number of galaxies we populated to light cones = ",xyz_final_hod.shape)
np.save("data_lc/xyz_final_hod.npy",xyz_final_hod)
np.save("data_lc/vel_final_hod.npy",vel_final_hod)

quit()
for i in range(ind_start,len(zs_mt)-1):

    # starting snapshot
    z_in = zs_mt[i]
    z_prev = zs_mt[i+1]
    print("redshift now and previous = ",z_in,z_prev)


    # what is the coordinate distance of the light cone at that redshift and the previous 
    if z_in < np.min(zs_all): chi_in = chi_of_z(np.min(zs_all)); z_in = 0.1 # to avoid getting out of the interpolation range
    else: chi_in = chi_of_z(z_in)
    chi_prev = chi_of_z(z_prev)
    delta_chi = chi_prev-chi_in
    delta_chi_old = 0.
    print("comoving distance now and previous = ",chi_in,chi_prev)

    # for this given redshift, let's sort the particles into smaller groups based on distance
    # have to compute distances for all halos and isolate those particles centered around some mean redshift
    # also load 3 lc shells and match your particles

    # load merger trees at this snapshot
    fns_in = sorted(glob.glob(merger_dir+"associations_z%4.3f.*.asdf"%(z_in)))
    fns_prev = sorted(glob.glob(merger_dir+"associations_z%4.3f.*.asdf"%(z_prev)))
    print("Number of files = ",len(fns_in),len(fns_prev))

    # starting and finishing superslab number (None is all)
    start_in = 0
    stop_in = 2#None
    start_prev = 0
    stop_prev = 2#None

    # reorder file names by super slab number
    fns_in = reorder_by_slab(fns_in)
    fns_prev = reorder_by_slab(fns_prev)
    
    # get comoving distance and other merger tree data for this snapshot and for the previous one
    com_dist_in, main_prog_in, halo_ind_in, pos_in, start_progs_in, num_progs_in, progs_in, N_halos_slabs_in, slabs_in = get_mt_info(fns_in,fields=fields_mt,origin=origin,start=start_in,stop=stop_in,copies=copies_in)
    com_dist_prev, main_prog_prev, halo_ind_prev, pos_prev, start_progs_prev, num_progs_prev, progs_prev, N_halos_slabs_prev, slabs_prev = get_mt_info(fns_prev,fields=fields_mt,origin=origin,start=start_prev,stop=stop_prev,copies=copies_prev)

    # number of halos in this step and previous step
    N_halos_in = len(com_dist_in)
    N_halos_prev = len(com_dist_prev)
    print("N_halos_in = ",N_halos_in)
    print("N_halos_prev = ",N_halos_prev)

    # if eligible, can be selected for light cone redshift catalog
    if i == ind_start:
        eligibility_in = np.ones(N_halos_in,dtype=bool)
    eligibility_prev = np.ones(N_halos_prev,dtype=bool)

    # everyone is eligibile in first redshift
    # ~elig is all False
    # thus all noinfo is False
    # we want to generally get rid of those that are not eligible i.e. elig = False
    # ~elig is True

    # mask where no merger tree info is available
    mask_noinfo_in = (main_prog_in <= 0) | (~eligibility_in)
    mask_info_in = ~mask_noinfo_in

    print("masked no info = ",np.sum(mask_noinfo_in)/len(mask_noinfo_in)*100.)

    # no info is denoted by 0 or -999, but -999 messes with unpacking, so we set it to 0
    main_prog_in[mask_noinfo_in] = 0

    # rework the main progenitor and halo indices to retun in proper order
    main_prog_in = correct_inds(main_prog_in, N_halos_slabs_prev,slabs_prev,start=start_prev,stop=stop_prev,copies=copies_prev)
    halo_ind_in = correct_inds(halo_ind_in, N_halos_slabs_in,slabs_in,start=start_in,stop=stop_in,copies=copies_in)
    halo_ind_prev = correct_inds(halo_ind_prev, N_halos_slabs_prev,slabs_prev,start=start_prev,stop=stop_prev,copies=copies_prev)

    # TESTING we only need this when loading incomplete merger trees because we're missing data
    mask_noinfo_in[main_prog_in > N_halos_prev] = False
    main_prog_in[main_prog_in > N_halos_prev] = 0

    # positions and comoving distances of main progenitor halos corresponding to the halos in current snapshot
    pos_prev_main_in = pos_prev[main_prog_in]
    com_dist_prev_main_in = com_dist_prev[main_prog_in]
    halo_ind_prev_main_in = halo_ind_prev[main_prog_in]

    # halos that have merger tree information
    pos_in_info = pos_in[mask_info_in]
    com_dist_in_info = com_dist_in[mask_info_in]
    pos_prev_main_in_info = pos_prev_main_in[mask_info_in]
    com_dist_prev_main_in_info = com_dist_prev_main_in[mask_info_in]
    halo_ind_in_info = halo_ind_in[mask_info_in]
    halo_ind_prev_main_in_info = halo_ind_prev_main_in[mask_info_in]
    eligibility_in_info = eligibility_in[mask_info_in]
    start_progs_in_info = start_progs_in[mask_info_in]
    num_progs_in_info = num_progs_in[mask_info_in]

    # halos that don't have merger tree information
    pos_in_noinfo = pos_in[mask_noinfo_in]
    com_dist_in_noinfo = com_dist_in[mask_noinfo_in]
    pos_prev_main_in_noinfo = pos_prev_main_in[mask_noinfo_in]
    com_dist_prev_main_in_noinfo = com_dist_prev_main_in[mask_noinfo_in]
    halo_ind_in_noinfo = halo_ind_in[mask_noinfo_in]
    eligibility_in_noinfo = eligibility_in[mask_noinfo_in]

    # select objects that are crossing the light cones
    mask_lc_in_info = ((com_dist_in_info > chi_in) & (com_dist_prev_main_in_info <= chi_prev)) | ((com_dist_in_info <= chi_in) & (com_dist_prev_main_in_info > chi_prev))
    mask_lc_in_noinfo = ((com_dist_in_noinfo >= chi_in - delta_chi_old/2.) & (com_dist_in_noinfo < chi_in + delta_chi/2.)) | (~eligibility_in_noinfo)
    #mask_lc_in_noinfo = ((com_dist_in_noinfo >= chi_in) & (com_dist_in_noinfo < chi_prev)) | (~eligibility_in_noinfo)

    print("masked no info crossing = ",np.sum(mask_lc_in_noinfo)/len(mask_lc_in_noinfo)*100.)
    print("masked with info crossing = ",np.sum(mask_lc_in_info)/len(mask_lc_in_info)*100.)

    # select halos with mt info that have had a light cone crossing
    pos_in_info_lc = pos_in_info[mask_lc_in_info]
    com_dist_in_info_lc = com_dist_in_info[mask_lc_in_info]
    pos_prev_main_in_info_lc = pos_prev_main_in_info[mask_lc_in_info]
    com_dist_prev_main_in_info_lc = com_dist_prev_main_in_info[mask_lc_in_info]
    halo_ind_prev_main_in_info_lc = halo_ind_prev_main_in_info[mask_lc_in_info]
    halo_ind_in_info_lc = halo_ind_in_info[mask_lc_in_info]
    eligibility_in_info_lc = eligibility_in_info[mask_lc_in_info]
    start_progs_in_info_lc = start_progs_in_info[mask_lc_in_info]
    num_progs_in_info_lc = num_progs_in_info[mask_lc_in_info]

    # select halos without mt info that have had a light cone crossing
    pos_in_noinfo_lc = pos_in_noinfo[mask_lc_in_noinfo]
    com_dist_in_noinfo_lc = com_dist_in_noinfo[mask_lc_in_noinfo]
    pos_prev_main_in_noinfo_lc = pos_prev_main_in_noinfo[mask_lc_in_noinfo]
    com_dist_prev_main_in_noinfo_lc = com_dist_prev_main_in_noinfo[mask_lc_in_noinfo]
    halo_ind_in_noinfo_lc = halo_ind_in_noinfo[mask_lc_in_noinfo]
    eligibility_in_noinfo_lc = eligibility_in_noinfo[mask_lc_in_noinfo]
    # todo: fix
    pos_star_in_noinfo_lc = pos_in_noinfo_lc
    vel_star_in_noinfo_lc = pos_in_noinfo_lc*0.

    if i != ind_start:
        pos_star_in_noinfo_lc[~eligibility_in_noinfo_lc] = pos_star_next
        vel_star_in_noinfo_lc[~eligibility_in_noinfo_lc] = vel_star_next
        halo_ind_in_noinfo_lc[~eligibility_in_noinfo_lc] = halo_ind_next

    # get chi star where lc crosses halo trajectory
    chi_star_in_info_lc, pos_star_in_info_lc, vel_star_in_info_lc, bool_star_in_info_lc = solve_crossing(com_dist_prev_main_in_info_lc,com_dist_in_info_lc,pos_prev_main_in_info_lc,pos_in_info_lc,chi_prev,chi_in)

    # add ineligible halos from last iteration
    bool_star_in_info_lc = (bool_star_in_info_lc) | (~eligibility_in_info_lc)

    # mark eligibility
    # version 1: only the main progenitor is marked ineligible
    halo_ind_next = halo_ind_prev_main_in_info_lc[~bool_star_in_info_lc]
    eligibility_prev[halo_ind_next] = False
    # version 2: all progenitors are marked ineligible
    # slower, but works (perhaps optimize with numba). Confusing part is why halo_inds has zeros
    '''
    # todo: slight issue is that we mask only for main prog - can perhaps add a clause that takes the main prog value if not an appropriate value is supplied by progs
    for j in range(len(start_progs_in_info_lc[~bool_star_in_info_lc])):
        start = (start_progs_in_info_lc[~bool_star_in_info_lc])[j]
        num = (num_progs_in_info_lc[~bool_star_in_info_lc])[j]
        prog_inds = progs_in[start:start+num]
        prog_inds = correct_inds(prog_inds,N_halos_slabs_prev,slabs_prev)
        halo_inds = halo_ind_prev[prog_inds]
        #if j < 100: print(halo_inds, halo_ind_next[j])
        eligibility_prev[halo_inds] = False
    '''

    # information to keep for next redshift considered
    vel_star_next = vel_star_in_info_lc[~bool_star_in_info_lc]
    pos_star_next = pos_star_in_info_lc[~bool_star_in_info_lc]

    # number of objects in light cone
    N_in_star_lc = np.sum(bool_star_in_info_lc)
    N_in_noinfo_lc = np.sum(mask_lc_in_noinfo)
    N_lc = N_in_star_lc+N_in_noinfo_lc

    print("with info and closer, noinfo, total = ",N_in_star_lc,N_in_noinfo_lc,N_lc,N_halos_in)

    # start new arrays for final output (assuming it is in and not prev)
    pos_interp_lc = np.zeros((N_lc,3))
    vel_interp_lc = np.zeros((N_lc,3))
    halo_ind_lc = np.zeros(N_lc,dtype=int)

    # record interpolated position and velocity
    pos_interp_lc[:N_in_star_lc] = pos_star_in_info_lc[bool_star_in_info_lc]
    vel_interp_lc[:N_in_star_lc] = vel_star_in_info_lc[bool_star_in_info_lc]
    halo_ind_lc[:N_in_star_lc] = halo_ind_in_info_lc[bool_star_in_info_lc]
    pos_interp_lc[N_in_star_lc:N_lc] = pos_star_in_noinfo_lc
    vel_interp_lc[N_in_star_lc:N_lc] = vel_star_in_noinfo_lc
    halo_ind_lc[N_in_star_lc:N_lc] = halo_ind_in_noinfo_lc




    # catalog directory
    catdir = os.path.join(cat_dir,"z%.3f"%z_in)


    # load halo catalog, setting unpack to False for speed
    cat = CompaSOHaloCatalog(catdir, load_subsamples='A_halo_pid', fields=fields_cat, unpack_bits = False)


    # halo catalog
    halo_table = cat.halos[halo_ind_lc]
    header = cat.header
    N_halos = len(cat.halos)
    print("N_halos = ",N_halos)

    # load the pid, set unpack_bits to True
    pid = cat.subsamples['pid']
    npstart = halo_table['npstartA']
    npout = halo_table['npoutA']
    npstart_new = np.zeros(len(npout),dtype=int)
    npstart_new[1:] = np.cumsum(npout)[:-1]
    npout_new = npout
    pid_new = np.zeros(np.sum(npout_new),dtype=pid.dtype)
    for j in range(len(npstart)):
        pid_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = pid[npstart[j]:npstart[j]+npout[j]]
    pid_table = np.empty(len(pid_new),dtype=[('pid',pid_new.dtype)])
    pid_table['pid'] = pid_new
    halo_table['npstartA'] = npstart_new
    halo_table['npoutA'] = npout_new

    # append new fields
    halo_table['index_halo'] = halo_ind_lc
    halo_table['x_interp'] = pos_interp_lc
    halo_table['v_interp'] = vel_interp_lc

    # create directory for this redshift
    if not os.path.exists(os.path.join(cat_lc_dir,"z%.3f"%z_in)):
        os.makedirs(os.path.join(cat_lc_dir,"z%.3f"%z_in))

    

    # update eligibility array
    eligibility_in = eligibility_prev

    # TESTING
    break

    # delete things at the end
    del com_dist_in, main_prog_in, halo_ind_in, pos_in, N_halos_slabs_in, slabs_in
    del com_dist_prev, main_prog_prev, halo_ind_prev, pos_prev, N_halos_slabs_prev, slabs_prev
    del pid, pid_new, pid_table, npstart, npout, npstart_new, npout_new
    del halo_table
    del cat

    gc.collect()

    delta_chi_old = delta_chi


# EVERYTHING DOWN HERE IS FOR PLOTTING

#pos_choice = pos_in_info_lc
pos_choice = pos_in[halo_ind_lc]

# selecting thin slab
pos_z_min = -995.
pos_z_max = -985.
ijk = 0
choice = (pos_choice[:,ijk] >= pos_z_min) & (pos_choice[:,ijk] < pos_z_max)
choice_lc = (pos_in_info[:,ijk] >= pos_z_min) & (pos_in_info[:,ijk] < pos_z_max)


circle_in = plt.Circle((origin[1], origin[2]), radius=chi_in, color='g', fill=False)
circle_prev = plt.Circle((origin[1], origin[2]), radius=chi_prev, color='r', fill=False)

ax = plt.gca()
ax.cla() # clear things for fresh plot

ax.scatter(pos_in_info[choice_lc,1],pos_in_info[choice_lc,2],s=0.01,alpha=0.5,color='orange')
ax.scatter(pos_choice[choice,1],pos_choice[choice,2],s=0.01,alpha=0.5,color='dodgerblue')
# circles for in and prev
ax.add_artist(circle_in)
ax.add_artist(circle_prev)
plt.xlabel([-1000,3000])
plt.ylabel([-1000,3000])
plt.axis('equal')
plt.show()

#dict_keys(['HaloIndex', 'HaloMass', 'HaloVmax', 'IsAssociated', 'IsPotentialSplit', 'MainProgenitor', 'MainProgenitorFrac', 'MainProgenitorPrec', 'MainProgenitorPrecFrac', 'NumProgenitors', 'Position', 'Progenitors'])
