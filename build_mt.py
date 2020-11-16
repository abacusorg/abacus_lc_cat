import glob
import asdf
import numpy as np
import sys
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rfn
import time
import gc
import os

from tools.merger import simple_load, get_slab_halo, extract_superslab

# reorder in terms of their slab number
def reorder_by_slab(fns):
    i_sort = np.argsort(extract_superslab(fns))
    tmp = fns.copy()
    for i in range(len(fns)):
        tmp[i] = fns[i_sort[i]]
    return tmp

# read redshifts from merger tree files
def get_zs_from_headers(snap_names):
    zs = np.zeros(len(snap_names))
    for i in range(len(snap_names)):
        snap_name = snap_names[i]
        f = asdf.open(snap_name,lazy_load=True, copy_arrays=True)
        z = np.float(f.tree['header']['Redshift'])
        f.close()
        zs[i] = z
    return zs

# unpack indices in Sownak's format of Nslice*1e12 + superSlabNum*1e9 + halo_position_superSlab
def unpack_inds(halo_ids):
    index = (halo_ids%1e9).astype(int)
    slab_number = ((halo_ids%1e12-index)/1e9).astype(int)
    return slab_number, index

# reorder indices: for given halo index array with corresponding n halos and slabs for its time epoch
def correct_inds(halo_ids, N_halos_slabs, slabs, start=0,stop=None,copies=1):
    # unpack indices
    slab_ids, ids = unpack_inds(halo_ids)

    # total number of halos in the slabs that we have loaded
    N_halos = np.sum(N_halos_slabs[start:stop])
        
    # set up offset array for all files
    offsets_all = np.zeros(len(slabs),dtype=int)
    offsets_all[1:] = np.cumsum(N_halos_slabs)[:-1]
    
    # select the halos belonging to given slab
    #offset = 0
    for i in range(start,stop):
        select = np.where(slab_ids == slabs[i])[0]
        ids[select] += offsets_all[i]
        #offset += N_halos_slabs[i]

    # add additional offset from multiple copies
    if copies == 2:
        #ids[:N_halos] += 0*N_halos
        ids[N_halos:2*N_halos] += 1*N_halos
    elif copies == 3:
        ids[N_halos:2*N_halos] += 1*N_halos
        ids[2*N_halos:3*N_halos] += 2*N_halos
        
    return ids

# load merger tree and progenitors information
def get_mt_info(fns,fields,origin,start=0,stop=None,copies=1):

    # if we are loading all progenitors and not just main
    if 'Progenitors' in fields:
        merger_tree, progs = simple_load(fns[start:stop],fields=fields)
    else:
        merger_tree = simple_load(fns[start:stop],fields=fields)

    # if a far redshift, need 2 copies only
    if copies == 2:
        merger_tree0 = merger_tree
        merger_tree1 = merger_tree.copy()
        merger_tree1['Position'] += np.array([0,0,Lbox])
        merger_tree2 = merger_tree.copy()
        merger_tree2['Position'] += np.array([0,Lbox,0])
        merger_tree = np.hstack((merger_tree1,merger_tree2))
        
    # if in intermediate redshift range, need 3 copies
    elif copies == 3:
        merger_tree0 = merger_tree
        merger_tree1 = merger_tree.copy()
        merger_tree1['Position'] += np.array([0,0,Lbox])
        merger_tree2 = merger_tree.copy()
        merger_tree2['Position'] += np.array([0,Lbox,0])
        merger_tree = np.hstack((merger_tree0,merger_tree1,merger_tree2))
        
    # get number of halos in each slab and number of slabs
    N_halos_slabs, slabs = get_slab_halo(fns)
    
    # load positions in Mpc/h, index of the main progenitors, index of halo
    pos = merger_tree['Position']
    main_prog = merger_tree['MainProgenitor']
    halo_ind = merger_tree['HaloIndex']

    # compute comoving distance between observer and every halo
    com_dist = np.sqrt(np.sum((pos-origin)**2,axis=1))
    
    # if loading all progenitors
    if 'Progenitors' in fields:
        num_progs = merger_tree['NumProgenitors']
        # get an array with the starting indices of the progenitors array
        start_progs = np.zeros(merger_tree.shape,dtype=int)
        start_progs[1:] = num_progs.cumsum()[:-1]
        
        return com_dist, main_prog, halo_ind, pos, start_progs, num_progs, progs, N_halos_slabs, slabs

    return com_dist, main_prog, halo_ind, pos, N_halos_slabs, slabs

# solve when the crossing of the light cones occurs and the interpolated position and velocity
def solve_crossing(r1,r2,pos1,pos2,chi1,chi2):
    # solve for eta_star, where chi = eta_0-eta
    # equation is r1+(chi1-chi)/(chi1-chi2)*(r2-r1) = chi
    # with solution chi_star = (r1(chi1-chi2)+chi1(r2-r1))/((chi1-chi2)+(r2-r1))
    chi_star = (r1*(chi1-chi2)+chi1*(r2-r1))/((chi1-chi2)+(r2-r1))
    
    # get interpolated positions of the halos
    v_avg = (pos2-pos1)/(chi1-chi2)
    pos_star = pos1+v_avg*(chi1-chi_star[:,None])

    # interpolated velocity [km/s]
    vel_star = v_avg*c #vel1+a_avg*(chi1-chi_star)

    # mark True if closer to chi2 (this snapshot) 
    bool_star = np.abs(chi1-chi_star) >  np.abs(chi2-chi_star)

    assert np.sum((chi_star > chi1) | (chi_star < chi2)) == 0, "Solution is out of bounds"
    
    return chi_star, pos_star, vel_star, bool_star

# speed of light
c = 299792.458 # km/s

# simulation name
sim_name = "AbacusSummit_base_c000_ph006"
#sim_name = "AbacusSummit_highbase_c000_ph100"

# initial redshift where we start building the trees and final (incl.)
z_start = 0.45#0.8#0.5
z_stop = 1.25#0.8#0.5

# simulation parameters
if '_highbase_' in sim_name:
    # box size
    Lbox = 1000. # Mpc/h
    # location of the origin in Mpc/h
    origin = np.array([-990.,-990.,-990.])/2.
    # directory where the merger tree is stored
    merger_dir = "/mnt/store2/bigsims/merger/"+sim_name+"/"
elif '_base_' in sim_name:
    # box size
    Lbox = 2000. # Mpc/h
    # location of the origin in Mpc/h
    origin = np.array([-990.,-990.,-990.])
    # directory where the merger tree is stored
    merger_dir = "/mnt/gosling2/bigsims/merger/"+sim_name+"/"
    
# want to show plots
want_plot = False

# directory where we save the final outputs
cat_lc_dir = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"
if not os.path.exists(cat_lc_dir): os.makedirs(cat_lc_dir)

# directory where we save the current state if we want to resume 
if not os.path.exists(os.path.join(cat_lc_dir,'tmp')): os.makedirs(os.path.join(cat_lc_dir,'tmp'))
resume = False
                      
# all redshifts, steps and comoving distances of light cones files; high z to low z
zs_all = np.load("data_headers/redshifts.npy")
chis_all = np.load("data_headers/coord_dist.npy")


# get functions relating chi and z
chi_of_z = interp1d(zs_all,chis_all)
z_of_chi = interp1d(chis_all,zs_all)

# all merger tree snapshots and corresponding redshifts
snaps_mt = sorted(glob.glob(merger_dir+"associations_z*.0.asdf.minified"))
if len(snaps_mt) == 0:
    snaps_mt = sorted(glob.glob(merger_dir+"associations_z*.0.asdf"))
# more accurate, slightly slower
if not os.path.exists("data/zs_mt.npy"):
    zs_mt = get_zs_from_headers(snaps_mt)
    np.save("data/zs_mt.npy",zs_mt)
zs_mt = np.load("data/zs_mt.npy")


# fields are we extracting from the merger trees
#fields_mt = ['HaloIndex','HaloMass','Position','MainProgenitor','Progenitors','NumProgenitors']
# lighter version
fields_mt = ['HaloIndex','Position','MainProgenitor']

# redshift of closest point on wall between original and copied box
z1 = z_of_chi(0.5*Lbox-origin[0])
# redshift of closest point where all three boxes touch
#z2 = z_of_chi((0.5*Lbox-origin[0])*np.sqrt(2))
# furthest point where all three boxes touch; TODO: I think that's what we need
z2 = z_of_chi((0.5*Lbox-origin[0])*np.sqrt(3))

# corresponding indices
ind_start = np.argmin(np.abs(zs_mt-z_start))
ind_stop = np.argmin(np.abs(zs_mt-z_stop))

# initialize difference between the conformal time of last two shells
delta_chi_old = 0.

# loop over each merger tree redshift
for i in range(ind_start,ind_stop+1):

    # this snapshot and previous
    z_this = zs_mt[i]
    z_prev = zs_mt[i+1]
    print("redshift of this and previous snapshot = ",z_this,z_prev)
    
    # up to z1, we work with original box
    if z_this < z1: copies_this = 1
    # after z2, we work with 2 copies of the box
    elif z_this > z2: copies_this = 2
    # between z1 and z2, we work with 3 copies of the box
    else: copies_this = 3

    # up to z1, we work with original box
    if z_prev < z1: copies_prev = 1
    # after z2, we work with 2 copies of the box
    elif z_prev > z2: copies_prev = 2
    # between z1 and z2, we work with 3 copies of the box; could be improved needs testing
    else: copies_prev = 3

    print("copies of the box needed for this and previous snapshot = ",copies_this,copies_prev)
    
    # number of copies should be same and equal to max of the two; repetitive
    copies_this = np.max([copies_this,copies_prev])
    copies_prev = np.max([copies_this,copies_prev])
    
    # previous redshift, distance between shells and copies
    if resume == True:
        z_this_tmp, delta_chi_old, copies_old = np.load(os.path.join(cat_lc_dir,'tmp','z_prev_delta_copies.npy'))
        assert np.abs(z_this-z_this_tmp) < 1.e-6, "Your recorded state is not for the correct redshift, can't resume from old"
        
    # what is the coordinate distance of the light cone at that redshift and the previous 
    assert z_this > np.min(zs_all), "You need to set starting redshift to the smallest value of the merger tree"
    chi_this = chi_of_z(z_this)
    chi_prev = chi_of_z(z_prev)
    delta_chi = chi_prev-chi_this
    print("comoving distance between this and previous snapshot = ",delta_chi)

    # read merger trees file names at this and previous snapshot
    
    fns_this = glob.glob(merger_dir+"associations_z%4.3f.*.asdf.minified"%(z_this))
    fns_prev = glob.glob(merger_dir+"associations_z%4.3f.*.asdf.minified"%(z_prev))
    if len(fns_this) == 0 or len(fns_prev) == 0:
        fns_this = glob.glob(merger_dir+"associations_z%4.3f.*.asdf"%(z_this))
        fns_prev = glob.glob(merger_dir+"associations_z%4.3f.*.asdf"%(z_prev))
    print("number of files = ",len(fns_this),len(fns_prev))

    # number of chunks
    n_chunks = len(fns_this)
    assert n_chunks == len(fns_prev), "Incomplete merger tree files"
    
    # reorder file names by super slab number
    fns_this = reorder_by_slab(fns_this)
    fns_prev = reorder_by_slab(fns_prev)
    
    # starting and finishing superslab chunks; it is best to use all
    start_this = 0
    stop_this = n_chunks
    start_prev = 0
    stop_prev = n_chunks
    
    # get comoving distance and other merger tree data for this snapshot and for the previous one
    if 'Progenitors' in fields_mt:
        com_dist_this, main_prog_this, halo_ind_this, pos_this, start_progs_this, num_progs_this, progs_this, N_halos_slabs_this, slabs_this = get_mt_info(fns_this,fields=fields_mt,origin=origin,start=start_this,stop=stop_this,copies=copies_this)
        com_dist_prev, main_prog_prev, halo_ind_prev, pos_prev, start_progs_prev, num_progs_prev, progs_prev, N_halos_slabs_prev, slabs_prev = get_mt_info(fns_prev,fields=fields_mt,origin=origin,start=start_prev,stop=stop_prev,copies=copies_prev)
    else:
        com_dist_this, main_prog_this, halo_ind_this, pos_this, N_halos_slabs_this, slabs_this = get_mt_info(fns_this,fields=fields_mt,origin=origin,start=start_this,stop=stop_this,copies=copies_this)
        com_dist_prev, main_prog_prev, halo_ind_prev, pos_prev, N_halos_slabs_prev, slabs_prev = get_mt_info(fns_prev,fields=fields_mt,origin=origin,start=start_prev,stop=stop_prev,copies=copies_prev)
    
    # number of halos in this step and previous step; this depends on the number of copies and files requested
    N_halos_this = len(com_dist_this)
    N_halos_prev = len(com_dist_prev)
    print("N_halos_this = ",N_halos_this)
    print("N_halos_prev = ",N_halos_prev)

    # if eligible, can be selected for light cone redshift catalog; 
    if i != ind_start or resume == True:
        # load last state if resuming
        if resume == True:
            eligibility_this = np.load(os.path.join(cat_lc_dir,'tmp','eligibility_prev.npy'))
        # needs more copies if transitioning from 1 to 3 and 3 to 2 intersections
        if copies_old == 1 and copies_this == 3:
            eligibility_this = np.hstack((eligibility_this,eligibility_this,eligibility_this))
        elif copies_old == 3 and copies_this == 2:
            len_copy = int(len(eligibility_this)//3)
            # overlap can only be in copy 1 or copy 3 (draw it)
            eligibility_this = np.hstack((eligibility_this[:len_copy],eligibility_this[-len_copy:]))
    # all start as eligible
    else:
        eligibility_this = np.ones(N_halos_this,dtype=bool)

    
    # for a newly opened redshift, everyone is eligible to be part of the light cone catalog
    eligibility_prev = np.ones(N_halos_prev,dtype=bool)

    # mask where no merger tree info is available or halos that are not eligible (because we don'to need to solve for eta star for those)
    mask_noinfo_this = (main_prog_this <= 0) | (~eligibility_this)
    mask_info_this = (~mask_noinfo_this) & (eligibility_this) # TESTING not sure used to not have anything and above was or neg

    # print percentage where no information is available or halo not eligible
    print("percentage no info or ineligible = ",np.sum(mask_noinfo_this)/len(mask_noinfo_this)*100.)

    # no info is denoted by 0 or -999 (or regular if ineligible), but -999 messes with unpacking, so we set it to 0
    main_prog_this[mask_noinfo_this] = 0

    # rework the main progenitor and halo indices to retun in proper order
    main_prog_this = correct_inds(main_prog_this, N_halos_slabs_prev, slabs_prev, start=start_prev, stop=stop_prev, copies=copies_prev)
    halo_ind_this = correct_inds(halo_ind_this, N_halos_slabs_this, slabs_this, start=start_this, stop=stop_this, copies=copies_this)
    halo_ind_prev = correct_inds(halo_ind_prev, N_halos_slabs_prev, slabs_prev, start=start_prev, stop=stop_prev, copies=copies_prev)

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
    if 'Progenitors' in fields_mt:
        start_progs_this_info = start_progs_this[mask_info_this]
        num_progs_this_info = num_progs_this[mask_info_this]

    # halos that don't have merger tree information
    pos_this_noinfo = pos_this[mask_noinfo_this]
    com_dist_this_noinfo = com_dist_this[mask_noinfo_this]
    halo_ind_this_noinfo = halo_ind_this[mask_noinfo_this]
    eligibility_this_noinfo = eligibility_this[mask_noinfo_this]
    #pos_prev_main_this_noinfo = pos_prev_main_this[mask_noinfo_this]
    #com_dist_prev_main_this_noinfo = com_dist_prev_main_this[mask_noinfo_this]
    
    # select objects that are crossing the light cones
    mask_lc_this_info = (((com_dist_this_info > chi_this) & (com_dist_prev_main_this_info <= chi_prev)) | \
                         ((com_dist_this_info > chi_prev) & (com_dist_prev_main_this_info <= chi_this))) & (eligibility_this_info)
    mask_lc_this_noinfo = ((com_dist_this_noinfo >= chi_this - delta_chi_old/2.) & \
                           (com_dist_this_noinfo < chi_this + delta_chi/2.)) & (eligibility_this_noinfo)

    # percentage of objects that are part of this or previous snapshot
    print("percentage of halos in light cone with and without progenitor info = ", \
          np.sum(mask_lc_this_info)/len(mask_lc_this_info)*100., np.sum(mask_lc_this_noinfo)/len(mask_lc_this_noinfo)*100.)

    # select halos with mt info that have had a light cone crossing
    pos_this_info_lc = pos_this_info[mask_lc_this_info]
    com_dist_this_info_lc = com_dist_this_info[mask_lc_this_info]
    pos_prev_main_this_info_lc = pos_prev_main_this_info[mask_lc_this_info]
    com_dist_prev_main_this_info_lc = com_dist_prev_main_this_info[mask_lc_this_info]
    halo_ind_prev_main_this_info_lc = halo_ind_prev_main_this_info[mask_lc_this_info]
    halo_ind_this_info_lc = halo_ind_this_info[mask_lc_this_info]
    eligibility_this_info_lc = eligibility_this_info[mask_lc_this_info]
    if 'Progenitors' in fields_mt:
        start_progs_this_info_lc = start_progs_this_info[mask_lc_this_info]
        num_progs_this_info_lc = num_progs_this_info[mask_lc_this_info]

    # select halos without mt info that have had a light cone crossing
    pos_this_noinfo_lc = pos_this_noinfo[mask_lc_this_noinfo]
    halo_ind_this_noinfo_lc = halo_ind_this_noinfo[mask_lc_this_noinfo]
    #com_dist_this_noinfo_lc = com_dist_this_noinfo[mask_lc_this_noinfo]
    #pos_prev_main_this_noinfo_lc = pos_prev_main_this_noinfo[mask_lc_this_noinfo]
    #com_dist_prev_main_this_noinfo_lc = com_dist_prev_main_this_noinfo[mask_lc_this_noinfo]
    #eligibility_this_noinfo_lc = eligibility_this_noinfo[mask_lc_this_noinfo]
    
    # save the position and (dummy) velocity of the halos in the light cone without progenitor information
    pos_star_this_noinfo_lc = pos_this_noinfo_lc
    vel_star_this_noinfo_lc = pos_this_noinfo_lc*0.
    
    # get chi star where lc crosses halo trajectory; bool is False where closer to previous
    chi_star_this_info_lc, pos_star_this_info_lc, vel_star_this_info_lc, bool_star_this_info_lc = solve_crossing(com_dist_prev_main_this_info_lc,com_dist_this_info_lc,pos_prev_main_this_info_lc,pos_this_info_lc,chi_prev,chi_this)
    
    # add ineligible halos if any from last iteration of the loop to those crossed in previous
    # marked ineligible (False) only if same halo has already been assigned to a light cone
    bool_elig_star_this_info_lc = (bool_star_this_info_lc) & (eligibility_this_info_lc)

    # number of objects in light cone
    N_this_star_lc = np.sum(bool_elig_star_this_info_lc)
    N_this_noinfo_lc = np.sum(mask_lc_this_noinfo)
    N_lc = N_this_star_lc+N_this_noinfo_lc

    print("in this snapshot: interpolated, no info, total = ", N_this_star_lc*100./N_lc, N_this_noinfo_lc*100./N_lc, N_lc)

    # start new arrays for final output (assuming it is in this snapshot and not in previous)
    pos_interp_lc = np.zeros((N_lc,3))
    vel_interp_lc = np.zeros((N_lc,3))
    halo_ind_lc = np.zeros(N_lc,dtype=int)

    # record interpolated position and velocity
    pos_interp_lc[:N_this_star_lc] = pos_star_this_info_lc[bool_elig_star_this_info_lc]
    vel_interp_lc[:N_this_star_lc] = vel_star_this_info_lc[bool_elig_star_this_info_lc]
    halo_ind_lc[:N_this_star_lc] = halo_ind_this_info_lc[bool_elig_star_this_info_lc]
    pos_interp_lc[-N_this_noinfo_lc:] = pos_star_this_noinfo_lc
    vel_interp_lc[-N_this_noinfo_lc:] = vel_star_this_noinfo_lc
    halo_ind_lc[-N_this_noinfo_lc:] = halo_ind_this_noinfo_lc

    # create directory for this redshift
    if not os.path.exists(os.path.join(cat_lc_dir,"z%.3f"%z_this)):
        os.makedirs(os.path.join(cat_lc_dir,"z%.3f"%z_this))
    
    # adding contributions from the previous
    if i != ind_start or resume == True:
        if resume == True:
            halo_ind_next = np.load(os.path.join(cat_lc_dir,'tmp','halo_ind_next.npy'))
            pos_star_next = np.load(os.path.join(cat_lc_dir,'tmp','pos_star_next.npy'))
            vel_star_next = np.load(os.path.join(cat_lc_dir,'tmp','vel_star_next.npy'))
            resume = False
        N_lc += len(halo_ind_next) # todo improve
        pos_interp_lc = np.vstack((pos_interp_lc,pos_star_next))
        vel_interp_lc = np.vstack((vel_interp_lc,vel_star_next))
        halo_ind_lc = np.hstack((halo_ind_lc,halo_ind_next))
        
    # save those arrays
    table_lc = np.empty(N_lc,dtype=[('halo_ind',halo_ind_lc.dtype),('pos_interp',(pos_interp_lc.dtype,3)),('vel_interp',(vel_interp_lc.dtype,3))])
    table_lc['halo_ind'] = halo_ind_lc
    table_lc['pos_interp'] = pos_interp_lc
    table_lc['vel_interp'] = vel_interp_lc
    np.save(os.path.join(cat_lc_dir,"z%.3f"%z_this,'table_lc.npy'),table_lc)
    
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
    if 'Progenitors' in fields_mt:
        for j in range(len(start_progs_this_info_lc[~bool_star_this_info_lc])):
            start = (start_progs_this_info_lc[~bool_star_this_info_lc])[j]
            num = (num_progs_this_info_lc[~bool_star_this_info_lc])[j]
            prog_inds = progs_this[start:start+num]
            prog_inds = correct_inds(prog_inds,N_halos_slabs_prev,slabs_prev)
            halo_inds = halo_ind_prev[prog_inds]
            #if j < 100: print(halo_inds, halo_ind_next[j])
            eligibility_prev[halo_inds] = False

    # information to keep for next redshift considered; should have dimensions equal to sum elig prev
    vel_star_next = vel_star_this_info_lc[~bool_star_this_info_lc]
    pos_star_next = pos_star_this_info_lc[~bool_star_this_info_lc]

    if want_plot:
        # select the halos in the light cones
        try:
            pos_choice = pos_this[halo_ind_lc]
        except:
            # not too sure
            pos_choice = pos_this[halo_ind_lc%len_copy]
        
        # selecting thin slab
        pos_x_min = -490.
        pos_x_max = -480.

        ijk = 0
        choice = (pos_choice[:,ijk] >= pos_x_min) & (pos_choice[:,ijk] < pos_x_max)
        #choice_lc = (pos_this_info[:,ijk] >= pos_x_min) & (pos_this_info[:,ijk] < pos_x_max)

        circle_this = plt.Circle((origin[1], origin[2]), radius=chi_this, color='g', fill=False)
        circle_prev = plt.Circle((origin[1], origin[2]), radius=chi_prev, color='r', fill=False)

        ax = plt.gca()
        ax.cla() # clear things for fresh plot

        #ax.scatter(pos_this_info[choice_lc,1],pos_this_info[choice_lc,2],s=0.01,alpha=0.5,color='orange')
        ax.scatter(pos_choice[choice,1],pos_choice[choice,2],s=0.01,alpha=0.5,color='dodgerblue')

        # circles for in and prev
        ax.add_artist(circle_this)
        ax.add_artist(circle_prev)
        plt.xlabel([-1000,3000])
        plt.ylabel([-1000,3000])
        plt.axis('equal')
        plt.show()
    
    del com_dist_this, main_prog_this, halo_ind_this, pos_this, N_halos_slabs_this, slabs_this
    del com_dist_prev, main_prog_prev, halo_ind_prev, pos_prev, N_halos_slabs_prev, slabs_prev
    gc.collect()
    
    # update values for difference in comoving distance, eligibility and number of copies
    delta_chi_old = delta_chi
    eligibility_this = eligibility_prev
    copies_old = copies_prev

    # save the current state so you can resume
    np.save(os.path.join(cat_lc_dir,'tmp','eligibility_prev.npy'),eligibility_prev)
    np.save(os.path.join(cat_lc_dir,'tmp','halo_ind_next.npy'),halo_ind_next)
    np.save(os.path.join(cat_lc_dir,'tmp','pos_star_next.npy'),pos_star_next)
    np.save(os.path.join(cat_lc_dir,'tmp','vel_star_next.npy'),vel_star_next)
    np.save(os.path.join(cat_lc_dir,'tmp','z_prev_delta_copies.npy'),np.array([z_prev, delta_chi, copies_prev]))

#dict_keys(['HaloIndex', 'HaloMass', 'HaloVmax', 'IsAssociated', 'IsPotentialSplit', 'MainProgenitor', 'MainProgenitorFrac', 'MainProgenitorPrec', 'MainProgenitorPrecFrac', 'NumProgenitors', 'Position', 'Progenitors'])
