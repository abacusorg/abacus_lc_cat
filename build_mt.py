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

# unpack indices in Sownak's format of Nslice*1e12 + superSlabNum*1e9 + halo_position_in_superSlab
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
        merger_tree1['Position'] += np.array([0,0,2000.])
        merger_tree2 = merger_tree.copy()
        merger_tree2['Position'] += np.array([0,2000.,0])
        merger_tree = np.hstack((merger_tree1,merger_tree2))
        
    # if in intermediate redshift range, need 3 copies
    elif copies == 3:
        merger_tree0 = merger_tree
        merger_tree1 = merger_tree.copy()
        merger_tree1['Position'] += np.array([0,0,2000.])
        merger_tree2 = merger_tree.copy()
        merger_tree2['Position'] += np.array([0,2000.,0])
        merger_tree = np.hstack((merger_tree0,merger_tree1,merger_tree2))
        
    # get the slab number and number of halos ordered
    N_halos_slabs, slabs = get_slab_halo(fns)
    
    # load positions in Mpc/h, index of the main progenitors, index of halo
    pos = merger_tree['Position']
    main_prog = merger_tree['MainProgenitor']
    halo_ind = merger_tree['HaloIndex']

    # compute comoving distance to observer of every halo
    com_dist = np.sqrt(np.sum((pos-origin)**2,axis=1))
    
    # if loading all progenitors
    if 'Progenitors' in fields:
        num_progs = merger_tree['NumProgenitors']
        # get an array with the starting indices of the progenitors array
        start_progs = np.zeros(merger_tree.shape,dtype=int)
        start_progs[1:] = num_progs.cumsum()[:-1]
        
        return com_dist, main_prog, halo_ind, pos, start_progs, num_progs, progs, N_halos_slabs, slabs

    return com_dist, main_prog, halo_ind, pos, start_progs, N_halos_slabs, slabs

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

    # mark eligibility based on chi_star; True is closer to chi2 (current redshift); 
    mask = (np.abs(chi1-chi_star) >  np.abs(chi2-chi_star))

    assert np.sum((chi_star > chi1) | (chi_star < chi2)) == 0, "Solution is out of bounds"
    
    return chi_star, pos_star, vel_star, mask

# TODO: copy halo info (just get rid of fields=fields_cat);  velocity interpolation (could be done when velocities are summoned, maybe don't interpolate); delete things properly; read parameters from header;

# cosmological parameters
h = 0.6736
H0 = h*100.# km/s/Mpc
Om_m = 0.315192
c = 299792.458# km/s

# simulation parameters
Lbox = 2000. # Mpc/h
PPD = 6912
NP = PPD**3

# want to show plots
want_plot = False

# location of the origin in Mpc/h
origin = np.array([-990.,-990.,-990.])

# simulation name
sim_name = "AbacusSummit_base_c000_ph006"
#sim_name = "AbacusSummit_highbase_c000_ph100"

# directory where the halo catalogs are saved
#cat_dir = "/mnt/store/lgarrison/"+sim_name+"/halos/"
cat_dir = "/mnt/store2/bigsims/"+sim_name+"/halos/"

# directory where the merger tree is stored
#merger_dir = "/mnt/store/AbacusSummit/merger/"+sim_name+"/"
merger_dir = "/mnt/store2/bigsims/merger/"+sim_name+"/"

# directory where we save the final outputs
cat_lc_dir = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"
if not os.path.exists(cat_lc_dir): os.makedirs(cat_lc_dir)

# all redshifts, steps and comoving distances of light cones files; high z to low z
zs_all = np.load("data_headers/redshifts.npy")
steps_all = np.load("data_headers/steps.npy")
chis_all = np.load("data_headers/coord_dist.npy")

# time step of furthest and closest shell
step_min = np.min(steps_all)
step_max = np.max(steps_all)

# get functions relating chi and z
chi_of_z = interp1d(zs_all,chis_all)
z_of_chi = interp1d(chis_all,zs_all)

# all merger tree snapshots and corresponding redshifts
snaps_mt = sorted(glob.glob(merger_dir+"associations_z*.0.asdf"))
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
#z2 = z_of_chi((0.5*Lbox-origin[0])*np.sqrt(2))
# furthest point where all three boxes touch; TODO: I think that's what we need
z2 = z_of_chi((0.5*Lbox-origin[0])*np.sqrt(3))

# initial redshift where we start building the trees
z_start = 0.4#0.8#0.5
z_stop = 0.5#0.8#0.5
#z_start = np.min(zs_mt)
ind_start = np.argmin(np.abs(zs_mt-z_start))
ind_stop = np.argmin(np.abs(zs_mt-z_stop))#len(zs_mt)-1
# initialize difference between the conformal time of last two shells
delta_chi_old = 0.

# loop over each merger tree redshift
for i in range(ind_start,ind_stop+1):

    # starting snapshot
    z_in = zs_mt[i]
    z_prev = zs_mt[i+1]
    print("redshift now and previous = ",z_in,z_prev)
    
    # up to z1, we work with original box
    # between z1 and z2, we work with 3 copies of the box
    # after z2, we work with 2 copies of the box
    if z_in < z1: copies_in = 1
    elif z_in > z2: copies_in = 2
    else: copies_in = 3
    if z_prev < z1: copies_prev = 1
    elif z_prev > z2: copies_prev = 2
    else: copies_prev = 3
    # set to max for both; still not work
    print("Copies of the box needed = ",copies_in,copies_prev)
    copies_in = np.max([copies_in,copies_prev])
    copies_prev = np.max([copies_in,copies_prev])
    
    # what is the coordinate distance of the light cone at that redshift and the previous 
    if z_in < np.min(zs_all): chi_in = chi_of_z(np.min(zs_all)); z_in = 0.1 # to avoid getting out of the interpolation range
    else: chi_in = chi_of_z(z_in)
    chi_prev = chi_of_z(z_prev)
    delta_chi = chi_prev-chi_in
    print("comoving distance now and previous = ",chi_in,chi_prev)

    # read merger trees file names at this and previous snapshot
    fns_in = sorted(glob.glob(merger_dir+"associations_z%4.3f.*.asdf"%(z_in)))
    fns_prev = sorted(glob.glob(merger_dir+"associations_z%4.3f.*.asdf"%(z_prev)))
    print("Number of files = ",len(fns_in),len(fns_prev))

    # number of chunks
    n_chunks = len(fns_in)
    assert n_chunks == len(fns_prev), "Incomplete merger tree files"
    
    # starting and finishing superslab number (None is all)
    #i_chunk = 0
    #start_in = i_chunk
    #stop_in = i_chunk+1
    #start_prev = (start_in-1)%n_chunks
    #stop_prev = (stop_in+1)%n_chunks
    start_in = 0
    stop_in = n_chunks
    start_prev = 0
    stop_prev = n_chunks
    
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

    # if eligible, can be selected for light cone redshift catalog; all start as eligible
    if i == ind_start:
        eligibility_in = np.ones(N_halos_in,dtype=bool)
    else:
        # transitioning from 1 to 3 and 3 to 2
        if copies_old == 1 and copies_in == 3:
            eligibility_in = np.hstack((eligibility_in,eligibility_in,eligibility_in))
        elif copies_old == 3 and copies_in == 2:
            len_copy = int(len(eligibility_in)//3)
            # overlap can only be in copy 1 or copy 3 (draw it)
            eligibility_in = np.hstack((eligibility_in[:len_copy],eligibility_in[-len_copy:]))
            
    eligibility_prev = np.ones(N_halos_prev,dtype=bool)

    # everyone is eligibile in first redshift
    # ~elig is all False
    # thus all noinfo is False
    # we want to generally get rid of those that are not eligible i.e. elig = False
    # ~elig is True

    # mask where no merger tree info is available or halos that are not eligible
    mask_noinfo_in = (main_prog_in <= 0) | (~eligibility_in)
    mask_info_in = ~mask_noinfo_in

    # print percentage where no information is available or halo not eligible
    print("percentage no info or ineligible = ",np.sum(mask_noinfo_in)/len(mask_noinfo_in)*100.)

    # no info is denoted by 0 or -999 (or regular if ineligible), but -999 messes with unpacking, so we set it to 0
    main_prog_in[mask_noinfo_in] = 0

    # rework the main progenitor and halo indices to retun in proper order
    main_prog_in = correct_inds(main_prog_in, N_halos_slabs_prev,slabs_prev,start=start_prev,stop=stop_prev,copies=copies_prev)
    halo_ind_in = correct_inds(halo_ind_in, N_halos_slabs_in,slabs_in,start=start_in,stop=stop_in,copies=copies_in)
    halo_ind_prev = correct_inds(halo_ind_prev, N_halos_slabs_prev,slabs_prev,start=start_prev,stop=stop_prev,copies=copies_prev)

    # TESTING REMOVE we only need this when loading incomplete merger trees because we're missing data
    #mask_noinfo_in[main_prog_in > N_halos_prev] = False
    #main_prog_in[main_prog_in > N_halos_prev] = 0

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
    #mask_lc_in_info = ((com_dist_in_info > chi_in) & (com_dist_prev_main_in_info <= chi_prev)) | ((com_dist_in_info <= chi_in) & (com_dist_prev_main_in_info > chi_prev))
    mask_lc_in_info = ((com_dist_in_info > chi_in) & (com_dist_prev_main_in_info <= chi_prev)) | ((com_dist_in_info <= chi_in) & (com_dist_prev_main_in_info > chi_prev)) & (eligibility_in_info)# TESTING
    #mask_lc_in_noinfo = ((com_dist_in_noinfo >= chi_in - delta_chi_old/2.) & (com_dist_in_noinfo < chi_in + delta_chi/2.)) | (~eligibility_in_noinfo)
    mask_lc_in_noinfo = ((com_dist_in_noinfo >= chi_in - delta_chi_old/2.) & (com_dist_in_noinfo < chi_in + delta_chi/2.)) & (eligibility_in_noinfo)# TESTING # one for closer to prev and one for already has passed

    # percentage of objects that are part of this or previous snapshot
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
    # todo: fix velocity
    pos_star_in_noinfo_lc = pos_in_noinfo_lc
    vel_star_in_noinfo_lc = pos_in_noinfo_lc*0.

    '''
    # this is for those objects that are crossing light cone closer to previous redshift
    if i != ind_start:
        pos_star_in_noinfo_lc[~eligibility_in_noinfo_lc] = pos_star_next
        vel_star_in_noinfo_lc[~eligibility_in_noinfo_lc] = vel_star_next
        halo_ind_in_noinfo_lc[~eligibility_in_noinfo_lc] = halo_ind_next
    '''
    
    # get chi star where lc crosses halo trajectory; bool is False where closer to previous
    chi_star_in_info_lc, pos_star_in_info_lc, vel_star_in_info_lc, bool_star_in_info_lc = solve_crossing(com_dist_prev_main_in_info_lc,com_dist_in_info_lc,pos_prev_main_in_info_lc,pos_in_info_lc,chi_prev,chi_in)
    
    # add ineligible halos if any from last iteration of the loop to those crossed in previous
    # marked ineligible (False) only if same halo has already been assigned to a light cone
    # for ~elig is True (assigned halos), bool is going to be true, so bool True if bad here
    # ~bool is either not closer to this redshift (but prev) (False) or has already been assigned (False)
    #bool_star_in_info_lc = (bool_star_in_info_lc) | (~eligibility_in_info_lc)
    bool_elig_star_in_info_lc = (bool_star_in_info_lc) & (eligibility_in_info_lc)#TESTING

    # number of objects in light cone
    N_in_star_lc = np.sum(bool_elig_star_in_info_lc)
    N_in_noinfo_lc = np.sum(mask_lc_in_noinfo)
    N_lc = N_in_star_lc+N_in_noinfo_lc

    print("with info and closer, noinfo, total = ",N_in_star_lc,N_in_noinfo_lc,N_lc,N_halos_in)

    # start new arrays for final output (assuming it is in and not prev)
    pos_interp_lc = np.zeros((N_lc,3))
    vel_interp_lc = np.zeros((N_lc,3))
    halo_ind_lc = np.zeros(N_lc,dtype=int)

    # record interpolated position and velocity
    pos_interp_lc[:N_in_star_lc] = pos_star_in_info_lc[bool_elig_star_in_info_lc]
    vel_interp_lc[:N_in_star_lc] = vel_star_in_info_lc[bool_elig_star_in_info_lc]
    halo_ind_lc[:N_in_star_lc] = halo_ind_in_info_lc[bool_elig_star_in_info_lc]
    pos_interp_lc[N_in_star_lc:N_lc] = pos_star_in_noinfo_lc
    vel_interp_lc[N_in_star_lc:N_lc] = vel_star_in_noinfo_lc
    halo_ind_lc[N_in_star_lc:N_lc] = halo_ind_in_noinfo_lc

    # create directory for this redshift
    if not os.path.exists(os.path.join(cat_lc_dir,"z%.3f"%z_in)):
        os.makedirs(os.path.join(cat_lc_dir,"z%.3f"%z_in))
    
    # adding contributions from the previous
    if i != ind_start:
        N_lc += len(halo_ind_next) # todo improve
        pos_interp_lc = np.vstack((pos_interp_lc,pos_star_next))
        vel_interp_lc = np.vstack((vel_interp_lc,vel_star_next))
        halo_ind_lc = np.hstack((halo_ind_lc,halo_ind_next))

    # save those arrays
    table_lc = np.empty(N_lc,dtype=[('halo_ind',halo_ind_lc.dtype),('pos_interp',(pos_interp_lc.dtype,3)),('vel_interp',(vel_interp_lc.dtype,3))])
    table_lc['halo_ind'] = halo_ind_lc
    table_lc['pos_interp'] = pos_interp_lc
    table_lc['vel_interp'] = vel_interp_lc
    np.save(os.path.join(cat_lc_dir,"z%.3f"%z_in,'table_lc.npy'),table_lc)
    
    # mark eligibility
    # version 1: only the main progenitor is marked ineligible; ~bool is closer to other
    halo_ind_next = halo_ind_prev_main_in_info_lc[~bool_star_in_info_lc]
    eligibility_prev[halo_ind_next] = False
    # get rid of objects assigned now since closer to this # TESTING
    halo_ind_assign = halo_ind_prev_main_in_info_lc[bool_star_in_info_lc]
    eligibility_prev[halo_ind_assign] = False
    # get rid of objects that have previously been assigned # TESTING
    halo_ind_inelig = halo_ind_prev_main_in_info_lc[~eligibility_in_info_lc]
    eligibility_prev[halo_ind_inelig] = False
    
    # version 2: all progenitors are marked ineligible
    # slower, but works (perhaps optimize with numba). Confusing part is why halo_inds has zeros
    '''
    # todo: slight issue is that we mask only for main prog - can perhaps add a clause that takes the main prog value if not an appropriate value is supplied by progs (main prog may not be in progs)
    for j in range(len(start_progs_in_info_lc[~bool_star_in_info_lc])):
        start = (start_progs_in_info_lc[~bool_star_in_info_lc])[j]
        num = (num_progs_in_info_lc[~bool_star_in_info_lc])[j]
        prog_inds = progs_in[start:start+num]
        prog_inds = correct_inds(prog_inds,N_halos_slabs_prev,slabs_prev)
        halo_inds = halo_ind_prev[prog_inds]
        #if j < 100: print(halo_inds, halo_ind_next[j])
        eligibility_prev[halo_inds] = False
    '''

    # information to keep for next redshift considered; should have dimensions equal to sum elig prev
    vel_star_next = vel_star_in_info_lc[~bool_star_in_info_lc]
    pos_star_next = pos_star_in_info_lc[~bool_star_in_info_lc]

    if want_plot and i != ind_start:
        # select the halos in the light cones
        pos_choice = pos_in[halo_ind_lc]

        # selecting thin slab
        pos_x_min = -995.
        pos_x_max = -985.

        ijk = 0
        choice = (pos_choice[:,ijk] >= pos_x_min) & (pos_choice[:,ijk] < pos_x_max)
        choice_lc = (pos_in_info[:,ijk] >= pos_x_min) & (pos_in_info[:,ijk] < pos_x_max)

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
    
    del com_dist_in, main_prog_in, halo_ind_in, pos_in, N_halos_slabs_in, slabs_in
    del com_dist_prev, main_prog_prev, halo_ind_prev, pos_prev, N_halos_slabs_prev, slabs_prev

    gc.collect()
    delta_chi_old = delta_chi
    eligibility_in = eligibility_prev
    copies_old = copies_prev
#dict_keys(['HaloIndex', 'HaloMass', 'HaloVmax', 'IsAssociated', 'IsPotentialSplit', 'MainProgenitor', 'MainProgenitorFrac', 'MainProgenitorPrec', 'MainProgenitorPrecFrac', 'NumProgenitors', 'Position', 'Progenitors'])
