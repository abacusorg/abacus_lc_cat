import glob
import asdf
import numpy as np
import os

def distance_period(x0, x1, Lbox):
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * Lbox, delta-Lbox, delta)
    dist = np.sqrt((delta**2).sum(axis=-1))
    return dist

# simulation name
#sim_name = "AbacusSummit_base_c000_ph006"
sim_name = "AbacusSummit_highbase_c021_ph000"

if 'highbase' in sim_name:
    Lbox = 1000.
else:
    Lbox = 2000.

# all merger tree redshifts; from low z to high
zs_mt = np.load("data/zs_mt.npy")

# directory where we have saved the final outputs from merger trees and halo catalogs
#cat_lc_dir = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"
cat_lc_dir = "/global/cscratch1/sd/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"

# initial redshift where we start building the trees
z_start = 0.5#0.45
z_stop = 0.5#0.575
ind_start = np.argmin(np.abs(zs_mt-z_start))
ind_stop = np.argmin(np.abs(zs_mt-z_stop))

# loop over each merger tree redshift
for i in range(ind_start,ind_stop+1):
    
    # starting snapshot
    z_in = zs_mt[i]
    print("redshift = ",z_in)

    fn = cat_lc_dir+"z%.3f/pid_lc.asdf"%(z_in)
    data_mt = asdf.open(fn, lazy_load=True, copy_arrays=True)['data']
    pid_lc = data_mt['pid']
    
    fn = cat_lc_dir+"z%.3f/pid_rv_lc.asdf"%(z_in)
    pid_rv_lc = asdf.open(fn, lazy_load=True, copy_arrays=True)['data']['pid']

    fn = cat_lc_dir+"z%.3f/halo_info_lc.asdf"%(z_in)
    data_halo = asdf.open(fn, lazy_load=True, copy_arrays=True)['data']

    print("loaded all three")
    n_halo = len(data_halo['npoutA'])
    nsubsamp = data_halo['npoutA'][:]
    print(nsubsamp.dtype)
    haloids = np.repeat(np.arange(n_halo), nsubsamp)
    print("haloids")
    npouts = np.repeat(nsubsamp, nsubsamp)
    print("npouts")
    assert len(haloids) == len(pid_lc), "number of halos and particles = %d %d"%(len(haloids),len(pid_lc))
    
    #print(len(np.unique(pid_lc))*100./len(pid_lc))
    #print(len(np.unique(pid_rv_lc))*100./len(pid_rv_lc))

    unmatched = pid_lc-pid_rv_lc != 0
    matched = pid_lc-pid_rv_lc == 0
    print("number unmatched = ",np.sum(unmatched))
    print("unmatched haloids = ",haloids[unmatched][:100])
    print("unmatched outs = ",npouts[unmatched][:100])
    if True:#try:
        halox = data_halo['x_L2com'][:]
        posx = np.repeat(halox[:,0], nsubsamp)
        posy = np.repeat(halox[:,1], nsubsamp)
        posz = np.repeat(halox[:,2], nsubsamp)
        pos_halos = np.vstack((posx,posy,posz)).T
        pos_lc = data_mt['pos'][:]

        dist = distance_period(pos_lc,pos_halos,Lbox)
        min_dist = np.min(dist)
        max_dist = np.max(dist)
        mean_dist = np.mean(dist)
        print("min, max, mean = ", min_dist, max_dist, mean_dist)
        
        np.save("pos_u_%d.npy"%i, pos_lc[unmatched])
        np.save("npout_u_%d.npy"%i, npouts[unmatched])
        np.save("haloid_u_%d.npy"%i, haloids[unmatched])
        np.save("dist_u_%d.npy"%i, dist[unmatched])
        np.save("pos_m_%d.npy"%i, pos_lc[matched])
    if False:#except:
        pass
    print("percentage match = ",np.sum(pid_lc-pid_rv_lc==0)*100./len(pid_lc))
    print("---------------------")
