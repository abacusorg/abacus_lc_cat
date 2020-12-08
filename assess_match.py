import glob
import asdf
import numpy as np
import os


# simulation name
sim_name = "AbacusSummit_base_c000_ph006"

# all merger tree redshifts; from low z to high
zs_mt = np.load("data/zs_mt.npy")

# directory where we have saved the final outputs from merger trees and halo catalogs
cat_lc_dir = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"

# initial redshift where we start building the trees
z_start = 0.5
z_stop = 0.8
ind_start = np.argmin(np.abs(zs_mt-z_start))
ind_stop = np.argmin(np.abs(zs_mt-z_stop))

# loop over each merger tree redshift
for i in range(ind_start,ind_stop+1):
    
    # starting snapshot
    z_in = zs_mt[i]
    print("redshift = ",z_in)
    fn = cat_lc_dir+"z%.3f/pid_lc.asdf"%(z_in)
    pid_lc = asdf.open(fn, lazy_load=True, copy_arrays=True)['data']['pid']
    fn = cat_lc_dir+"z%.3f/pid_rv_lc.asdf"%(z_in)
    pid_rv_lc = asdf.open(fn, lazy_load=True, copy_arrays=True)['data']['pid']

    print("percentage match = ",np.sum(pid_lc-pid_rv_lc==0)*100./len(pid_lc))
    print("---------------------")
