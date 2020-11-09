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

from compaso_halo_catalog import CompaSOHaloCatalog

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

# save light cone catalog
def save_asdf(table,filename,header,cat_lc_dir,z_in,i_chunk=None):
    # cram into a dictionary
    data_dict = {}
    for j in range(len(table.dtype.names)):
        field = table.dtype.names[j]
        data_dict[field] = table[field]
        
    # create data tree structure
    data_tree = {
        "data": data_dict,
        "header": header,
    }

    # save the data and close file
    output_file = asdf.AsdfFile(data_tree)
    if i_chunk is None:
        output_file.write_to(os.path.join(cat_lc_dir,"z%.3f"%z_in,filename+"_z%.3f.%d.asdf"%(z_in,i_chunk)))
    else:
        output_file.write_to(os.path.join(cat_lc_dir,"z%.3f"%z_in,filename+"_z%.3f.asdf"%(z_in)))
    output_file.close()

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
want_plot = True

# location of the origin in Mpc/h
origin = np.array([-990.,-990.,-990.])

# simulation name
sim_name = "AbacusSummit_base_c000_ph006"
#sim_name = "AbacusSummit_highbase_c000_ph100"

# directory where the halo catalogs are saved
#cat_dir = "/mnt/store/lgarrison/"+sim_name+"/halos/"
cat_dir = "/mnt/store2/bigsims/"+sim_name+"/halos/"

# directory where we save the final outputs
cat_lc_dir = "/mnt/store1/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"

# directory where the merger tree is stored
#merger_dir = "/mnt/store/AbacusSummit/merger/"+sim_name+"/"
merger_dir = "/mnt/store2/bigsims/merger/"+sim_name+"/"

# all merger tree snapshots and corresponding redshifts
snaps_mt = sorted(glob.glob(merger_dir+"associations_z*.0.asdf"))
# more accurate, slightly slower
if not os.path.exists("data/zs_mt.npy"):
    zs_mt = get_zs_from_headers(snaps_mt)
    np.save("data/zs_mt.npy",zs_mt)
zs_mt = np.load("data/zs_mt.npy")

# fields are we extracting from the catalogs
fields_cat = ['id','npstartA','npoutA','N','x_L2com','v_L2com']

# initial redshift where we start building the trees
z_start = 0.5
#z_start = np.min(zs_mt)
ind_start = np.argmin(np.abs(zs_mt-z_start))

# loop over each merger tree redshift
for i in range(ind_start,len(zs_mt)-1):

    # starting snapshot
    z_in = zs_mt[i]
    
    # load the light cone arrays
    table_lc = np.load(os.path.join(cat_lc_dir,"z%.3f"%z_in,'table_lc.npy'))
    halo_ind_lc = table_lc['halo_ind']
    pos_interp_lc = table_lc['pos_interp']
    vel_interp_lc = table_lc['vel_interp']

    # catalog directory
    #catdir = os.path.join(cat_dir,"z%.3f"%z_in,'halo_info','halo_info_%03d.asdf'%(i_chunk))
    catdir = os.path.join(cat_dir,"z%.3f"%z_in)
    
    # load halo catalog, setting unpack to False for speed
    cat = CompaSOHaloCatalog(catdir, load_subsamples='A_halo_pid', fields=fields_cat, unpack_bits = False)
    
    # halo catalog
    halo_table = cat.halos[halo_ind_lc]
    header = cat.header
    N_halos = len(cat.halos)
    print("N_halos = ",N_halos)

    # load the pid, set unpack_bits to True if you want other information
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

    # append new fields; if 0 velocity change with the original
    halo_table['index_halo'] = halo_ind_lc
    halo_table['x_interp'] = pos_interp_lc
    halo_table['v_interp'] = vel_interp_lc

    # save to files
    #save_asdf(halo_table,"halo_info_lc",header,cat_lc_dir,z_in,i_chunk)
    #save_asdf(pid_table,"pid_lc",header,cat_lc_dir,z_in,i_chunk)
    save_asdf(halo_table,"halo_info_lc",header,cat_lc_dir,z_in)
    save_asdf(pid_table,"pid_lc",header,cat_lc_dir,z_in)

    # update eligibility array
    eligibility_in = eligibility_prev

    # delete things at the end
    del pid, pid_new, pid_table, npstart, npout, npstart_new, npout_new
    del halo_table
    del cat

    gc.collect()
    
#dict_keys(['HaloIndex', 'HaloMass', 'HaloVmax', 'IsAssociated', 'IsPotentialSplit', 'MainProgenitor', 'MainProgenitorFrac', 'MainProgenitorPrec', 'MainProgenitorPrecFrac', 'NumProgenitors', 'Position', 'Progenitors'])
