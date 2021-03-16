import asdf
import numpy as np
import os
from numba import jit
from bitpacked import unpack_rvint, unpack_pids

def load_lc_pid_rv(lc_pid_fn, lc_rv_fn, Lbox, PPD):
    # load and unpack pids
    lc_pids = asdf.open(lc_pid_fn, lazy_load=True, copy_arrays=True)
    lc_pid = lc_pids['data']['packedpid'][:]
    lc_pid = unpack_pids(lc_pid, box=Lbox, ppd=PPD, pid=True)['pid']
    lc_pids.close()

    # load positions and velocities
    lc_rvs = asdf.open(lc_rv_fn, lazy_load=True, copy_arrays=True)
    lc_rv = lc_rvs['data']['rvint'][:]
    lc_rvs.close()
    return lc_pid, lc_rv

def load_mt_pid(mt_fn,Lbox,PPD):
    # load mtree catalog
    print("load mtree file = ",mt_fn)
    mt_pids = asdf.open(mt_fn, lazy_load=True, copy_arrays=True)
    mt_pid = mt_pids['data']['pid'][:]
    mt_pid = unpack_pids(mt_pid, box=Lbox, ppd=PPD, pid=True)['pid']
    header = mt_pids['header']
    mt_pids.close()

    return mt_pid, header

def load_mt_pid_pos_vel(mt_fn,Lbox,PPD):
    # load mtree catalog
    print("load mtree file = ",mt_fn)
    mt_pids = asdf.open(mt_fn, lazy_load=True, copy_arrays=True)
    mt_pid = mt_pids['data']['pid'][:]
    mt_pid = unpack_pids(mt_pid, box=Lbox, ppd=PPD, pid=True)['pid']
    mt_pos = mt_pids['data']['pos'][:]
    mt_vel = mt_pids['data']['vel'][:]
    header = mt_pids['header']
    mt_pids.close()

    return mt_pid, mt_pos, mt_vel, header

def load_mt_npout(halo_mt_fn):
    # load mtree catalog
    print("load halo mtree file = ",halo_mt_fn)
    f = asdf.open(halo_mt_fn, lazy_load=True, copy_arrays=True)
    mt_npout = f['data']['npoutA'][:]
    f.close()
    return mt_npout

def load_mt_origin(halo_mt_fn):
    # load mtree catalog
    print("load halo mtree file = ",halo_mt_fn)
    f = asdf.open(halo_mt_fn, lazy_load=True, copy_arrays=True)
    mt_origin = f['data']['origin'][:]
    f.close()
    return mt_origin

@jit(nopython = True)
def reindex_pid(pid, npstart, npout):
    npstart_new = np.zeros(len(npout),dtype=np.int64)
    npstart_new[1:] = np.cumsum(npout)[:-1]
    npout_new = npout
    pid_new = np.zeros(np.sum(npout_new), dtype=pid.dtype)

    for j in range(len(npstart)):
        pid_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = pid[npstart[j]:npstart[j]+npout[j]]

    return pid_new, npstart_new, npout_new

@jit(nopython = True)
def reindex_pid_pos(pid,pos,npstart,npout):
    npstart_new = np.zeros(len(npout),dtype=np.int64)
    npstart_new[1:] = np.cumsum(npout)[:-1]
    npout_new = npout
    pid_new = np.zeros(np.sum(npout_new),dtype=pid.dtype)
    pos_new = np.zeros((np.sum(npout_new),3),dtype=pos.dtype)
    
    for j in range(len(npstart)):
        pid_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = pid[npstart[j]:npstart[j]+npout[j]]
        pos_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = pos[npstart[j]:npstart[j]+npout[j]]
            
    return pid_new, pos_new, npstart_new, npout_new

@jit(nopython = True)
def reindex_pid_pos_vel(pid,pos,vel,npstart,npout):
    npstart_new = np.zeros(len(npout),dtype=np.int64)
    npstart_new[1:] = np.cumsum(npout)[:-1]
    npout_new = npout
    pid_new = np.zeros(np.sum(npout_new),dtype=pid.dtype)
    pos_new = np.zeros((np.sum(npout_new),3),dtype=pos.dtype)
    vel_new = np.zeros((np.sum(npout_new),3),dtype=vel.dtype)
    
    for j in range(len(npstart)):
        pid_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = pid[npstart[j]:npstart[j]+npout[j]]
        pos_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = pos[npstart[j]:npstart[j]+npout[j]]
        vel_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = vel[npstart[j]:npstart[j]+npout[j]]
            
    return pid_new, pos_new, vel_new, npstart_new, npout_new


# save light cone catalog
def save_asdf(table,filename,header,cat_lc_dir):
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
    output_file.write_to(os.path.join(cat_lc_dir,filename+".asdf"))
    output_file.close()
