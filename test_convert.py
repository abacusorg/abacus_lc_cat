import numpy as np
import time
import glob
import os

ind = 8
#ind = 12
#ind = 16

pos_u = np.load('pos_u_%d.npy'%ind)
pid_u = np.load('pid_u_%d.npy'%ind)
npout_u = np.load('npout_u_%d.npy'%ind)
haloid_u = np.load('haloid_u_%d.npy'%ind)
pos_m = np.load('pos_m_%d.npy'%ind)
print(pos_u.shape,pos_m.shape)



# HERE WE FIND PARTICLES THAT ARE OFF THE EDGES
min_dist = 10.
far_u = (np.abs(np.abs(pos_u[:,0])-500.)>min_dist) & (np.abs(np.abs(pos_u[:,1])-500.)>min_dist) & (np.abs(np.abs(pos_u[:,2])-500.)>min_dist)

pos_u = pos_u[far_u]
pid_u = pid_u[far_u]
npout_u = npout_u[far_u]
haloid_u = haloid_u[far_u]

far_m = (np.abs(np.abs(pos_m[:,0])-500.)>min_dist) & (np.abs(np.abs(pos_m[:,1])-500.)>min_dist) & (np.abs(np.abs(pos_m[:,2])-500.)>min_dist)
pos_m = pos_m[far_m]

dist_u = np.sqrt(np.sum((pos_u-np.array([-990,-990,-990]))**2,axis=1))
dist_m = np.sqrt(np.sum((pos_m-np.array([-990,-990,-990]))**2,axis=1))
print(dist_m.min(),dist_m.mean(),dist_m.max())
print(dist_u.min(),dist_u.mean(),dist_u.max())
print(pos_u.shape)
print(dist_u)
#print(pos_u[:150])
print(haloid_u[:150])
print(npout_u[:150])
#quit()

# THIS PART IS FOR LOOKING FOR THESE PARTICLES IN THE SHELLS
from pathlib import Path
from tools.aid_asdf import load_lc_pid_rv
from bitpacked import unpack_rvint

sim_name = "AbacusSummit_highbase_c021_ph000"
light_cone_parent = Path("/global/project/projectdirs/desi/cosmosim/Abacus")

# directory where light cones are saved
lc_dir = light_cone_parent / sim_name / "lightcones"
lc_rv_fns = sorted(glob.glob(os.path.join(lc_dir, 'rv/LightCone1*')))
lc_pid_fns = sorted(glob.glob(os.path.join(lc_dir, 'pid/LightCone1*')))

Lbox = 1000.
PPD = 3456
ids = np.array([0,90])
print("Looking for:")
for idx in ids: print(pid_u[idx],pos_u[idx],dist_u[idx])

for i in range(142,len(lc_pid_fns)):
    lc_pid_fn = lc_pid_fns[i]
    lc_rv_fn = lc_rv_fns[i]

    t1 = time.time()
    lc_pid, lc_rv = load_lc_pid_rv(lc_pid_fn,lc_rv_fn,Lbox,PPD)
    print(lc_pid_fn, "took ", time.time()-t1)



    for j in range(len(ids)):
        missing_pid = pid_u[ids[j]]
        if (missing_pid == lc_pid).any():
            print("FOUND %d in file = "%missing_pid,lc_pid_fn)
            pos, vel = unpack_rvint(lc_rv[missing_pid == lc_pid],Lbox)
            print("true and lc pos = ",pos,pos_u[ids[j]])
            
            if 'LightCone1' in lc_pid_fn:
                observer = np.array([-990,-990,-2990])
            elif 'LightCone2' in lc_pid_fn:
                observer = np.array([-990,-2990,-990])
            else: observer = np.array([-990.,-990.,-990.])
            print("distance to observer = ",np.sqrt(np.sum((observer-pos_u[ids[j]])**2)))
            quit()
quit()

# THIS PART IS FOR RECORDING POSITIONS AS TXT FILES FOR PLOTTING IN PLOTLY
def downsample(pos,fac=1000):
    inds = np.arange(pos.shape[0])
    print(len(inds))
    np.random.shuffle(inds)
    inds = inds[::fac]
    print(len(inds))
    pos = pos[inds]
    return pos

pos_m = downsample(pos_m,fac=1000)
#pos_u = downsample(pos_u,fac=1000)
np.savetxt('pos_m.csv', pos_m, delimiter=",")
np.savetxt('pos_u.csv', pos_u, delimiter=",")
quit()
