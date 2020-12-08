import numpy as np
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# hod model number
model_no = 2#1

z_in = 0.800
z_in = 0.576

# HOD directory on alan
hod_dir = "/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/HOD/z%.3f/model_%d_rsd/"%(z_in,model_no)

sats_fns = sorted(glob.glob(hod_dir+"*sats*"))
cent_fns = sorted(glob.glob(hod_dir+"*cent*"))

print(len(cent_fns),len(sats_fns))

def load_gals(fns,dim):

    for fn in fns:
        tmp_arr = np.fromfile(fn).reshape(-1,dim)
        try:
            gal_arr = np.vstack((gal_arr,tmp_arr))
        except:
            gal_arr = tmp_arr
            
    return gal_arr

sats_arr = load_gals(sats_fns,dim=9)
cent_arr = load_gals(cent_fns,dim=9)

# first three columns of file are positions, next are velocities, then halo index and finally halo mass
sats_pos = sats_arr[:,0:3]
cent_pos = cent_arr[:,0:3]
sats_mass = sats_arr[:,-1]
cent_mass = cent_arr[:,-1]

print(sats_pos.shape)
print(cent_pos.shape)

print(cent_pos.min())
print(cent_pos.max())
print(sats_pos.min())
print(sats_pos.max())

x_min = 0
x_max = x_min+40.
i = 1
j = 2
k = 0
sel_cent = (cent_pos[:,k] > x_min) & (cent_pos[:,k] < x_max)
sel_sats = (sats_pos[:,k] > x_min) & (sats_pos[:,k] < x_max)

print("number of centrals in cross section = ",np.sum(sel_cent))
print("number of satellites in cross section = ",np.sum(sel_sats))

plt.title("Cross-section of the simulation")
plt.scatter(cent_pos[sel_cent,i],cent_pos[sel_cent,j],color='dodgerblue',s=1,alpha=0.8,label='centrals')
plt.scatter(sats_pos[sel_sats,i],sats_pos[sel_sats,j],color='orangered',s=1,alpha=0.8,label='satellites')
plt.axis('equal')
plt.xlabel('Y [Mpc]')
plt.ylabel('Z [Mpc]')
plt.savefig("scatter.png")
plt.show()
