import numpy as np
import glob
import os
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# hod model number
model_no = 5#4#3#2#1

#z_in = 1.400
#z_in = 1.327
#z_in = 1.251
#z_in = 1.179
#z_in = 1.100
#z_in = 1.026
#z_in = 0.950
#z_in = 0.878
z_in = 0.800
#z_in = 0.726
#z_in = 0.651
#z_in = 0.576
#z_in = 0.500


blah, bin_left, bin_cents, bin_width, dndz = np.loadtxt("nz_blanc.txt", delimiter=',', unpack=True)
#bin_cents = .5*(bin_edges[1:]+bin_edges[:-1])
print(bin_cents)
octant = 41523./8.
n_gal = dndz*octant*bin_width
arg = np.argmin(np.abs(bin_cents-z_in))
print("expected number of galaxies = ",n_gal[arg])
print("expected dndz = ",dndz[arg])

# HOD directory on alan
hod_dir = "/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/HOD/z%.3f/model_%d_rsd/"%(z_in,model_no)

sats_fns = sorted(glob.glob(hod_dir+"*sats*"))
cent_fns = sorted(glob.glob(hod_dir+"*cent*"))

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
n_gals = sats_arr.shape[0]+cent_arr.shape[0]
print("number of gals in bin = ",n_gals)

# first three columns of file are positions, next are velocities, then halo index and finally halo mass
sats_pos = sats_arr[:,0:3]
cent_pos = cent_arr[:,0:3]
sats_vel = sats_arr[:,3:6]
cent_vel = cent_arr[:,3:6]
sats_mass = sats_arr[:,-1]
cent_mass = cent_arr[:,-1]

dist = np.sqrt(np.sum((cent_pos - np.array([10, 10, 10]))**2, axis=1))
dist_min = dist[dist > 1500].min()
dist_max = dist[dist < 3000].max()
print("median, min, max, diff = ",np.median(dist), dist_min, dist_max, dist_max-dist_min)

vol_diff = 4./3*np.pi*(dist_max**3-dist_min**3)/8.
n = n_gals/vol_diff
print("number density of sample = ", n)
n_goal = n_gal[arg]/vol_diff
print("number density goal = ", n_goal)
print("fractional difference [percentage] = ", (n-n_goal)*100./n)

print("Satellites = ", sats_pos.shape[0])
print("Centrals = ", cent_pos.shape[0])

x_min = 100
x_max = x_min+40.
i = 1
j = 2
k = 0
sel_cent = (cent_pos[:,k] > x_min) & (cent_pos[:,k] < x_max)
sel_sats = (sats_pos[:,k] > x_min) & (sats_pos[:,k] < x_max)


print(np.sum((sats_pos[:, 0] > 10000)))
print(np.sum((sats_pos[:, 0] < -100.)))
print(np.sum((sats_pos[:, 0] > 4500) | (sats_pos[:, 0] < -100)))
print(np.sum((sats_pos[:, 1] > 4500) | (sats_pos[:, 1] < -100)))
print(np.sum((sats_pos[:, 2] > 4500) | (sats_pos[:, 2] < -100)))
print(np.min(sats_mass), np.max(sats_mass))
print(sats_pos[:,0].min())
print(sats_pos[:,0].max())
print(sats_pos[:,1].min())
print(sats_pos[:,1].max())
print(sats_pos[:,2].min())
print(sats_pos[:,2].max())
print(sats_vel[:,0].min())
print(sats_vel[:,0].max())
print(sats_vel[:,1].min())
print(sats_vel[:,1].max())
print(sats_vel[:,2].min())
print(sats_vel[:,2].max())
print(cent_pos.min())
print(cent_pos.max())

print("number of centrals in cross section = ",np.sum(sel_cent))
print("number of satellites in cross section = ",np.sum(sel_sats))

plt.title("Cross-section of the simulation")
plt.scatter(cent_pos[sel_cent,i],cent_pos[sel_cent,j],color='dodgerblue',s=1,alpha=0.8,label='centrals')
plt.scatter(sats_pos[sel_sats,i],sats_pos[sel_sats,j],color='orangered',s=1,alpha=0.8,label='satellites')
plt.axis('equal')
plt.xlabel('Y [Mpc]')
plt.ylabel('Z [Mpc]')
plt.legend()
plt.savefig("scatter.png")
plt.show()
