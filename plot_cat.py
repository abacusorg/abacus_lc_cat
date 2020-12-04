import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import asdf
from scipy.interpolate import interp1d

# simulation name
#sim_name = "AbacusSummit_base_c000_ph006"
sim_name = "AbacusSummit_highbase_c000_ph100"

# directory where we save the final outputs
cat_lc_dir = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"

# all redshifts, steps and comoving distances of light cones files; high z to low z
zs_all = np.load("data_headers/redshifts.npy")
chis_all = np.load("data_headers/coord_dist.npy")
zs_all[-1] = np.float('%.1f'%zs_all[-1])

# get functions relating chi and z
chi_of_z = interp1d(zs_all,chis_all)
z_of_chi = interp1d(chis_all,zs_all)

if 'highbase' not in sim_name:
    x_min = -1000
    x_max = x_min+10.
else:
    x_min = -450
    x_max = x_min+10.

# choice of redshift

# base
#z = 0.5
#z = 0.576
#z = 0.651
z = 0.726
#z = 0.800
#z = 0.878
#z = 0.950
#z = 1.026
#z = 1.1
#z = 1.179

# highbase

z = 0.300
z = 0.351
z = 0.400
z = 0.450
z = 0.500 #mildly weird
z = 0.577
z = 0.652
z = 0.8


file_type = 'halo_info'
#file_type = 'pid_rv'
#file_type = 'table_lc'

if file_type == 'halo_info':
    fn = cat_lc_dir+"z%.3f/halo_info_lc.asdf"%(z)
    f = asdf.open(fn, lazy_load=True, copy_arrays=True)

    pos = f['data']['pos_interp']
    #pos = f['data']['x_L2com']

    del f

if file_type == 'pid_rv':
    fn = cat_lc_dir+"z%.3f/pid_rv_lc.asdf"%(z)
    f = asdf.open(fn, lazy_load=True, copy_arrays=True)
    pos = f['data']['pos']

    del f

if file_type == 'table_lc':
    vel = np.load(cat_lc_dir+"z%.3f/table_lc.npy"%(z))['vel_interp']
    not_interp = (np.sum(np.abs(vel),axis=1) - 0.) < 1.e-6
    print(np.sum(not_interp)/len(not_interp)*100.)
    
    pos = np.load(cat_lc_dir+"z%.3f/table_lc.npy"%(z))['pos_interp']
    chi = np.load(cat_lc_dir+"z%.3f/table_lc.npy"%(z))['chi_interp']
    
    z_interp = z_of_chi(chi)[not_interp]
    #pos = pos[~not_interp]

print(pos.shape)


def print_minimax(array):
    print("minimum value = ",array.min())
    print("maximum value = ",array.max())

    
x = pos[:,0]
y = pos[:,1]
z = pos[:,2]

print(np.sum(y>900)/len(y)*100.)

print_minimax(x)
print_minimax(y)
print_minimax(z)

choice = (x > x_min) & (x < x_max)

plt.scatter(y[choice],z[choice],s=0.1,alpha=1.)
#plt.scatter(y,z,s=0.01,alpha=1.)
plt.axis('equal')
plt.savefig('test.png')
plt.show()
