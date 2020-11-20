import numpy as np
import matplotlib.pyplot as plt
import asdf

# simulation name
#sim_name = "AbacusSummit_base_c000_ph006"
sim_name = "AbacusSummit_highbase_c000_ph100"

# directory where we save the final outputs
cat_lc_dir = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"

# choice of redshift

# base

#z = 0.5
#z = 0.576
#z = 0.651
#z = 0.726
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

#file_type = 'halo_info'
#file_type = 'pid_rv'
file_type = 'table_lc'

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
    pos = np.load(cat_lc_dir+"z%.3f/table_lc.npy"%(z))['pos_interp']
    vel = np.load(cat_lc_dir+"z%.3f/table_lc.npy"%(z))['vel_interp']
    not_interp = np.sum(vel,axis=1) == 0.
    print(np.sum(not_interp)/len(not_interp)*100.)
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

x_min = 490.
x_max = x_min+10.

choice = (x > x_min) & (x < x_max)

plt.scatter(y[choice],z[choice],s=0.1,alpha=1.)
#plt.scatter(y,z,s=0.01,alpha=1.)
plt.axis('equal')
plt.show()
