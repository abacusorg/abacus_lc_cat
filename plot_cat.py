import numpy as np
import matplotlib.pyplot as plt
import asdf

# simulation name
sim_name = "AbacusSummit_base_c000_ph006"

# directory where we save the final outputs
cat_lc_dir = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"

# choice of redshift
z = 0.5
#z = 0.576
#z = 0.651
#z = 0.726

fn = cat_lc_dir+"z%.3f/halo_info_lc_z%.3f.asdf"%(z,z)
#fn = cat_lc_dir+"z%.3f/pid_rv_lc_z%.3f.asdf"%(z,z)
#f = asdf.open(fn, lazy_load=True, copy_arrays=True)
#pos = f['data']['pos_interp']
#pos = f['data']['x_L2com']
#pos = f['data']['pos']
#del f

pos = np.load(cat_lc_dir+"z%.3f/table_lc.npy"%(z))['pos_interp']
vel = np.load(cat_lc_dir+"z%.3f/table_lc.npy"%(z))['vel_interp']
not_interp = np.sum(vel,axis=1) == 0.
print(np.sum(not_interp)/len(not_interp)*100.)
pos = pos[~not_interp]
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

x_min = 0
x_max = x_min+10.

choice = (x > x_min) & (x < x_max)

plt.scatter(y[choice],z[choice],s=0.01,alpha=1.)
#plt.scatter(y,z,s=0.01,alpha=1.)
plt.axis('equal')
plt.show()
