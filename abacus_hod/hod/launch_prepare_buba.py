import os
import glob

import asdf

from prepare_buba import prepare_slab

# simulation and light cones specs
simname = 'AbacusSummit_base_c000_ph006'
#light_cones_dir = '/mnt/gosling1/boryanah/light_cone_catalog/'
light_cones_dir = '/global/cscratch1/sd/boryanah/light_cone_catalog/'
num_slabs = 1

tracer_flags = {'LRG': True, 'ELG': True, 'QSO': True}

# dimensions for the density
N_dim = 1024

# seeding
newseed = 600

# not needed cause we are using light cones
simdir = ''

# True if using ELGs
MT = True

# we don't need those for now
want_ranks = False
cleaning = True

def extract_redshift(fn):
    red = float(fn.split('z')[-1][:5])
    return red

# random seeds
cat_lc_dir = os.path.join(light_cones_dir, simname, 'halos_light_cones')
sim_slices = sorted(glob.glob(os.path.join(cat_lc_dir,'z*')))
redshifts = [extract_redshift(sim_slices[i]) for i in range(len(sim_slices))]
print("redshifts = ",redshifts)

# loop through all available redshifts
for z_mock in redshifts:

    # location to save the subsampled halos and particles
    savedir = f'/global/cscratch1/sd/boryanah/AbacusHOD_scratch/'+simname+'/z{z_mock:4.3f}/'
    #savedir = f'/mnt/gosling1/boryanah/AbacusHOD_scratch/'+simname+f'/z{z_mock:4.3f}/'
    print(savedir)

    for i in range(num_slabs):
        prepare_slab(i, savedir, simdir, simname, z_mock, tracer_flags, MT, want_ranks, cleaning, N_dim, newseed, light_cones=True, light_cones_dir=light_cones_dir)
