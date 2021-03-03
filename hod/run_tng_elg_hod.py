import numpy as np
import os
import glob
import multiprocessing
from multiprocessing import Pool
import yaml
from itertools import repeat

# abacus readers
from compaso_halo_catalog import CompaSOHaloCatalog
import asdf

# HOD script
import generate_hod_catalog as galcat

# user choices
#simname = 'AbacusSummit_highbase_c000_ph100'
sim_name = 'AbacusSummit_base_c000_ph006'

# location where light cones are saved
cat_lc_dir = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"
#cat_lc_dir = "/global/cscratch1/sd/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"

def extract_redshift(fn):
    red = float(fn.split('z')[-1][:5])
    return red

# random seeds
seeds = [0]

sim_slices = sorted(glob.glob(os.path.join(cat_lc_dir,'z*')))
redshifts = [extract_redshift(sim_slices[i]) for i in range(len(sim_slices))]
print("redshifts = ",redshifts)

# load header to read parameters
f = asdf.open(os.path.join(sim_slices[0],'halo_info_lc.asdf'),lazy_load=True,copy_arrays=False)
header = f['header']

# parameters
params = {}
params['simdir'] = cat_lc_dir
params['h'] = header['H0']/100.
params['Nslab'] = 1 # only one because light cones
params['Lbox'] = header['BoxSize'] # box size in Mpc/h
params['Mpart'] = header['ParticleMassHMsun'] # Msun/h, mass of each particle 
params['velz2kms'] = header['VelZSpace_to_kms']/params['Lbox']# B.H. check # H(z)/(1+z), km/s/Mpc
params['n_cutoff'] = 0 # minimum number of particles
params['m_cutoff'] = params['n_cutoff']*params['Mpart']
params['seeds'] = seeds

# user choices
params['simname'] = sim_name
params['want_pid'] = False
params['rsd'] = False
params['verbose'] = False
params['num_sims'] = len(redshifts) # do for how many redshifts

def taylor_expand(logM_0,logM_prime,a,a_0):
    logM = logM_0 + logM_prime*(a_0-a)
    return logM

#for key in params.keys():
#print(key,params[key])

def gen_gal_onesim_onehod(design, param):
    savedir = param['datadir']
    
    for eseed in param['seeds']: 
        savedir = os.path.join(savedir,"model_%d"%param['model_no'])
        if param['rsd']:
            savedir = savedir+"_rsd"
        # if we are doing repeats, save them in separate directories
        if not eseed == 0:
            savedir = savedir+"_%3d"%eseed
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            
        param['savedir'] = savedir
        
        # generate mock
        galcat.gen_gal_cat(design, param, whatseed = eseed)


if __name__ == "__main__":

    # HOD parameters
    # median redshift of the sample

    # read params from yaml file
    path2config = 'config/TNG_ELG_HOD.yaml'
    config = yaml.load(open(path2config))
    der_dic = {}
    fid_dic = {}
    zs = np.zeros(3)
    pars = np.zeros(3)
    for key in config['z0']['ELG_params'].keys():
        for i in range(3):
            zs[i] = config[f'z{i}']['z']
            pars[i] = config[f'z{i}']['ELG_params'][key]
            if i == 0:
                fid_dic[key] = config[f'z{i}']['ELG_params'][key]
                z_fid = config[f'z{i}']['z']
        der_dic[key] = (pars[-1] - pars[0])/(zs[-1] - zs[0])
    params['model_no'] = 5
        
    newparams = []
    newdesigns = []
    for i in range(len(redshifts)):
        # check exist
        if not os.path.exists(os.path.join(sim_slices[i],'halo_info_lc.asdf')) or \
           not os.path.exists(os.path.join(sim_slices[i],'pid_rv_lc.asdf')):
            print("Missing files for z = ",redshifts[i]); continue

        redshift = redshifts[i]
        print("Files exist for z = ",redshift)

        # TESTING
        if redshift > 1.4: continue
        
        newparam = params.copy()
        newparam['subsample_directory'] = sim_slices[i]
        newparam['z'] = redshift
        newparam['datadir'] = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/HOD/z%.3f"%redshift # where to save output
        newparams.append(newparam)

        # TESTING
        if redshift <= 0.726:
            newdesigns.append(fid_dic)
            continue
        
        newdesign = {}
        for key in fid_dic.keys():
            newdesign[key] = fid_dic[key] + der_dic[key]*(redshift - z_fid)        
        newdesigns.append(newdesign)

        print(newdesign.items())

    # generate one halo catalog
    #gen_gal_onesim_onehod(newdesign,newparam)

    p = multiprocessing.Pool(40)
    p.starmap(gen_gal_onesim_onehod, zip(newdesigns,newparams))
    p.close()
    p.join()
    
