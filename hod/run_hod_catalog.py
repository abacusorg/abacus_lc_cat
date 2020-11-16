import numpy as np
import os
import glob
import multiprocessing
from multiprocessing import Pool
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


z = 1.179
z = 1.100
z = 1.026
z = 0.950
z = 0.878
z = 0.800
z = 0.726
#z = 0.651
#z = 0.576
 

# list of redshifts TODO check if there is a halo_info and pid_rv_lc file
redshift = z
redshifts = [redshift] # read from glob glob

# simulation slice # tuks redshift[0]
sim_slice = os.path.join(cat_lc_dir,'z%.3f'%redshifts[0])

# load header to read parameters tuks
f = asdf.open((os.path.join(sim_slice,'halo_info_lc.asdf')),lazy_load=True,copy_arrays=False)
header = f['header']

# parameters
params = {}
params['simdir'] = cat_lc_dir
params['simname'] = sim_name
params['want_pid'] = False
params['h'] = header['H0']/100.
params['Nslab'] = 1 # only one because light cones
params['Lboxh'] = header['BoxSize']
params['Lbox'] = params['Lboxh']/params['h'] # Mpc, box size 
params['Mpart'] = header['ParticleMassHMsun']/params['h'] # Msun, mass of each particle 
params['velz2kms'] = header['VelZSpace_to_kms']/params['Lbox']# B.H. check # H(z)/(1+z), km/s/Mpc
params['n_cutoff'] = 70 # minimum number of particles
params['m_cutoff'] = params['n_cutoff']*params['Mpart']
params['num_sims'] = 1 # len redshifts can probs do for many redshifts
params['rsd'] = True
params['verbose'] = False

# tuks should not be handed as params but rather zipped
params['subsample_directory'] = sim_slice
params['z'] = redshift
params['datadir'] = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/HOD/z%.3f"%redshift # where do you want to save the output


def taylor_expand(logM_0,logM_prime,a,a_0):
    logM = logM_0 + logM_prime*(a-a_0)
    return logM

for key in params.keys():
    print(key,params[key])

allseeds = [0] 
def gen_gal_onesim_onehod(design, datadir = params['datadir'], params = params, verbose = params['verbose']):
    for eseed in allseeds: 
        M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 
                                                          'M1', 
                                                          'sigma', 
                                                          'alpha', 
                                                          'kappa'))

        if params['rsd']:
            datadir = datadir+"_rsd"
        savedir = os.path.join(datadir,"CompaSO_%.2f_%.2f_%.2f_%.2f_%.2f"%(np.log10(M_cut),np.log10(M1),sigma,alpha,kappa))
        if params['rsd']:
            savedir = savedir+"_rsd"
        # if we are doing repeats, save them in separate directories
        if not eseed == 0:
            savedir = savedir+"_%3d"%eseed
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        # generate mock
        galcat.gen_gal_cat(design, params, savedir, whatseed = eseed, rsd = params['rsd'], m_cutoff = params['m_cutoff'], verbose = verbose, want_pid = params['want_pid'])


if __name__ == "__main__":
    # example hod

    # median redshift of the sample
    z_0 = 0.8
    a_0 = 1./(1+z_0)

    # base HOD parameters
    logM_cut = 13.27
    logM1 = 14.30
    sigma = 0.78
    alpha = 1.09
    kappa = 0.21

    # first derivative parameters
    logM_cut_prime = 1.e-5
    logM1_prime = 1.e-5

    newdesigns = []
    for i in range(len(redshifts)):
        a = 1./1+redshifts[i]
        logM_cut = taylor_expand(logM_cut,logM_cut_prime,a,a_0)
        logM1 = taylor_expand(logM1,logM1_prime,a,a_0)
        print(logM_cut,logM1)
        
        newdesign = {'M_cut': 10.**logM_cut, 
                     'M1': 10.**logM1,
                     'sigma': sigma,
                     'alpha': alpha,
                     'kappa': kappa}

        newdesigns.append(newdesign)

    
    # generate one halo catalog
    #gen_gal_onesim_onehod(0,newdesign)

    p = multiprocessing.Pool(2)
    p.starmap(gen_gal_onesim_onehod, zip(newdesigns))
    #p.starmap(gen_gal_onesim_onehod, zip(newdesign))
    p.close()
    p.join()
