#!/usr/bin/env python3
'''
This is a script for loading simulation data and generating subsamples.

Usage
-----
$ python -m abacusnbody.hod.AbacusHOD.prepare_sim --path2config /path/to/config.yaml
'''

import os
from pathlib import Path
import yaml

import numpy as np
import random
import time
from astropy.table import Table
import h5py
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
from itertools import repeat
import argparse
import asdf

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

import multiprocessing
from multiprocessing import Pool

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/abacus_hod.yaml'


# https://arxiv.org/pdf/2001.06018.pdf Figure 13 shows redshift evolution of LRG HOD 
# the subsampling curve for halos
def subsample_halos(m, MT):
    x = np.log10(m)
    if MT:
        return 1.0/(1.0 + 10*np.exp(-(x - 11.2)*25)) # MT
    else:
        return 1.0/(1.0 + 0.1*np.exp(-(x - 13.3)*5)) # LRG only

def subsample_particles(m, MT):
    x = np.log10(m)
    # return 4/(200.0 + np.exp(-(x - 13.7)*8)) # LRG only
    if MT:
        return 4/(200.0 + np.exp(-(x - 13.2)*6)) # MT
    else:
        return 4/(200.0 + np.exp(-(x - 13.7)*8)) # LRG only

def get_smo_density_oneslab(i, simdir, simname, z_mock, N_dim, cleaning):
    slabname = simdir+simname+'/halos/z'+str(z_mock).ljust(5, '0')\
    +'/halo_info/halo_info_'+str(i).zfill(3)+'.asdf'

    cat = CompaSOHaloCatalog(
        slabname, fields = ['N', 'x_L2com'], cleaned_halos = cleaning)
    Lbox = cat.header['BoxSizeHMpc']
    halos = cat.halos

    if cleaning:
        halos = halos[halos['N'] > 0]

    # get a 3d histogram with number of objects in each cell                                                                       
    D, edges = np.histogramdd(halos['x_L2com'], weights = halos['N'],
        bins = N_dim, range = [[-Lbox/2, Lbox/2],[-Lbox/2, Lbox/2],[-Lbox/2, Lbox/2]])   
    return D


def get_smo_density(smo_scale, numslabs, simdir, simname, z_mock, N_dim, cleaning):   
    Dtot = 0
    for i in range(numslabs):
        Dtot += get_smo_density_oneslab(i, simdir, simname, z_mock, N_dim, cleaning)   

    # gaussian smoothing 
    Dtot = gaussian_filter(Dtot, sigma = smo_scale, mode = "wrap")

    # average number of particles per cell                                                                                         
    D_avg = np.sum(Dtot)/N_dim**3                                                                                                                                                                                                                              
    return Dtot / D_avg - 1

def prepare_slab(i, savedir, simdir, simname, z_mock, tracer_flags, MT, want_ranks, cleaning, N_dim, newseed, light_cones=False, light_cones_dir=''):
    outfilename_halos = savedir+'/halos_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushod'
    outfilename_particles = savedir+'/particles_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushod'
    print("processing slab ", i)
    if MT:
        outfilename_halos += '_MT'
        outfilename_particles += '_MT'
    if want_ranks:
        outfilename_particles += '_withranks'
    outfilename_particles += '_new.h5'
    outfilename_halos += '_new.h5'

    np.random.seed(newseed + i)
    # if file already exists, just skip
    if os.path.exists(outfilename_halos) \
    and os.path.exists(outfilename_particles):
        return 0


    # load the halo catalog slab
    print("loading halo catalog ")
    if light_cones:
        fields = ['N', 'N_interp', 'x_L2com', 'v_L2com', 'pos_interp', 'vel_interp', 'npstartA', 'npoutA', 'haloindex', 'sigmav3d_L2com']
    else:
        fields = ['N', 'x_L2com', 'v_L2com', 'r90_L2com', 'r25_L2com', 'npstartA', 'npoutA', 'id', 'sigmav3d_L2com']
    
    if light_cones:
        assert light_cones_dir != '', "You haven't specified light cone directory"
        # halo table filename
        halos_fn = os.path.join(light_cones_dir, 'halo_light_cones', simname, f'z{z_mock:4.3f}', 'lc_halo_info.asdf')
        
        # open the halo file
        with asdf.open(halos_fn, lazy_load=True, copy_arrays=True) as f:
            halos = f['data']
            header = f['header']
            cols = {col:np.array(halos[col]) for col in fields}
        halos = Table(cols, copy=False)

        # rename the columns to agree with the rest of the code
        halos['x_L2com'] = halos['pos_interp']
        halos['v_L2com'] = halos['vel_interp']
        halos['id'] = halos['haloindex']
        #halos['v_L2com'] = halos['vel_avg'] # use averaged particle positions
        halos['N'] = halos['N_interp']
        N_halos = len(halos['N'])
                
        # testing: needs to be changed once we copy all halo fields
        halos['r25_L2com'] = np.ones(N_halos)
        halos['r90_L2com'] = np.ones(N_halos)

        # load the particles
        with asdf.open(os.path.join(light_cones_dir, 'halo_light_cones', simname, f'z{z_mock:4.3f}', 'lc_pid_rv.asdf'), lazy_load=True, copy_arrays=True) as f:
            parts = f['data']
            header = f['header']
            cols = {col:np.array(parts[col]) for col in ['pos', 'vel']}
        parts = Table(cols, copy=False)
    else:
        slabname = simdir+simname+'/halos/z'+str(z_mock).ljust(5, '0')\
        +'/halo_info/halo_info_'+str(i).zfill(3)+'.asdf'

        cat = CompaSOHaloCatalog(slabname, subsamples=dict(A=True, rv=True), fields = fields, 
            cleaned_halos = cleaning)
        halos = cat.halos
        if cleaning:
            halos = halos[halos['N'] > 0]

        parts = cat.subsamples
        header = cat.header

    
    Lbox = header['BoxSizeHMpc']
    Mpart = header['ParticleMassHMsun'] # msun / h 
    H0 = header['H0']
    h = H0/100.0

    # # form a halo table of the columns i care about 
    # creating a mask of which halos to keep, which halos to drop
    p_halos = subsample_halos(halos['N']*Mpart, MT)
    mask_halos = np.random.random(N_halos) < p_halos
    print("total number of halos, ", N_halos, "keeping ", np.sum(mask_halos))

    halos['mask_subsample'] = mask_halos
    halos['multi_halos'] = 1.0 / p_halos

    nbins = 100
    mbins = np.logspace(np.log10(3e10), 15.5, nbins + 1)

    print("computing density rank")
    fenv_rank = np.zeros(N_halos)
    if light_cones:
        print("TBH, I am just lazy, but to do this properly would need to use density maps of the full boxes and figure out wrapping cause the light cones go beyond the box")
    else:
        dens_grid = np.array(h5py.File(savedir+"/density_field.h5", 'r')['dens'])
        ixs = np.floor((np.array(halos['x_L2com']) + Lbox/2) / (Lbox/N_dim)).astype(np.int) % N_dim
        halos_overdens = dens_grid[ixs[:, 0], ixs[:, 1], ixs[:, 2]]
        
        for ibin in range(nbins):
            mmask = (halos['N']*Mpart > mbins[ibin]) & (halos['N']*Mpart < mbins[ibin + 1])
            if np.sum(mmask) > 0:
                if np.sum(mmask) == 1:
                    fenv_rank[mmask] = 0
                else:
                    new_fenv_rank = halos_overdens[mmask].argsort().argsort()
                    fenv_rank[mmask] = new_fenv_rank / np.max(new_fenv_rank) - 0.5
    halos['fenv_rank'] = fenv_rank

    # compute delta concentration
    print("computing c rank")
    halos_c = halos['r90_L2com']/halos['r25_L2com']
    deltac_rank = np.zeros(N_halos)
    #if light_cones:
    #    print("Concentration not implemented!")
    if True:#else:
        for ibin in range(nbins):
            mmask = (halos['N']*Mpart > mbins[ibin]) & (halos['N']*Mpart < mbins[ibin + 1])
            if np.sum(mmask) > 0:
                if np.sum(mmask) == 1:
                    deltac_rank[mmask] = 0
                else:
                    new_deltac = halos_c[mmask] - np.median(halos_c[mmask])
                    new_deltac_rank = new_deltac.argsort().argsort()
                    deltac_rank[mmask] = new_deltac_rank / np.max(new_deltac_rank) - 0.5
    halos['deltac_rank'] = deltac_rank

    # the new particle start, len, and multiplier
    halos_pstart = halos['npstartA']
    halos_pnum = halos['npoutA']
    #halos_pstart = np.zeros(len(halos_pnum), dtype=int)
    #halos_pstart[1:] = np.cumsum(halos_pnum)[:-1]
    halos_pstart_new = np.zeros(N_halos)
    halos_pnum_new = np.zeros(N_halos)

    # particle arrays for ranks and mask
    N_parts = parts['vel'][:].shape[0]
    mask_parts = np.zeros(N_parts)
    len_old = N_parts
    ranks_parts = np.full(len_old, -1.0)
    ranksv_parts = np.full(len_old, -1.0)
    ranksr_parts = np.full(len_old, -1.0)
    ranksp_parts = np.full(len_old, -1.0)
    pos_parts = np.full((len_old, 3), -1.0)
    vel_parts = np.full((len_old, 3), -1.0)
    hvel_parts = np.full((len_old, 3), -1.0)
    Mh_parts = np.full(len_old, -1.0)
    Np_parts = np.full(len_old, -1.0)
    downsample_parts = np.full(len_old, -1.0)
    idh_parts = np.full(len_old, -1)
    deltach_parts = np.full(len_old, -1.0)
    fenvh_parts = np.full(len_old, -1.0)

    print("compiling particle subsamples")
    start_tracker = 0
    for j in np.arange(N_halos):
        if j % 10000 == 0:
            print("halo id", j, end = '\r')
        if mask_halos[j]:
            # updating the mask tagging the particles we want to preserve
            subsample_factor = subsample_particles(halos['N'][j] * Mpart, MT)
            submask = np.random.binomial(n = 1, p = subsample_factor, size = halos_pnum[j])
            # updating the particles' masks, downsample factors, halo mass
            mask_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = submask
            # print(j, halos_pstart, halos_pnum, p_halos, downsample_parts)
            downsample_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = p_halos[j]
            hvel_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = halos['v_L2com'][j]
            Mh_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = halos['N'][j] * Mpart # in msun / h
            Np_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = np.sum(submask)
            idh_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = halos['id'][j] 
            deltach_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = deltac_rank[j]
            fenvh_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = fenv_rank[j]

            # updating the pstart, pnum, for the halos
            halos_pstart_new[j] = start_tracker
            halos_pnum_new[j] = np.sum(submask)
            start_tracker += np.sum(submask)

            if want_ranks:
                if np.sum(submask) == 0:
                    continue
                # extract particle index
                indices_parts = np.arange(
                    halos_pstart[j], halos_pstart[j] + halos_pnum[j])[submask.astype(bool)]
                indices_parts = indices_parts.astype(int)
                if np.sum(submask) == 1:
                    ranks_parts[indices_parts] = 0
                    ranksv_parts[indices_parts] = 0
                    ranksp_parts[indices_parts] = 0
                    ranksr_parts[indices_parts] = 0
                    continue
                
                # make the rankings
                theseparts = parts[
                    halos_pstart[j]: halos_pstart[j] + halos_pnum[j]][submask.astype(bool)]
                theseparts_pos = theseparts['pos']
                theseparts_vel = theseparts['vel']
                theseparts_halo_pos = halos['x_L2com'][j]
                theseparts_halo_vel = halos['v_L2com'][j]

                dist2_rel = np.sum((theseparts_pos - theseparts_halo_pos)**2, axis = 1)
                newranks = dist2_rel.argsort().argsort() 
                ranks_parts[indices_parts] = (newranks - np.mean(newranks)) / np.mean(newranks)

                v2_rel = np.sum((theseparts_vel - theseparts_halo_vel)**2, axis = 1)
                newranksv = v2_rel.argsort().argsort() 
                ranksv_parts[indices_parts] = (newranksv - np.mean(newranksv)) / np.mean(newranksv)

                # get rps
                # calc relative positions
                r_rel = theseparts_pos - theseparts_halo_pos 
                r0 = np.sqrt(np.sum(r_rel**2, axis = 1))
                r_rel_norm = r_rel/r0[:, None]

                # list of peculiar velocities of the particles
                vels_rel = theseparts_vel - theseparts_halo_vel # velocity km/s
                # relative speed to halo center squared
                v_rel2 = np.sum(vels_rel**2, axis = 1) 

                # calculate radial and tangential peculiar velocity
                vel_rad = np.sum(vels_rel*r_rel_norm, axis = 1)
                newranksr = vel_rad.argsort().argsort() 
                ranksr_parts[indices_parts] = (newranksr - np.mean(newranksr)) / np.mean(newranksr)

                # radial component
                v_rad2 = vel_rad**2 # speed
                # tangential component
                v_tan2 = v_rel2 - v_rad2

                # compute the perihelion distance for NFW profile
                m = halos['N'][j]*Mpart / h # in kg
                rs = halos['r25_L2com'][j]
                c = halos['r90_L2com'][j]/rs
                r0_kpc = r0*1000 # kpc
                alpha = 1.0/(np.log(1+c)-c/(1+c))*2*6.67e-11*m*2e30/r0_kpc/3.086e+19/1e6

                # iterate a few times to solve for rp
                x2 = v_tan2/(v_tan2+v_rad2)

                num_iters = 20 # how many iterations do we want
                factorA = v_tan2 + v_rad2
                factorB = np.log(1+r0_kpc/rs)
                for it in range(num_iters):
                    oldx = np.sqrt(x2)
                    x2 = v_tan2/(factorA + alpha*(np.log(1+oldx*r0_kpc/rs)/oldx - factorB))
                x2[np.isnan(x2)] = 1
                # final perihelion distance 
                rp2 = r0_kpc**2*x2
                newranksp = rp2.argsort().argsort() 
                ranksp_parts[indices_parts] = (newranksp - np.mean(newranksp)) / np.mean(newranksp)

        else:
            halos_pstart_new[j] = -1
            halos_pnum_new[j] = -1

    halos['npstartA'] = halos_pstart_new
    halos['npoutA'] = halos_pnum_new
    halos['randoms'] = np.random.random(N_halos) # attaching random numbers
    halos['randoms_gaus_vrms'] = np.random.normal(loc = 0, 
        scale = halos["sigmav3d_L2com"]/np.sqrt(3), size = N_halos) # attaching random numbers

    # output halo file 
    print("outputting new halo file ")
    # output_dir = savedir+'/halos_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushodMT_new.h5'
    if os.path.exists(outfilename_halos):
        os.remove(outfilename_halos)
    print(outfilename_halos, outfilename_particles)
    newfile = h5py.File(outfilename_halos, 'w')

    if light_cones:
        halos = Table(halos)
        parts = Table(parts)
    
    dataset = newfile.create_dataset('halos', data = halos[mask_halos])
    newfile.close()

    # output the new particle file
    print("adding rank fields to particle data ")
    mask_parts = mask_parts.astype(bool)
    parts = parts[mask_parts]
    N_parts = parts['vel'][:].shape[0]
    print("pre process particle number ", len_old, " post process particle number ", N_parts)
    if want_ranks:
        parts['ranks'] = ranks_parts[mask_parts]
        parts['ranksv'] = ranksv_parts[mask_parts]
        parts['ranksr'] = ranksr_parts[mask_parts]
        parts['ranksp'] = ranksp_parts[mask_parts]
    parts['downsample_halo'] = downsample_parts[mask_parts]
    parts['halo_vel'] = hvel_parts[mask_parts]
    parts['halo_mass'] = Mh_parts[mask_parts]
    parts['Np'] = Np_parts[mask_parts]
    parts['halo_id'] = idh_parts[mask_parts]
    parts['randoms'] = np.random.random(N_parts)
    parts['halo_deltac'] = deltach_parts[mask_parts]
    parts['halo_fenv'] = fenvh_parts[mask_parts]
    
    print("are there any negative particle values? ", np.sum(parts['downsample_halo'] < 0), 
        np.sum(parts['halo_mass'] < 0))
    print("outputting new particle file ")
    # output_dir = savedir+'/particles_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushodMT_new.h5'
    if os.path.exists(outfilename_particles):
        os.remove(outfilename_particles)
    newfile = h5py.File(outfilename_particles, 'w')
    dataset = newfile.create_dataset('particles', data = parts)
    newfile.close()

    print("pre process particle number ", len_old, " post process particle number ", N_parts)

def main(path2config, params = None):
    print("compiling compaso halo catalogs into subsampled catalogs")

    config = yaml.load(open(path2config))
    # update params if needed
    if params is None:
        params = {}
    config.update(params)

    simname = config['sim_params']['sim_name'] # "AbacusSummit_base_c000_ph006"
    simdir = config['sim_params']['sim_dir']
    z_mock = config['sim_params']['z_mock']
    savedir = config['sim_params']['subsample_dir']+simname+"/z"+str(z_mock).ljust(5, '0') 
    cleaning = config['sim_params']['cleaned_halos']

    halo_info_fns = \
    list((Path(simdir) / Path(simname) / 'halos' / ('z%4.3f'%z_mock) / 'halo_info').glob('*.asdf'))
    numslabs = len(halo_info_fns)

    tracer_flags = config['HOD_params']['tracer_flags']
    MT = False
    if tracer_flags['ELG'] or tracer_flags['QSO']:
        MT = True
    want_ranks = config['HOD_params']['want_ranks']
    newseed = 600
    N_dim = config['HOD_params']['Ndim']

    os.makedirs(savedir, exist_ok = True)

    print("reading sim ", simname, "redshift ", z_mock)
    start = time.time()
    if not os.path.exists(savedir+"/density_field.h5"):
        dens_grid = get_smo_density(config['HOD_params']['density_sigma'],
             numslabs, simdir, simname, z_mock, N_dim, cleaning)
        print("Generating density field took ", time.time() - start)
        # np.savez(savedir+"/density_field", dens = dens_grid)
        newfile = h5py.File(savedir+"/density_field.h5", 'w')
        dataset = newfile.create_dataset('dens', data = dens_grid)
        newfile.close()

    p = multiprocessing.Pool(config['prepare_sim']['Nparallel_load'])
    p.starmap(prepare_slab, zip(range(numslabs), repeat(savedir), 
        repeat(simdir), repeat(simname), repeat(z_mock), 
        repeat(tracer_flags), repeat(MT), repeat(want_ranks), 
        repeat(cleaning), repeat(N_dim), repeat(newseed)))
    p.close()
    p.join()

    print("done, took time ", time.time() - start)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())

    main(**args)

    # # Simulation parameters
    # param_dict = {
    # 'sim_params' :
    #     {
    #     'sim_name': 'AbacusSummit_base_c000_ph006',                                 # which simulation 
    #     'sim_dir': '/mnt/gosling2/bigsims/',                                        # where is the simulation
    #     'output_dir': '/mnt/marvin1/syuan/scratch/data_mocks_summit_new',          # where to output galaxy mocks
    #     'subsample_dir': '/mnt/marvin1/syuan/scratch/data_summit/',                 # where to output subsample data
    #     'z_mock': 0.5,                                                             # which redshift slice
    #     'Nthread_load': 7                                                          # number of thread for organizing simulation outputs (prepare_sim)
    #     }
    # }
    # for i in range(25):
    #     param_dict['sim_params']['sim_name'] = 'AbacusSummit_base_c000_ph'+str(i).zfill(3)
    #     main(**args, params = param_dict)
