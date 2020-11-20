#!/usr/bin/env python

"""
Module implementation of a simple Halo Occupation 
Distribution (HOD) for the AbacusSummit simulations. 

Add to .bashrc:
export PYTHONPATH="/path/to/AbacusSummitHOD:$PYTHONPATH"

"""

import numpy as np
import os
from math import erfc
import asdf
import h5py
from scipy import special
from glob import glob
from numba import jit

@jit(nopython = True)
def n_cen(M_in, design_array, m_cutoff = 4e12): 
    """
    Computes the expected number of central galaxies given a halo mass and 
    the HOD design. 

    Parameters
    ----------

    M_in : float
        Halo mass in solar mass.

    design_array : np array
        Array containing the five baseline HOD parameters. 
        
    m_cutoff: float, optional
        Ignore halos smaller than this mass.

    Returns
    -------

    n_cen : float
        Number of centrals expected for the halo within the range (0, 1).
        This number should be interpreted as a probability.

    """
    if M_in < m_cutoff:
        return 0
    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    return 0.5*erfc(np.log(M_cut/M_in)/(2**.5*sigma))

@jit(nopython = True)
def n_sat(M_in, design_array, m_cutoff = 4e12): 
    """
    Computes the expected number of satellite galaxies given a halo mass and 
    the HOD design. 

    Parameters
    ----------

    M_in : float
        Halo mass in solar mass.

    design_array : np array
        Array containing the five baseline HOD parameters. 
        
    m_cutoff: float, optional
        Ignore halos smaller than this mass.

    Returns
    -------

    n_sat : float
        Expected number of satellite galaxies for the said halo.

    """

    if M_in < m_cutoff: # this cutoff ignores halos with less than 100 particles
        return 0

    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    if M_in < kappa*M_cut:
        return 0

    return ((M_in - kappa*M_cut)/M1)**alpha*0.5*erfc(np.log(M_cut/M_in)/(2**.5*sigma))


def gen_cent(halo_ids, halo_pos, halo_vels, halo_vrms, halo_mass, design_array, rsd, fcent, velz2kms, lbox, m_cutoff = 4e12, whatseed = 0):
    """
    Function that generates central galaxies and its position and velocity 
    given a halo catalog and HOD designs. The generated 
    galaxies are output to file fcent. 

    Parameters
    ----------

    halo_ids : numpy.array
        Array of halo IDs.

    halo_pos : numpy.array
        Array of halo positions of shape (N, 3) in box units.

    halo_vels : numpy.array
        Array of halo velocities of shape (N, 3) in km/s.

    halo_vrms: numpy.array
        Array of halo particle velocity dispersion in km/s.

    halo_mass : numpy.array
        Array of halo mass in solar mass.

    design_array : np array
        Array containing the five baseline HOD parameters. 
        
    fcent : file pointer
        Pointer to the central galaxies output file location. 

    rsd : boolean
        Flag of whether to implement RSD. 

    velz2kms : float
        Parameter for converting velocity to RSD position

    lbox : float
        Side length of the box

    m_cutoff : float, optional
        Cut-off mass in Msun

    whatseed: int, optional
        RNG seed

    Outputs
    -------

    For each halo, if there exists a central, the function outputs the 
    3D position (Mpc/h), halo ID, and halo mass (Msun/h) to file.

    """

    # TESTING
    #rng = np.random.RandomState(whatseed)

    # parse out the hod parameters 
    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    # form the probability array
    ps = 0.5*special.erfc(np.log(M_cut/halo_mass)/(2**.5*sigma))

    # generate a bunch of numbers for central occupation
    #r_cents = rng.random(len(ps))
    r_cents = np.random.random(len(ps))
    # do we have centrals?
    mask_cents = r_cents < ps 

    # generate central los velocity
    vrms_los = halo_vrms/1.7320508076 # km/s
    #extra_vlos = rng.normal(loc = 0, scale = vrms_los)
    extra_vlos = np.random.normal(loc = 0, scale = vrms_los)# testing

    # compile the centrals
    pos_cents = halo_pos[mask_cents]
    vel_cents = halo_vels[mask_cents]
    vel_cents[:, 2] += extra_vlos[mask_cents] # add on velocity bias
    mass_cents = halo_mass[mask_cents]
    ids_cents = halo_ids[mask_cents]

    # rsd
    if rsd:
        pos_cents[:, 2] = (pos_cents[:, 2] + vel_cents[:, 2]/velz2kms) #TESTING % lbox

    # output to file
    newarray = np.concatenate((pos_cents, vel_cents, ids_cents[:, None], mass_cents[:, None]), axis = 1)
    newarray.tofile(fcent)



@jit(nopython = True)
def gen_sats(halo_ids, halo_pos, halo_vels, halo_mass, halo_pstart, halo_pnum, part_pos, part_vel, part_pid, rsd, velz2kms, lbox, m_cutoff = 4e12, whatseed = 0, want_pid = False):
    
    """
    Function that generates satellite galaxies and their positions and 
    velocities given a halo catalog and HOD designs. 

    The generated galaxies are output to binary file fsats. 

    Parameters
    ----------

    halo_ids : numpy.array
        Array of halo IDs.

    halo_pos : numpy.array
        Array of halo positions of shape (N, 3) in box units.

    halo_vels : numpy.array
        Array of halo velocities of shape (N, 3) in km/s.

    halo_mass : numpy.array
        Array of halo mass in solar mass.

    halo_pstart : numpy.array
        Array of particle start indices for each halo.

    halo_pnum : numpy.array
        Array of number of particles for halos. 

    part_pos : numpy.array
        Array of particle positions

    part_vel : numpy.array
        Array of particle velocities

    part_pid : numpy.array
        Array of particle ids

    design_array : np array
        Array containing the five baseline HOD parameters. 
                
    rsd : boolean
        Flag of whether to implement RSD. 

    velz2kms : float
        Parameter for converting velocity to RSD position

    lbox : float
        Side length of the box

    m_cutoff : float, optional
        Cut-off mass in Msun

    Outputs
    -------

    For each halo, the function returns the satellite galaxies, specifically
    the 3D position (Mpc/h), velocities, halo ID, and halo mass (Msun/h)


    """
    #np.random.seed(whatseed + 14838492)TESTING

    # standard hod design
    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]


    # TODO: don't hardcode 9 in
    # loop through the halos to populate satellites
    data_sats = np.zeros((1, 9))
    for i in np.arange(len(halo_ids)):
        thishalo_id = halo_ids[i]
        thishalo_mass = halo_mass[i]
        # load the particle subsample belonging to the halo
        start_ind = halo_pstart[i]
        # converting to int32 since numba does not recognize arange on uint32
        numparts = np.int32(halo_pnum[i])
        # if there are no particles in the particle subsample, move on
        if numparts == 0:
            continue

        # extract the particle positions and vels
        ss_pos = part_pos[start_ind: start_ind + numparts] # Mpc/h
        ss_vels = part_vel[start_ind: start_ind + numparts] # km/s
        ss_pids = part_pid[start_ind: start_ind + numparts] # integer

        # generate a list of random numbers that will track each particle
        random_list = np.random.random(numparts)

        # compute the expected number of satellites
        # TESTING
        #N_sat = n_sat(halo_mass[i], design_array, m_cutoff) * halo_multi[i]
        N_sat = n_sat(halo_mass[i], design_array, m_cutoff)

        if N_sat == 0:
            continue
        # we do this step after generating the random list  

        # the probability of each particle hosting a satellite
        eachprob = float(N_sat)/numparts
        eachprob_array = np.ones(numparts)*eachprob
        totprob = np.sum(eachprob_array)

        temp_indices = np.arange(numparts)
        temp_range = numparts - 1
        
        newmask = random_list < eachprob_array
        # generate the position and velocity of the galaxies
        sat_pos = ss_pos[newmask]
        sat_vels = ss_vels[newmask]
        sat_pids = ss_pids[newmask]

        # so a lot of the sat_pos are empty, in that case, just pass
        if len(sat_pos) == 0:
            continue

        # rsd, modify the satellite positions by their velocities
        if rsd:
            sat_pos[:,2] = (sat_pos[:,2] + sat_vels[:,2]/velz2kms)# TESTING % lbox

        # output
        for j in range(len(sat_pos)):
            newline_sat = np.array([[sat_pos[j, 0],
                                     sat_pos[j, 1],
                                     sat_pos[j, 2],
                                     sat_vels[j, 0],
                                     sat_vels[j, 1],
                                     sat_vels[j, 2],
                                     sat_pids[j],
                                     thishalo_id, 
                                     thishalo_mass]])
            data_sats = np.vstack((data_sats, newline_sat))

    return data_sats[1:]


def gen_gals(directory, design, fcent, fsats, rsd, params, m_cutoff = 4e12, whatseed = 0, want_pid = False):
    """
    Function that compiles halo catalog from directory and implements 
    HOD. 
    
    The galaxies are output to two binary files, one for centrals, one for satellites. 

    Parameters
    ----------

    directory : string 
        Directory of the halo and particle files. 

    design : dict
        Dictionary of the five baseline HOD parameters. 

    fcent : file pointer
        Pointer to the central galaxies output file location. 

    fsats : file pointer
        Pointer to the satellite galaxies output file location. 

    rsd : boolean
        Flag of whether to implement RSD. 

    params : dict
        Dictionary of various simulation parameters. 

    m_cutoff: float, optional
        Ignore halos smaller than this mass.

    """

    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 
                                                      'M1', 
                                                      'sigma', 
                                                      'alpha', 
                                                      'kappa'))

    # TESTING
    #design_array = np.array([M_cut, M1, sigma, alpha, kappa])
    global design_array
    design_array = np.array([M_cut, M1, sigma, alpha, kappa])

    # box size
    lbox = params['Lbox']

    # z coordinate to velocity
    velz2kms = params['velz2kms']
    
    # loop over all the halos files and pull out the relevant data fields 
    files = glob(os.path.join(directory,'halo_info_lc*.asdf'))   
    num_files = len(files)
    print("number of files = ",num_files)
    numhalos = 0 # track how many halos we include
    for i in np.arange(num_files):

        # open the halo files
        halos = asdf.open(files[i])
        maskedhalos = halos['data']
        
        # extracting the halo properties that we need
        halo_ids = maskedhalos['id'][:] # halo IDs
        #halo_pos = maskedhalos['x_L2com'] # halo positions, Mpc/h
        halo_pos = maskedhalos['pos_interp'][:]+lbox/2. # halo positions, Mpc/h
        #halo_vels = maskedhalos['v_L2com'] # halo velocities, km/s
        halo_vels = maskedhalos['vel_interp'][:] # halo velocities, km/s
        #halo_vrms = maskedhalos['sigmav3d_L2com'] # halo velocity dispersions, km/s
        halo_vrms = np.ones(halo_vels.shape[0])*.0003 # TODO TESTING FIXME # halo velocity dispersions, km/s
        halo_mass = maskedhalos['N'][:]*params['Mpart'] # halo mass, Msun/h
        halo_pstart = maskedhalos['npstartA'][:] # starting index of particles
        halo_pnum = maskedhalos['npoutA'][:] # number of particles
        halo_multi = np.ones(len(halo_pnum))
        halo_submask = np.ones(len(halo_pnum),dtype=bool)
        #halo_multi = maskedhalos['multi_halos'] # tuks no need 
        #halo_submask = maskedhalos['mask_subsample']  #tuks no need

        # for each halo, generate central galaxies and output to file
        gen_cent(halo_ids, halo_pos, halo_vels, halo_vrms, halo_mass, 
                 design_array, rsd, fcent, velz2kms, lbox, m_cutoff = m_cutoff, whatseed = whatseed)

        # open particle file
        particles = asdf.open(os.path.join(directory,'pid_rv_lc.asdf'))
        subsample = particles['data']
        part_pos = subsample['pos'][:]+lbox/2. # Mpc/h
        part_vel = subsample['vel'][:] # km/s
        if want_pid:
            part_pid = subsample['pid'][:]
        else:
            part_pid = np.zeros(part_pos.shape[0],dtype=int)
        
        # for each halo, generate satellites and output to file        
        data_sats = gen_sats(halo_ids, halo_pos, 
                             halo_vels, halo_mass, halo_pstart, 
                             halo_pnum, part_pos, part_vel, part_pid,
                             rsd, velz2kms, lbox, m_cutoff = m_cutoff,
                             whatseed = whatseed, want_pid = want_pid)

        
        data_sats.tofile(fsats)

        halos.close()
        particles.close()


def gen_gal_cat(design, params, whatseed = 0):
    """
    Main interface that takes in the simulation number, HOD design and 
    and outputs the resulting central and satellite galaxy
    catalogs to file in binary format. 

    Parameters
    ----------

    design : dict
        Dictionary of the five baseline HOD parameters. 

    params : dict
        Dictionary of various simulation parameters. 

    whatseed : integer, optional
        The initial seed to the random number generator. 

    """

    if not type(params['rsd']) is bool:
        print("Error: rsd has to be a boolean.")

    # directory where the subsamples are stored #tuks
    subsample_directory = params['subsample_directory']
    
    # if this directory does not exist, create it
    if not os.path.exists(params['savedir']):
        os.makedirs(params['savedir'])

    # binary output galaxy catalog
    if params['verbose']:
        print("Building galaxy catalog (binary output)")
    fcent = open(os.path.join(params['savedir'],"halos_gal_cent"),'wb')
    fsats = open(os.path.join(params['savedir'],"halos_gal_sats"),'wb')

    # find the halos, populate them with galaxies and write them to files
    gen_gals(subsample_directory, design, fcent, fsats, params['rsd'], params, m_cutoff = params['m_cutoff'], whatseed = whatseed, want_pid = params['want_pid'])

    # close the files in the end
    fcent.close()
    fsats.close()
