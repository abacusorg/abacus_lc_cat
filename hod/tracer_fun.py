#!/usr/bin/env python

# increasing logM1 goes downwards for satellites amplitude
# increasing logM_cut pushes both centrals and satellites right but both asymptote to the same for large mass
# kappa moves things right for sats but they asymptote to same
# sigma changes the width of the centrals transitioning
# gamma determines the steepness of the initial climb for centrals - increasing means steeper
"""
at z = 0.8:

    ELG_params:
        p_max: 0.18
        Q: 100.
        logM_cut: 11.8
        kappa: 1.8
        sigma: 0.58
        logM1: 13.73
        alpha: 0.7
        gamma: 6.12
        A_s: 1.

at z = 1.1:


at z = 1.4:

    ELG_params:
        p_max: 0.1
        Q: 100.
        logM_cut: 12.2
        kappa: 1.5
        sigma: 0.58
        logM1: 13.73
        alpha: 0.7
        gamma: 5.22

"""


"""
Script for calculating the HOD function for satellite
and central galaxies of the following three types:
 - luminous red galaxies (LRGs)
 - emission-line galaxies (ELGs)
 - quasi stellar objects (QSOs)

Usage:
------
$ ./tracer_fun --help
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erf
import argparse
import yaml
import asdf

def N_sat(M_h, M_cut, kappa, M_1, alpha, A_s=1., **kwargs):
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for all tracers with an optional amplitude parameter, A_s.
    """
    
    N = A_s*((M_h-kappa*M_cut)/M_1)**alpha
    N[np.isnan(N)] = 0.
    
    return N

def N_cen_LRG(M_h, M_cut, sigma, **kwargs):
    """
    Standard Zheng et al. (2005) central HOD parametrization for LRGs.
    """
    N = 0.5*(1 + erf((np.log10(M_h)-np.log10(M_cut))/sigma))
    return N

def N_cen_ELG_v1(M_h, p_max, Q, M_cut, sigma, gamma, **kwargs):
    """
    HOD function for ELG centrals taken from arXiv:1910.05095.
    """
    phi = phi_fun(M_h, M_cut, sigma)
    Phi = Phi_fun(M_h, M_cut, sigma, gamma)
    A = A_fun(p_max, Q, phi, Phi)
    N = 2.*A*phi*Phi + 1./(2.*Q)*(1 + erf((np.log10(M_h)-np.log10(M_cut))/0.01))
    return N

def N_cen_ELG_v2(M_h, p_max, M_cut, sigma, gamma, **kwargs):
    """
    HOD function for ELG centrals taken from arXiv:2007.09012.
    """
    N = np.zeros(len(M_h))
    N[M_h <= M_cut] = p_max*Gaussian_fun(np.log10(M_h[M_h <= M_cut]), np.log10(M_cut), sigma)
    N[M_h > M_cut] = p_max*(M_h[M_h > M_cut]/M_cut)**gamma/(np.sqrt(2.*np.pi)*sigma)
    return N

def N_cen_QSO(M_h, p_max, M_cut, sigma, **kwargs):
    """
    HOD function (Zheng et al. (2005) with p_max) for QSO centrals taken from arXiv:2007.09012.
    """
    N = 0.5*p_max*(1 + erf((np.log10(M_h)-np.log10(M_cut))/sigma))
    return N


def phi_fun(M_h, M_cut, sigma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    phi = Gaussian_fun(np.log10(M_h),np.log10(M_cut), sigma)
    return phi

def Phi_fun(M_h, M_cut, sigma, gamma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    x = gamma*(np.log10(M_h)-np.log10(M_cut))/sigma
    Phi = 0.5*(1 + erf(x/np.sqrt(2)))
    return Phi
    
def A_fun(p_max, Q, phi, Phi):
    """
    Aiding function for N_cen_ELG_v1().
    """
    A = (p_max-1./Q)/np.max(2.*phi*Phi)
    return A
    
def Gaussian_fun(x, mean, sigma):
    """
    Gaussian function with centered at `mean' with standard deviation `sigma'.
    """
    return norm.pdf(x, loc=mean, scale=sigma)

def main(tracer, path2config, want_plot=False):
    # HOD design for each tracer:
    #LRG: M_cut, kappa, sigma, M_1, alpha
    #ELG_v1: p_max, Q, M_cut, kappa, sigma, M_1, alpha, gamma
    #ELG_v2: p_max, M_cut, kappa, sigma, M_1, alpha, A_s
    #QSO: p_max, M_cut, kappa, sigma, M_1, alpha, A_s

    # Example values
    if tracer == 'ELG_v1':
        p_max = 0.33;
        Q = 100.;
        M_cut = 10.**11.7;#11.75
        kappa = 1.5;#1.
        sigma = 0.58;
        M_1 = 10.**13.33;#13.53
        alpha = 0.8;#1.
        gamma = 4.12;
        A_s = 1.
        
    elif tracer == 'ELG_v2':
        p_max = 0.00537;
        Q = 100.;
        M_cut = 10.**11.515;
        kappa = 1.;
        sigma = 0.08;
        M_1 = 10.**13.53;
        alpha = 1.;
        gamma = -1.4;
        A_s = 1.

    else:
        p_max = 0.33;
        Q = 100.;
        M_cut = 10.**11.7;
        kappa = 1.;
        sigma = 0.58;
        M_1 = 10.**13.3;
        alpha = 0.8;
        gamma = 4.12;
        A_s = 1.
        
    HOD_design = {
        'p_max': p_max,
        'Q': Q,
        'M_cut': M_cut,
        'kappa': kappa,
        'sigma': sigma,
        'M_1': M_1,
        'alpha': alpha,
        'gamma': gamma,
        'A_s': A_s
    }

    # TESTING
    config = yaml.load(open(path2config))
    example = config['z0']
    redshift = example['z']
    HOD_design = example['ELG_params']
    HOD_design['M_cut'] = 10.**HOD_design['logM_cut']
    HOD_design['M_1'] = 10.**HOD_design['logM1']
        
    
    # location where light cones are saved
    sim_name = 'AbacusSummit_base_c000_ph006'
    cat_lc_dir = "/mnt/gosling1/boryanah/light_cone_catalog/"+sim_name+"/halos_light_cones/"
    with asdf.open(cat_lc_dir+"z%4.3f/halo_info_lc.asdf"%redshift, lazy_load=True, copy_arrays=True) as f:
        N = f['data']['N'][:].astype(np.float)
        header = f['header']
    m_part = header['ParticleMassHMsun']
    N *= m_part
    print("40 particles = %.1e"%(m_part*40))

    print(len(N), redshift)
    print("41904189 rows 0.800725654346")
    
    # select range for computing the HOD function
    #bins = np.logspace(11, 15, 100)
    bins = np.logspace(np.log10(40*m_part), 15, 100)
    
    hmf, bins = np.histogram(N, bins=bins)
    bin_cents = (bins[1:]+bins[:-1])*.5
    M_h = bin_cents
    
    # calculate the HOD function for the satellites and centrals
    if tracer == 'LRG':
        HOD_cen = N_cen_LRG(M_h, **HOD_design)
    elif tracer == 'ELG_v1':
        HOD_cen = N_cen_ELG_v1(M_h, **HOD_design)
    elif tracer == 'ELG_v2':
        HOD_cen = N_cen_ELG_v2(M_h, **HOD_design)
    elif tracer == 'QSO':
        HOD_cen = N_cen_QSO(M_h, **HOD_design)
    HOD_sat = N_sat(M_h, **HOD_design)


    hist_sat = hmf*HOD_sat
    hist_cen = hmf*HOD_cen
    print(HOD_design.items())
    
    print("expected number of sats = ", np.sum(hist_sat))
    print("expected number of cents = ", np.sum(hist_cen))
    quit()
    plt.plot(bin_cents, hist_cen)
    plt.plot(bin_cents, hist_sat)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    if want_plot:
        keys = ['gamma']
        for key in keys:
            print(key)
            for i in range(3):
                HOD_design[key] = (HOD_design[key]+i*1)
                HOD_cen = N_cen_ELG_v1(M_h, **HOD_design)
                HOD_sat = N_sat(M_h, **HOD_design)
                plt.plot(M_h, HOD_sat, ls='--', label=str(HOD_design[key]+i*0.1))
                plt.plot(M_h, HOD_cen, ls='-', label=tracer)
            plt.legend()
            plt.xlim([1.e11, 1.e15])
            plt.ylim([1.e-3, 1.e1])
            plt.xscale("log")
            plt.yscale("log")
            plt.show()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--tracer', help='Select tracer type', choices=["LRG", "ELG_v1", "ELG_v2", "QSO"], default='ELG_v1')
    parser.add_argument('--path2config', help='Path to configuration file', default='config/TNG_ELG_HOD.yaml')
    parser.add_argument('--want_plot', help='Plot HOD distribution?', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
