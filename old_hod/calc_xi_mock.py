# allsims testhod rsd
import numpy as np
import os,sys
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import rc, rcParams
rcParams.update({'font.size': 12})
from astropy.table import Table
import astropy.io.fits as pf
from astropy.cosmology import WMAP9 as cosmo
import scipy.spatial as spatial
import multiprocessing
from multiprocessing import Pool
from scipy import signal
from matplotlib import gridspec


from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0= 67.26 , Om0=0.316)

# start timer
startclock = time.time()

import Corrfunc
# from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
# from Corrfunc.utils import convert_3d_counts_to_cf, convert_rp_pi_counts_to_wp
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.theory import wp, xi


from halotools.mock_observables import s_mu_tpcf  
from halotools.mock_observables import tpcf_multipole, tpcf

from halotools import mock_observables as mo 

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


np.random.seed(100)

# constants
params = {}
params['z'] = 0.5
params['h'] = 0.6726
params['Nslab'] = 3
params['Lboxh'] = 1100 # Mpc / h, box size
params['num_sims'] = 8#20

# rsd?
rsd = True
params['rsd'] = rsd
# central design 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi
# newdesign = {'M_cut': 10**13.388561621598246, 
#              'M1': 10**14.368350668860444, 
#              'sigma': 0.9578873443987804, 
#              'alpha': 1.125255701631058,
#              'kappa': 0.1627141175814102}    
# newdecor = {'s': 0.12091331472434913, 
#             's_v': 0.46545652711766233, 
#             'alpha_c': 0.04257541855657376, 
#             's_p': -0.9950033794379908, 
#             's_r': 0,
#             'A': -0.49232121475774787,
#             'Ae': 0.04645331702616213} 
# this fit actually has a very similar chi2 in the new fitter, chi2_xi = 63.38, whereas the newkappa fit gives 66.46

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, reseed 1
# newdesign = {'M_cut': 10**13.211232264704174, 
#              'M1': 10**14.316803985512022, 
#              'sigma': 0.2617982615453086, 
#              'alpha': 1.1787590802716943,
#              'kappa': 1.5399404686059694}    
# newdecor = {'s': -0.4686954037796145, 
#             's_v': 0.35010100395893207, 
#             'alpha_c': 0.2563084066785468, 
#             's_p': -0.7220797667024432, 
#             's_r': 0,
#             'A': -0.8411989940174688,
#             'Ae': 0.032483226555552476} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, reseed 2
# newdesign = {'M_cut': 10**13.25620369098887, 
#              'M1': 10**14.337715288487576, 
#              'sigma': 0.3731978386363788, 
#              'alpha': 1.2592809185458456,
#              'kappa': 1.8023282617534613}    
# newdecor = {'s': -0.7817889142443399, 
#             's_v': 0.09362682674905184, 
#             'alpha_c': 0.26380332168750653, 
#             's_p': -0.6175930893744729, 
#             's_r': 0,
#             'A': -0.9792035780331434,
#             'Ae': 0.040580417868850166} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, reseed 4
# newdesign = {'M_cut': 10**13.146986296143194, 
#              'M1': 10**14.163271507352293, 
#              'sigma': 0.10000000000000009, 
#              'alpha': 1.0244808619203352,
#              'kappa': 2.7278293002792084}    
# newdecor = {'s': -0.4989105053211327, 
#             's_v': 0.23874873235709332, 
#             'alpha_c': 0.2282824961454798, 
#             's_p': -0.8818913370145306, 
#             's_r': 0,
#             'A': -0.8420508746374157,
#             'Ae': 0.0483451164405961} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, rm1bin
# newdesign = {'M_cut': 10**13.220179144868366, 
#              'M1': 10**14.168186217263013, 
#              'sigma': 0.5952553999364837, 
#              'alpha': 1.0457807430919726,
#              'kappa': 2.072535726724783}    
# newdecor = {'s': -0.3739325254748556, 
#             's_v': -0.04313425697400186, 
#             'alpha_c': 0.1892419000187775, 
#             's_p': -0.9978838713022982, 
#             's_r': 0,
#             'A': -0.6035399569409844,
#             'Ae': 0.04097581506207816} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, rm1bin, reseed 2
# newdesign = {'M_cut': 10**13.229604867721681, 
#              'M1': 10**14.247419627020166, 
#              'sigma': 0.4862729710015685, 
#              'alpha': 1.1059165453298727,
#              'kappa': 1.9020154400124676}    
# newdecor = {'s': -0.35110158604599606, 
#             's_v': 0.16678372890844861, 
#             'alpha_c': 0.21897041491908698, 
#             's_p': -0.9740790671756002, 
#             's_r': 0,
#             'A': -0.7323458808572154,
#             'Ae': 0.037295692757868426} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, rm1bin, lic
# newdesign = {'M_cut': 10**13.502834939366624, 
#              'M1': 10**14.41324163972613, 
#              'sigma': 1.0764296653464305, 
#              'alpha': 1.1032730750573532,
#              'kappa': 0.5232916879175762}    
# newdecor = {'s': -0.16207268570904765, 
#             's_v': 0.3124173390195846, 
#             'alpha_c': 0.149938748482915, 
#             's_p': -1.0, 
#             's_r': 0,
#             'A': -0.6785265662374191,
#             'Ae': 0.04598190164886478} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, rm1bin, lic, reseed 2
# newdesign = {'M_cut': 10**13.528722924208395, 
#              'M1': 10**14.407270108635105, 
#              'sigma': 1.09455419152773, 
#              'alpha': 1.0462571761094808,
#              'kappa': 0.5898107319115208}    
# newdecor = {'s': -0.10009770845953002, 
#             's_v': 0.19651843138395803, 
#             'alpha_c': 0.18217209084935337, 
#             's_p': -0.9990435229367586, 
#             's_r': 0,
#             'A': -0.5822632568449885,
#             'Ae': 0.036230521835892565} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, asym lic
# newdesign = {'M_cut': 10**13.49198697522889, 
#              'M1': 10**14.372808384954217, 
#              'sigma': 1.0431960561773315, 
#              'alpha': 1.1617412040765824,
#              'kappa': 1.024343285840858}    
# newdecor = {'s': -0.4864991000630932, 
#             's_v': -0.04265903615254899, 
#             'alpha_c': 0.19022436841461454, 
#             's_p': -0.9791909434181436, 
#             's_r': 0,
#             'A': -0.7551931474898375,
#             'Ae': 0.051145003622344344} 
# this fit has a chi2_xi = 62.68 in the new kappa fitter

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, newpibins
# newdesign = {'M_cut': 10**13.153017057862609, 
#              'M1': 10**14.184127811640048, 
#              'sigma': 0.3986009878956881, 
#              'alpha': 1.1397472049176691,
#              'kappa': 1.8495599852572444}    
# newdecor = {'s': -0.3305881786849435, 
#             's_v': 0.1535091017151968, 
#             'alpha_c': 0.14741607693318914, 
#             's_p': -1.0, 
#             's_r': 0,
#             'A': -0.7562075964709362,
#             'Ae': 0.05646201202791944} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, asym lic, newpibins
# newdesign = {'M_cut': 10**13.555465938406698, 
#              'M1': 10**14.360545573957374, 
#              'sigma': 1.2453263473232414, 
#              'alpha': 1.0962691193504375,
#              'kappa': 0.6492093749864681}    
# newdecor = {'s': -0.3138959521390018, 
#             's_v': -0.06872727043440933, 
#             'alpha_c': 0.1341073400184064, 
#             's_p': -1.0, 
#             's_r': 0,
#             'A': -0.6296830143757656,
#             'Ae': 0.058417762251943} 

# # CMAES fits on wpmultipole, all hod params, linear A, linear fenv, 5mpc, avgn, cov400
# newdesign = {'M_cut': 10**13.496729397890908, 
#              'M1': 10**14.41053674056244, 
#              'sigma': 1.0991529779447204, 
#              'alpha': 0.9484214219386737,
#              'kappa': 0.5843733140601354}    
# newdecor = {'s': 0.20275411021510312, 
#             's_v': 0.4307648150196516, 
#             'alpha_c': 0.11973147137120776, 
#             's_p': -1.0, 
#             's_r': 0,
#             'A': -0.5416693464056683,
#             'Ae': 0.042838720388859004} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, sigmoid
# newdesign = {'M_cut': 10**13.483618550697493, 
#              'M1': 10**14.548590503867032, 
#              'sigma': 0.8146271233109161, 
#              'alpha': 1.3,
#              'kappa': 10**(-0.7690121304215453)}    
# newdecor = {'s': -0.07705240702511965, 
#             's_v': 0.7196607817814157, 
#             'alpha_c': 0.24720239083690032, 
#             's_p': -0.9968024356393409, 
#             's_r': 0,
#             'A': -0.9809510747836853,
#             'Ae': 0.03558816320277228} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, sigmoid, newkappa, reseed 390
# newdesign = {'M_cut': 10**13.37295046948863, 
#              'M1': 10**14.42557838944547, 
#              'sigma': 0.6691727094227253, 
#              'alpha': 1.2264992336728515,
#              'kappa': 10**(-0.07943146839094262)}    
# newdecor = {'s': -0.41496111471094327, 
#             's_v': 0.5685822884156442, 
#             'alpha_c': 0.24507967630006933, 
#             's_p': -0.8363034069150131, 
#             's_r': 0,
#             'A': -0.9748704242122554,
#             'Ae': 0.04693302624320874} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, newkappa
# newdesign = {'M_cut': 10**13.323825564795841, 
#              'M1': 10**14.468782971172098, 
#              'sigma': 0.593615892944289, 
#              'alpha': 1.2989654768088204,
#              'kappa': 10**(-0.8602513373632192)}    
# newdecor = {'s': 0.11299732886922442, 
#             's_v': 0.8953358075047386, 
#             'alpha_c': 0.20655487923712668, 
#             's_p': -0.9717945498639274, 
#             's_r': 0,
#             'A': -0.8338625760010008,
#             'Ae': 0.03784569098773295} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, newkappa, 1
# newdesign = {'M_cut': 10**13.417644048551379, 
#              'M1': 10**14.455287784107973, 
#              'sigma': 0.8507385946230487, 
#              'alpha': 1.2034093829541428,
#              'kappa': 10**(-0.4432477905934858)}    
# newdecor = {'s': 0.039898941201963606, 
#             's_v': 0.7040176052574895, 
#             'alpha_c': 0.17306686941070804, 
#             's_p': -0.9998026471447804, 
#             's_r': 0,
#             'A': -0.8020759075348068,
#             'Ae': 0.049493023884225995} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, newkappa, 1, new, correct kappa
# newdesign = {'M_cut': 10**13.338944893190124, 
#              'M1': 10**14.476950752907403, 
#              'sigma': 0.6103037164790116, 
#              'alpha': 1.3174966267089432,
#              'kappa': 0.2149153025034436}    
# newdecor = {'s': 0.08816917027516914, 
#             's_v': 0.8583636205517335, 
#             'alpha_c': 0.21631398863964169, 
#             's_p': -1.0, 
#             's_r': 0,
#             'A': -0.8781409157999217,
#             'Ae': 0.040333674435968074} 
# # chi2 = 64.06076020671918
# # this is the one we are using as standard..... this following one is the same, but without Ae
# newdesign = {'M_cut': 10**13.354235963825529, 
#              'M1': 10**14.517920884968769, 
#              'sigma': 0.5237031356823908, 
#              'alpha': 1.398690944682553,
#              'kappa': 0.1166445042334662}    
# newdecor = {'s': -0.022823272673439317, 
#             's_v': 0.6161775729951983, 
#             'alpha_c': 0.2639738674015627, 
#             's_p': -0.9375555566107862, 
#             's_r': 0,
#             'A': -0.7897003296636026,
#             'Ae': 0} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, newkappa, reseed 234
# newdesign = {'M_cut': 10**13.389589286465174, 
#              'M1': 10**14.490877989276152, 
#              'sigma': 0.7305454647756873, 
#              'alpha': 1.3415203291419073,
#              'kappa': 10**(-0.9517193687926441)}    
# newdecor = {'s': 0.05934045653169603, 
#             's_v': 0.6693308223008227, 
#             'alpha_c': 0.23278342684534992, 
#             's_p': -0.9984464848940741, 
#             's_r': 0,
#             'A': -0.833824104932075,
#             'Ae': 0.036102689945680805} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, newkappa, totcov
# newdesign = {'M_cut': 10**13.33964176510646, 
#              'M1': 10**14.441998522941187, 
#              'sigma': 0.6008612253603542, 
#              'alpha': 1.322023132165448,
#              'kappa': 10**(-0.18154736395822857)}    
# newdecor = {'s': -0.20452090106520165, 
#             's_v': 0.4698617030970368, 
#             'alpha_c': 0.24938587768778908, 
#             's_p': -1.0, 
#             's_r': 0,
#             'A': -0.8836901534689072,
#             'Ae': 0.03267965603608849}

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, newkappa, epsilon instead of sigma
# newdesign = {'M_cut': 10**13.29794143704705, 
#              'M1': 10**14.3418836022415, 
#              'sigma': 0.4813494870207162, 
#              'alpha': 1.2561838176491578,
#              'kappa': 2.1847471590129675}    
# newdecor = {'s': -0.8273950737173078, 
#             's_v': -0.1595759658216517, 
#             'alpha_c': 0.28031572664865423, 
#             's_p': -0.7711789080638323, 
#             's_r': 0,
#             'A': -0.9821177693630414,
#             'Ae': 0.035743141945415106}

# # CMAES fits on xi2xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, newkappa, xi2xi
# newdesign = {'M_cut': 10**13.43725115903052, 
#              'M1': 10**14.535419030383952, 
#              'sigma': 0.8279681063137849, 
#              'alpha': 1.318902497082193,
#              'kappa': 0.10788610616883626}    
# newdecor = {'s': -0.9000154389211223, 
#             's_v': 0.22199292333422183, 
#             'alpha_c': 0.19199304422219832, 
#             's_p': 0.29909137373757505, 
#             's_r': 0,
#             'A': -0.9232183764105382,
#             'Ae': 0.0551847080029011} 

# # # CMAES fits on xi2xi4xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, newkappa, xi2xi4xi
# newdesign = {'M_cut': 10**13.644450758204961, 
#              'M1': 10**14.429872163485793, 
#              'sigma': 1.371139500445369, 
#              'alpha': 0.9726151898616708,
#              'kappa': 0.24591927019124707}    
# newdecor = {'s': 0.0825284361798616, 
#             's_v': 0.26144602825763574, 
#             'alpha_c': 0.0416235915729668, 
#             's_p': -0.9959313598701458, 
#             's_r': 0,
#             'A': -0.4542800941218707,
#             'Ae': 0.05168983898016036} 

# # # CMAES fits on xismu, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, 
# newdesign = {'M_cut': 10**13.514103911905108, 
#              'M1': 10**14.445206125977752, 
#              'sigma': 1.1125580523817935, 
#              'alpha': 1.1399037803672243,
#              'kappa': 0.2803357755123174}    
# newdecor = {'s': -0.0636029128925964, 
#             's_v': 0.07114938870993782, 
#             'alpha_c': 0.1339911756562896, 
#             's_p': -0.40034520495909426, 
#             's_r': 0,
#             'A': -0.6105567853515889,
#             'Ae': 0.04771056263423072}

# # wp only fit, vanilla hod + s + Ae
# newdesign = {'M_cut': 10**13.535799509507724, 
#              'M1': 10**14.430735413758939, 
#              'sigma': 1.1197876489277627, 
#              'alpha': 1.0630257386551396,
#              'kappa': 0.0433317820555811}    
# newdecor = {'s': -0.5748053352108813, 
#             's_v': 0, 
#             'alpha_c': 0, 
#             's_p':  0, 
#             's_r': 0,
#             'A':  0,
#             'Ae': 0.027885900996989503} 

# # wp only fit, vanilla hod + s + A
# newdesign = {'M_cut': 10**13.328068037356093, 
#              'M1': 10**14.45169303152465, 
#              'sigma': 0.471812055989258463, 
#              'alpha': 1.460408362695425,
#              'kappa': 0.022411414460887184}    
# newdecor = {'s': -0.6166114190055553, 
#             's_v': 0, 
#             'alpha_c': 0, 
#             's_p':  0, 
#             's_r': 0,
#             'A':  0.21227906506120525,
#             'Ae': 0} 

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, newkappa, no Ae
# newdesign = {'M_cut': 10**13.353624482867241, 
#              'M1': 10**14.517944261274222, 
#              'sigma': 0.5215065297787211, 
#              'alpha': 1.3939029828033735,
#              'kappa': 0.1332020258208893}    
# newdecor = {'s': -0.011268399448813865, 
#             's_v': 0.6406828102916402, 
#             'alpha_c': 0.2638259634168132, 
#             's_p': -0.9421859444644138, 
#             's_r': 0,
#             'A': -0.7914429755805419,
#             'Ae': 0} 

'''
# CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, newkappa, 1, new, no A
newdesign = {'M_cut': 10**13.373048727632613, 
             'M1': 10**14.329856115607837, 
             'sigma': 0.9382440724135253, 
             'alpha': 1.013656460030536,
             'kappa': 0.19709980972635074}    
newdecor = {'s': 0.27253604440055584, 
            's_v': 0.14423685141919557, 
            'alpha_c': 0.06936489117335409, 
            's_p': -1.0, 
            's_r': 0,
            'A': 0,
            'Ae': 0.03172111555716747} 

'''
# BORYANA B.H.

newdesign = {'M_cut': 10**13.496729397890908, 
             'M1': 10**14.41053674056244, 
             'sigma': 1.0991529779447204, 
             'alpha': 0.9484214219386737,
             'kappa': 0.5843733140601354}    
newdecor = {'s': 0.20275411021510312, 
            's_v': 0.4307648150196516, 
            'alpha_c': 0.11973147137120776, 
            's_p': -1.0, 
            's_r': 0,
            'A': -0.5416693464056683,
            'Ae': 0.042838720388859004,
            'ic': 1.}  

# # CMAES fits on xi, all hod params, linear A, linear fenv, 5mpc, avgn, cov400, fullxi, Lic1, newkappa, 1, new, no A Ae
# newdesign = {'M_cut': 10**13.162899793962989, 
#              'M1': 10**14.337773116790009, 
#              'sigma': 0.1128558932473106, 
#              'alpha': 1.15954612111661,
#              'kappa': 0.23723758483677634}    
# newdecor = {'s': 0.5803392366095864, 
#             's_v': 0.5176205537245805, 
#             'alpha_c': 0.20955124348375032, 
#             's_p': -1.0, 
#             's_r': 0,
#             'A': 0,
#             'Ae': 0} 

decorator = "_xifit_linearA_fenv_avgn_cov400_newkappa_correct_noA"
# decorator = "_xifit_linearA_fenv_avgn_cov400_newkappa_reseed4"
# decorator = "_compare_vanilla"
# decorator = "_wponlyfit_vanilla_s_A"

pimax = 30.0 # h-1 mpc
pi_bin_size = 5
ximin = 0.02
ximax = 100

def calc_xi_mock_natural(whichsim, design, decorations, rpbins, pibins, params, newseed, rsd = rsd):

    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(decorations.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    # data directory
    scratchdir = "/data_m200b_mod_new"
    cdatadir = "/mnt/store/boryanah/scratch" + scratchdir
    if rsd:
        cdatadir = cdatadir+"_rsd"
    savedir = cdatadir+"/rockstar_"\
    +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
    +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
    if rsd:
        savedir = savedir+"_rsd"
    if not newseed == 0:
        savedir = savedir+"_"+str(newseed)
    # if os.path.exists(savedir+"/data_xirppi_natural_"+str(whichsim)+".npz"):
    #     return 0

    # these are the pre fc mocks
    filename_cent = savedir+"/halos_gal_cent_"+str(whichsim)
    filename_sat = savedir+"/halos_gal_sats_"+str(whichsim)

    # read in the galaxy catalog
    fcent = np.fromfile(filename_cent)
    fsats = np.fromfile(filename_sat)

    # reshape the file data
    fcent = np.array(np.reshape(fcent, (-1, 9)))
    fsats = np.array(np.reshape(fsats, (-1, 9)))

    pos_cent = fcent[:,0:3]
    pos_sats = fsats[:,0:3]

    # full galaxy catalog
    pos_full = np.concatenate((pos_cent, pos_sats))
    ND = float(len(pos_full))

    # convert to h-1 mpc
    pos_full = pos_full * params['h'] % params['Lboxh']

    pi_bin_size = int(pibins[2] - pibins[1])
    pimax = pibins[-1]

    DD_counts = DDrppi(1, 4, int(pimax), rpbins,
        pos_full[:, 0], pos_full[:, 1], pos_full[:, 2], boxsize = params['Lboxh'], periodic = True)['npairs']
    DD_counts_new = np.array([np.sum(DD_counts[i:i+pi_bin_size]) for i in range(0, len(DD_counts), pi_bin_size)])
    DD_counts_new = DD_counts_new.reshape((len(rpbins) - 1, int(pimax/pi_bin_size)))

    # now calculate the RR count from theory
    RR_counts_new = np.zeros((len(rpbins) - 1, int(pimax/pi_bin_size)))
    for i in range(len(rpbins) - 1):
        RR_counts_new[i] = np.pi*(rpbins[i+1]**2 - rpbins[i]**2)*pi_bin_size / params['Lboxh']**3 * ND**2 * 2
    xirppi_reshaped = DD_counts_new / RR_counts_new - 1
    # print("cf xi done, time spent : ", time.time() - start)
    fname = savedir+"/data_xirppi_natural_"+str(whichsim)
    np.savez(fname, rbins = rpbins, pimax = pimax, xi = xirppi_reshaped, DD = DD_counts, ND = ND)
    return xirppi_reshaped.flatten() # 1d array

    # # compute with halotools
    # xi = mo.rp_pi_tpcf(pos_full, rpbins, pibins, period = params['Lboxh'])
    # # print(xi)
    # # print("ht xi done, time spent : ", time.time() - start)
    # fname = savedir+"/data_xirppi_natural_"+str(whichsim)
    # np.savez(fname, rbins = rpbins, pibins = pibins, xi = xi, ND = ND)
    # print(xi)
    # return xi.flatten() # 1d array


def calc_xi2xi_persim(whichsim, design, decorations, rbins, params, newseed, rsd = rsd):

    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(decorations.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    # data directory
    scratchdir = "/data_m200b_mod_new"
    cdatadir = "/mnt/store/boryanah/scratch" + scratchdir
    if rsd:
        cdatadir = cdatadir+"_rsd"
    savedir = cdatadir+"/rockstar_"\
    +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
    +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
    if rsd:
        savedir = savedir+"_rsd"
    if not newseed == 0:
        savedir = savedir+"_"+str(newseed)

    filename_cent = savedir+"/halos_gal_cent_"+str(whichsim)
    filename_sat = savedir+"/halos_gal_sats_"+str(whichsim)

    # read in the galaxy catalog
    fcent = np.fromfile(filename_cent)
    fsats = np.fromfile(filename_sat)

    # reshape the file data
    fcent = np.array(np.reshape(fcent, (-1, 9)))
    fsats = np.array(np.reshape(fsats, (-1, 9)))

    pos_cent = fcent[:,0:3]
    pos_sats = fsats[:,0:3]

    # full galaxy catalog
    pos_full = np.concatenate((pos_cent, pos_sats))
    ND = float(len(pos_full))

    # convert to h-1 mpc
    pos_full = pos_full * params['h'] % params['Lboxh']

    # calculate multipoles
    mu_bins = np.linspace(0, 1, 20)
    xi_s_mu = s_mu_tpcf(pos_full, rbins, mu_bins, period=params['Lboxh'])

    xi2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
    xi4 = tpcf_multipole(xi_s_mu, mu_bins, order=4)

    # calc xi rp pi
    DD_counts = DDrppi(1, 1, pimax, rbins,
        pos_full[:, 0], pos_full[:, 1], pos_full[:, 2], boxsize = params['Lboxh'], periodic = True)['npairs']
    DD_counts_new = np.array([np.sum(DD_counts[i:i+pi_bin_size]) for i in range(0, len(DD_counts), pi_bin_size)])
    DD_counts_new = DD_counts_new.reshape((len(rbins) - 1, int(pimax/pi_bin_size)))

    # now calculate the RR count from theory
    RR_counts_new = np.zeros((len(rbins) - 1, int(pimax/pi_bin_size)))
    for i in range(len(rbins) - 1):
        RR_counts_new[i] = np.pi*(rbins[i+1]**2 - rbins[i]**2)*pi_bin_size / params['Lboxh']**3 * ND**2 * 2
    xirppi_reshaped = DD_counts_new / RR_counts_new - 1

    fname = savedir+"/data_xi2xi_natural_"+str(whichsim)
    np.savez(fname, rbins = rbins, pimax = pimax, xi = xirppi_reshaped, DD = DD_counts, xi2 = xi2, xi4 = xi4, ND = ND)

    return xirppi_reshaped, xi2, xi4


def calc_xir(whichsim, design, decorations, rbins, params, newseed, rsd = rsd):

    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(decorations.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    # data directory
    scratchdir = "/data_m200b_mod_new"
    cdatadir = "/mnt/store/boryanah/scratch" + scratchdir
    if rsd:
        cdatadir = cdatadir+"_rsd"
    savedir = cdatadir+"/rockstar_"\
    +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
    +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
    if rsd:
        savedir = savedir+"_rsd"
    if not newseed == 0:
        savedir = savedir+"_"+str(newseed)

    # these are the pre fc mocks
    filename_cent = savedir+"/halos_gal_cent_"+str(whichsim)
    filename_sat = savedir+"/halos_gal_sats_"+str(whichsim)

    # read in the galaxy catalog
    fcent = np.fromfile(filename_cent)
    fsats = np.fromfile(filename_sat)

    # reshape the file data
    fcent = np.array(np.reshape(fcent, (-1, 9)))
    fsats = np.array(np.reshape(fsats, (-1, 9)))

    pos_cent = fcent[:, 0:3] * params['h'] % params['Lboxh']
    pos_sats = fsats[:, 0:3] * params['h'] % params['Lboxh']

    # full galaxy catalog
    # pos_full = np.concatenate((pos_cent, pos_sats)) * params['h'] % params['Lboxh']
    pos_full = np.concatenate((pos_cent, pos_sats))

    results = xi(params['Lboxh'], 1, rbins, pos_full[:, 0], pos_full[:, 1], pos_full[:, 2])
    newxir = np.array([row[3] for row in results])

    results_cent = xi(params['Lboxh'], 1, rbins, pos_cent[:, 0], pos_cent[:, 1], pos_cent[:, 2])
    newxir_cent = np.array([row[3] for row in results_cent])

    results_sats = xi(params['Lboxh'], 1, rbins, pos_sats[:, 0], pos_sats[:, 1], pos_sats[:, 2])
    newxir_sats = np.array([row[3] for row in results_sats])

    fname = savedir+"/data_xir_"+str(whichsim)
    np.savez(fname, rbins = rbins, xir = newxir, xir_cent = newxir_cent, xir_sats = newxir_sats)


def calc_wp(whichsim, design, decorations, rpbins, params, newseed, rsd = rsd):

    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(decorations.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    # data directory
    scratchdir = "/data_m200b_mod_new"
    cdatadir = "/mnt/store/boryanah/scratch" + scratchdir
    if rsd:
        cdatadir = cdatadir+"_rsd"
    savedir = cdatadir+"/rockstar_"\
    +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
    +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
    if rsd:
        savedir = savedir+"_rsd"
    if not newseed == 0:
        savedir = savedir+"_"+str(newseed)

    # these are the pre fc mocks
    filename_cent = savedir+"/halos_gal_cent_"+str(whichsim)
    filename_sat = savedir+"/halos_gal_sats_"+str(whichsim)

    # read in the galaxy catalog
    fcent = np.fromfile(filename_cent)
    fsats = np.fromfile(filename_sat)

    # reshape the file data
    fcent = np.array(np.reshape(fcent, (-1, 9)))
    fsats = np.array(np.reshape(fsats, (-1, 9)))

    pos_cent = fcent[:,0:3] * params['h'] % params['Lboxh']
    pos_sats = fsats[:,0:3] * params['h'] % params['Lboxh']
    print(len(pos_cent), len(pos_sats))

    # full galaxy catalog
    # pos_full = np.concatenate((pos_cent, pos_sats)) * params['h'] % params['Lboxh']
    pos_full = np.concatenate((pos_cent, pos_sats))

    xs = pos_full[:,0]
    ys = pos_full[:,1]
    zs = pos_full[:,2]

    wp_results = wp(params['Lboxh'], pimax, 1, rpbins, xs, ys, zs, 
        verbose=False, output_rpavg=False) # this is all done in mpc / h
    newwp = np.array([row[3] for row in wp_results])

    fname = savedir+"/data_wp_"+str(whichsim)
    np.savez(fname, rbins = rpbins, pimax = pimax, wp = newwp)

    return newwp

def plot_xi(design, decorations, rpbins, pibins, params, newseeds, rsd = rsd):

    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(decorations.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    xi_sum = 0
    xi2_sum = 0
    for eseed in newseeds:
        # data directory
        scratchdir = "/data_m200b_mod_new"
        cdatadir = "/mnt/store/boryanah/scratch" + scratchdir
        if rsd:
            cdatadir = cdatadir+"_rsd"
        savedir = cdatadir+"/rockstar_"\
        +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
        +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
        if rsd:
            savedir = savedir+"_rsd"
        if not eseed == 0:
            savedir = savedir+"_"+str(eseed)

        for whichsim in range(params['num_sims']):
            fname = savedir+"/home/syuan/s3PCF_fenv/data_xirppi_natural_"+str(whichsim)
            newxi = np.load(fname+".npz")['xi']
            xi_sum += newxi
            xi2_sum += newxi**2

    xi_avg = xi_sum/params['num_sims']/len(newseeds)
    xi2_avg = xi2_sum/params['num_sims']/len(newseeds)

    xi_std = np.sqrt((xi2_avg - xi_avg**2)/params['num_sims'])/len(newseeds)

    print("avg s/n : ", np.mean(abs(xi_avg/xi_std)))
    print(xi_avg)

    fig = pl.figure(figsize = (5, 4))
    pim = 30
    pl.imshow(xi_avg.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = cm.viridis, norm=colors.LogNorm(vmin = ximin, vmax = ximax))
    cbar = pl.colorbar()
    cbar.set_label('$\\xi(r_\perp, \pi)$', rotation = 270, labelpad = 20)
    # pl.xscale('log')
    pl.xlabel('$\log r_\perp$ ($h^{-1}$Mpc)')
    pl.ylabel('$\pi$ ($h^{-1}$Mpc)')
    # pl.yticks(np.linspace(0, 30, 7), (0, 0.5, 1, 5, 10, 20, 30))
    pl.tight_layout()
    plotname = "plots/plot_xirppi_mock_reseeded"+decorator
    fig.savefig(plotname+".pdf", dpi = 300)

def plot_xir(design, decorations, rs, params, newseeds, rsd = rsd):


    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(decorations.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    xir_sum = 0
    xir_cent_sum = 0
    xir_sats_sum = 0
    for eseed in newseeds:
        # data directory
        scratchdir = "/data_m200b_mod_new"
        cdatadir = "/mnt/store/boryanah/scratch" + scratchdir
        if rsd:
            cdatadir = cdatadir+"_rsd"
        savedir = cdatadir+"/rockstar_"\
        +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
        +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
        if rsd:
            savedir = savedir+"_rsd"
        if not eseed == 0:
            savedir = savedir+"_"+str(eseed)

        for whichsim in range(params['num_sims']):
            fname = savedir+"/home/syuan/s3PCF_fenv/data_xir_"+str(whichsim)
            newdata = np.load(fname+".npz")
            xir_sum += newdata['xir']
            xir_cent_sum += newdata['xir_cent']
            xir_sats_sum += newdata['xir_sats']

    xir_avg = xir_sum/params['num_sims']/len(newseeds)
    xir_cent_avg = xir_cent_sum/params['num_sims']/len(newseeds)
    xir_sats_avg = xir_sats_sum/params['num_sims']/len(newseeds)

    np.savez("data/data_xir_avg", xir = xir_avg, xir_cent = xir_cent_avg, xir_sats = xir_sats_avg, 
        rs = rs)
    print(xir_avg)

def plot_wp(design, decorations, rps, params, newseeds, rsd = rsd):

    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(decorations.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    wp_sum = 0
    for eseed in newseeds:
        # data directory
        scratchdir = "/data_m200b_mod_new"
        cdatadir = "/mnt/store/boryanah/scratch" + scratchdir
        if rsd:
            cdatadir = cdatadir+"_rsd"
        savedir = cdatadir+"/rockstar_"\
        +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
        +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
        if rsd:
            savedir = savedir+"_rsd"
        if not eseed == 0:
            savedir = savedir+"_"+str(eseed)

        for whichsim in range(params['num_sims']):
            fname = savedir+"/data_wp_"+str(whichsim)
            newwp = np.load(fname+".npz")['wp']
            wp_sum += newwp

    wp_avg = wp_sum/params['num_sims']/len(newseeds)

    # hong's data
    hong_wp_data = np.loadtxt("/home/syuan/s3PCF_fenv/hong_data_final/wp_cmass_final_finebins_z0.46-0.60")
    wp_hong = hong_wp_data[:, 1]
    rwp_hong = rps * wp_hong  # (h-1 mpc)^2

    # covariance
    hong_wp_covmat = np.loadtxt("/home/syuan/s3PCF_fenv/hong_data_final/wpcov_cmass_final_finebins_z0.46-0.60")
    hong_rwp_covmat = np.zeros(np.shape(hong_wp_covmat))
    for i in range(np.shape(hong_wp_covmat)[0]):
        for j in range(np.shape(hong_wp_covmat)[1]):
            hong_rwp_covmat[i, j] = hong_wp_covmat[i, j]*rps[i]*rps[j]
    hong_rwp_covmat_inv = np.linalg.inv(hong_rwp_covmat)
    hong_rwp_covmat_inv_short = np.linalg.inv(hong_rwp_covmat)[2:, 2:]
    rwp_hong_err = 1/np.sqrt(np.diag(hong_rwp_covmat_inv))

    fig = pl.figure(figsize=(8.5, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1]) 

    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlabel('$r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax1.set_ylabel('$r_p w_p$ ($h^{-1} \mathrm{Mpc})^2$')
    ax1.errorbar(rps[2:], rwp_hong[2:], yerr = rwp_hong_err[2:], label = 'observed')
    ax1.plot(rps[2:], wp_avg[2:]*rps[2:], label = 'mock')
    ax1.set_xscale('log')
    ax1.set_xlim(0.1, 50)
    ax1.legend(loc='best', prop={'size': 13})

    delta_rwp = (wp_avg * rps - rwp_hong)[2:]
    chi2s = delta_rwp * np.dot(hong_rwp_covmat_inv_short, delta_rwp)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlabel('$r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$\chi^2$')
    ax2.set_xscale('log')
    ax2.plot(rps[2:], chi2s, 'r-o', label = "$X^2 = $"+str(np.sum(chi2s))[:6])
    ax2.set_xlim(0.1, 50)
    ax2.legend(loc='best', prop={'size': 13})

    pl.tight_layout()
    fig.savefig("plots/plot_wp_reseeded"+decorator+".pdf", dpi=720)
    np.savez("data/data_wp_reseeded"+decorator, wp = wp_avg, rp = rps, rwp_hong = rwp_hong, rwp_hong_err = rwp_hong_err)

def compare_to_boss(design, decorations, rpbins, params, newseeds, rsd = rsd):

    # load mock
    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(decorations.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    xi_sum = 0
    xi2_sum = 0
    for eseed in newseeds:
        # data directory
        scratchdir = "/data_m200b_mod_new"
        cdatadir = "/mnt/store/boryanah/scratch" + scratchdir
        if rsd:
            cdatadir = cdatadir+"_rsd"
        savedir = cdatadir+"/rockstar_"\
        +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
        +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
        if rsd:
            savedir = savedir+"_rsd"
        if not eseed == 0:
            savedir = savedir+"_"+str(eseed)

        for whichsim in range(params['num_sims']):
            fname = savedir+"/data_xirppi_natural_"+str(whichsim)
            newxi = np.load(fname+".npz")['xi']
            print(np.shape(newxi))
            xi_sum += newxi
            xi2_sum += newxi**2

    xi_avg = xi_sum/params['num_sims']/len(newseeds)
    print("mock xi ", xi_avg)
    xi2_avg = xi2_sum/params['num_sims']/len(newseeds)

    # covariance matrix 
    xicov = np.load("/home/syuan/s3PCF_fenv/data/data_xi_cov400_norm.npz")['xicovnorm']
    xicov_inv = np.linalg.inv(xicov) # boss cov
    xi_errs = np.sqrt(np.load("/home/syuan/s3PCF_fenv/data/data_xi_cov400_norm.npz")['diag'])

    # load boss
    # xi_boss = np.loadtxt("./hong_data_final/xip_cmass_final_coarsebins_z0.46-0.60_newpibin")[:, 2].reshape(8, 6)
    xi_boss = np.loadtxt("/home/syuan/s3PCF_fenv/hong_data_final/xip_cmass_final_coarsebins_z0.46-0.60")
    delta_xi = xi_avg - xi_boss
    print("boss xi ", xi_boss)
    print("delta xi / xi ", delta_xi / xi_boss)
    # print(np.shape(xi_avg), np.shape(xi_boss))
    delta_xi_norm = (xi_avg - xi_boss)/xi_errs.reshape(np.shape(xi_boss))
    # delta_xi[0] = 0
    # delta_xi_norm[0] = 0

    zmin2 = -10 # np.min(delta_xi) # -3
    zmax2 = 10 # np.max(delta_xi) # 9.5
    mycmap2 = cm.get_cmap('bwr')
    pim = 30

    # # (xi_mock - xi_boss) / err
    # fig = pl.figure(figsize=(5, 4))
    # pl.imshow(delta_xi_norm.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
    #     extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
    #     cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=zmin2, vmax=zmax2))
    # cbar = pl.colorbar()
    # cbar.set_label('$(\\xi_{\\rm{mock}}-\\xi_{\\rm{BOSS}})/\sigma(\\xi)$', rotation = 270, labelpad = 20)
    # # pl.xscale('log')
    # pl.xlabel('$\log r_\perp$ ($h^{-1}$Mpc)')
    # pl.ylabel('$\pi$ ($h^{-1}$Mpc)')
    # pl.tight_layout()
    # plotname = "./plots/plot_delta_xirppi_mock_boss"+decorator
    # fig.savefig(plotname+".pdf", dpi = 300)

    # (xi_mock - xi_boss) / xi_mock
    fig = pl.figure(figsize=(5, 4))
    pim = 30
    delta_xi_byxi = delta_xi / xi_avg
    pl.imshow(delta_xi_byxi.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=-0.25, vmax=0.25))
    cbar = pl.colorbar()
    cbar.set_label('$(\\xi_{\\rm{mock}}-\\xi_{\\rm{BOSS}})/\\xi_{\\rm{mock}}$', rotation = 270, labelpad = 20)
    # pl.xscale('log')
    pl.xlabel('$\log r_\perp$ ($h^{-1}$Mpc)')
    pl.ylabel('$\pi$ ($h^{-1}$Mpc)')
    pl.yticks(np.linspace(0, 30, 7), pibins)
    pl.tight_layout()
    plotname = "plots/plot_delta_xirppi_mock_boss_byxi"+decorator
    fig.savefig(plotname+".pdf", dpi = 300)

    # make a triple plot, xi, delta xi, chi2
    fig = pl.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(ncols = 3, nrows = 2, width_ratios = [1, 1, 1], height_ratios = [1, 12]) 

    # plot 1
    pibins[0] = 0
    ax1 = fig.add_subplot(gs[3])
    ax1.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax1.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col1 = ax1.imshow(xi_avg.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = cm.viridis, norm=colors.LogNorm(vmin = 0.01, vmax = 30))
    ax1.set_yticks(np.linspace(0, 30, 7))
    ax1.set_yticklabels(pibins)

    ax0 = fig.add_subplot(gs[0])
    cbar = pl.colorbar(col1, cax = ax0, orientation="horizontal")
    cbar.set_label('$\\xi(r_p, \pi)$', labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')

    # plot 2
    ax2 = fig.add_subplot(gs[4])
    ax2.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col2 = ax2.imshow(delta_xi_norm.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=zmin2, vmax=zmax2))
    ax2.set_yticks(np.linspace(0, 30, 7))
    ax2.set_yticklabels(pibins)

    ax3 = fig.add_subplot(gs[1])
    cbar = pl.colorbar(col2, cax = ax3, orientation="horizontal", ticks = [-10, -5, 0, 5, 10])
    cbar.set_label("$(\\xi_{\\rm{mock}}-\\xi_{\\rm{BOSS}})/\sigma(\\xi)$", labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')
    # cbar.set_ticks(np.linspace(-1, 1, num = 5))

    # plot 3
    chi2s = (delta_xi_norm.flatten() * np.dot(xicov_inv, delta_xi_norm.flatten())).reshape(np.shape(delta_xi))
    print(chi2s, np.sum(chi2s))
    ax2 = fig.add_subplot(gs[5])
    ax2.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col2 = ax2.imshow(chi2s.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=-100, vmax=100))
    ax2.set_yticks(np.linspace(0, 30, 7))
    ax2.set_yticklabels(pibins)

    ax3 = fig.add_subplot(gs[2])
    cbar = pl.colorbar(col2, cax = ax3, orientation="horizontal")
    cbar.set_label("$\chi^2$", labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')
    # cbar.set_ticks(np.linspace(-1, 1, num = 5))

    pl.subplots_adjust(wspace=20)
    pl.tight_layout()
    fig.savefig("plots/plot_xi_mock_diff_2plot"+decorator+".pdf", dpi=720)


    # make a triple plot, wp, xi comparison, chi2 
    wpdata = np.load("/home/syuan/s3PCF_fenv/data/data_wp_reseeded"+decorator+".npz")
    wp_avg, rps, rwp_hong, rwp_hong_err = wpdata['wp'], wpdata['rp'], wpdata['rwp_hong'], wpdata['rwp_hong_err']

    fig = pl.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(ncols = 3, nrows = 1, width_ratios = [1, 1, 1]) 
    gs.update(wspace=0.3, hspace=0.1)

    # plot 1
    pibins[0] = 0
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax1.set_ylabel('$r_p w_p$ ($h^{-1} \mathrm{Mpc})^2$')
    ax1.errorbar(np.log10(rps[2:]), rwp_hong[2:], yerr = rwp_hong_err[2:], label = 'observed')
    ax1.plot(np.log10(rps[2:]), wp_avg[2:]*rps[2:], label = 'mock')
    # ax1.set_xscale('log')
    # ax1.set_xlim(0.1, 50)
    ax1.legend(loc='best', prop={'size': 13})

    # plot 2
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios = [1, 12], subplot_spec=gs[1], wspace=0.1)
    ax2 = fig.add_subplot(gs00[1])
    ax2.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col2 = ax2.imshow(delta_xi_norm.T*np.sqrt(7/9), interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=zmin2, vmax=zmax2))
    ax2.set_yticks(np.linspace(0, 30, 7))
    ax2.set_yticklabels(pibins)

    ax3 = fig.add_subplot(gs00[0])
    cbar = pl.colorbar(col2, cax = ax3, orientation="horizontal", ticks = [-10, -5, 0, 5, 10])
    cbar.set_label("$(\\xi_{\\rm{mock}}-\\xi_{\\rm{BOSS}})/\sigma(\\xi)$", labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')
    # cbar.set_ticks(np.linspace(-1, 1, num = 5))

    # plot 3
    chi2s = (delta_xi_norm.flatten() * np.dot(xicov_inv, delta_xi_norm.flatten())).reshape(np.shape(delta_xi))
    print(chi2s, np.sum(chi2s))
    gs11 = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios = [1, 12], subplot_spec=gs[2], wspace=0.1)
    ax2 = fig.add_subplot(gs11[1])
    ax2.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col2 = ax2.imshow(chi2s.T*7/9, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=-100, vmax=100))
    ax2.set_yticks(np.linspace(0, 30, 7))
    ax2.set_yticklabels(pibins)

    ax3 = fig.add_subplot(gs11[0])
    cbar = pl.colorbar(col2, cax = ax3, orientation="horizontal")
    cbar.set_label("$\chi^2$", labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')
    # cbar.set_ticks(np.linspace(-1, 1, num = 5))

    pl.subplots_adjust(wspace=20)
    pl.tight_layout()
    fig.savefig("plots/plot_xiwpchi2"+decorator+".pdf", dpi=720)

    print(chi2s.T*7/9)


if __name__ == "__main__":

    # load saito bins
    hong_wp_data = np.loadtxt("/home/syuan/s3PCF_fenv/hong_data/wp_cmass_z0.46-0.6")
    rp_saito = hong_wp_data[:,0] # h-1 mpc
    num_bins = len(rp_saito)
    rp_saito_log = np.log10(rp_saito)
    delta_rp_saito = 0.125 # h-1 mpc
    rp_bins_log = np.linspace(np.min(rp_saito_log) - delta_rp_saito/2, 
                              np.max(rp_saito_log) + delta_rp_saito/2, num_bins + 1)
    rp_bins = 10**rp_bins_log # h-1 mpc

    # courser bins 
    rp_bins_log_course = np.linspace(np.min(rp_bins_log), 
                              np.max(rp_bins_log), 8 + 1)
    rp_bins_course = 10**rp_bins_log_course # h-1 mpc

    # pibins = np.array([1e-5, 0.5, 1.0, 5, 10, 20, 30])
    pibins = np.array([1e-5, 5, 10, 15, 20, 25, 30])

    # rp_bins_course = np.logspace(-2, 1.5, 50)
    # pibins = np.linspace(0, 30, 31)
    # pibins[0] = 1e-5

    # newseeds = np.array([245, 263, 164, 679, 230, 945, 206, 390, 786,  82, 522, 117, 703,
    #    490,  80,  75, 918,  71, 799, 804, 392, 800, 978, 608, 542, 187,
    #    848, 917, 100, 723, 894,  20, 548, 464, 897, 818, 269, 220, 252,
    #    389, 605, 889, 808,  21, 881, 805, 211, 406,  15, 511, 306, 304,
    #    998, 813, 884, 913, 756, 847, 950, 497, 253, 981, 457, 335, 720,
    #    269, 878, 292, 640, 549, 658, 564, 976, 891, 694, 335,  58, 883,
    #    670, 325, 816,  99, 123, 300, 780, 711,   4, 988,   9, 328, 168,
    #    351, 478, 386, 190,  51, 333, 818, 792, 811])
    newseeds = [0]
    def run_onebox(whichsim):
        for eseed in newseeds:
            calc_xi_mock_natural(whichsim, newdesign, newdecor, rp_bins_course, pibins, params, eseed, rsd = rsd)
            # calc_xir(whichsim, newdesign, newdecor, rp_bins_course, params, eseed, rsd = rsd)
            calc_wp(whichsim, newdesign, newdecor, rp_bins, params, eseed, rsd = rsd)

    start = time.time()
    p = multiprocessing.Pool(10)
    p.map(run_onebox, range(params['num_sims']))
    print(time.time() - start)

    # plot_xi(newdesign, newdecor, rp_bins_course, pibins, params, newseeds, rsd = rsd)
    # plot_xir(newdesign, newdecor, np.sqrt(rp_bins_course[1:]*rp_bins_course[:-1]), params, newseeds, rsd = rsd)
    plot_wp(newdesign, newdecor, rp_saito, params, newseeds, rsd = rsd)
    compare_to_boss(newdesign, newdecor, rp_bins, params, newseeds, rsd = rsd)

