import asdf
import numpy as np
import scipy.stats as scist
import matplotlib.pyplot as plt

# todo: can speed up by using info about how many halos instead concatenating

def extract_superslab(fns):
    z_in = extract_redshift(fns)
    return np.array([np.int(fn.split('z%.3f.'%z_in)[1].split('.asdf')[0]) for fn in fns])
    
def extract_redshift(fns):
    redshift = float(fns[0].split('z')[-1][:5])
    return redshift

def load_merger(filenames,return_mass=False,trim=True):
    for i in range(len(filenames)):
        fn = filenames[i]
        print("File number %i of %i"%(i,len(filenames)-1))
        f = asdf.open(fn,lazy_load=True, copy_arrays=True)
        print(f.tree['data'].keys())
        HI = f.tree['data']['HaloIndex']
        MP = f.tree['data']['MainProgenitor']
        MPP = f.tree['data']['MainProgenitorPrec']
        try:
            iS = f.tree['data']['IsPotentialSplit']
        except:
            iS = np.zeros(len(MP),dtype=int)
            print("Working with coarsified tree")

        if return_mass:
            HM = f.tree['data']['HaloMass']
            #HM = f.tree['data']['Position'][:,0]

        if trim:
            mask = (MP > 0) & (MPP > 0) & (HI > 0)

            HI = HI[mask]
            MP = MP[mask]
            MPP = MPP[mask]
            iS = iS[mask]
            if return_mass:
                HM = HM[mask]

        if i == 0:

            out_HI = HI
            out_MP = MP
            out_MPP = MPP
            out_iS = iS
            if return_mass:
                out_HM = HM
        else:
            out_HI = np.hstack((out_HI,HI))
            out_MP = np.hstack((out_MP,MP))
            out_MPP = np.hstack((out_MPP,MPP))
            out_iS = np.hstack((out_iS,iS))
            if return_mass:
                out_HM = np.hstack((out_HM,HM))

    if not return_mass:
        out_HM = np.zeros(len(out_HI))-1
    return out_HI, out_MP, out_MPP, out_iS, out_HM

def simple_load(filenames,fields):
    
    for i in range(len(filenames)):
        fn = filenames[i]
        print("File number %i of %i"%(i,len(filenames)-1))
        f = asdf.open(fn,lazy_load=True, copy_arrays=True)

        N_halos = f.tree['data'][fields[0]].shape[0]
        num_progs_cum = 0
        if i == 0:                

            dtypes = []
            for field in fields:
                dtype = f.tree['data'][field].dtype
                try:
                    shape = f.tree['data'][field].shape[1]
                    dtype = (dtype,shape)
                except:
                    pass
                if field != 'Progenitors':
                    dtypes.append((field,dtype))
                                
                
                if field == 'Progenitors':
                    final_progs = f.tree['data']['Progenitors']
                    try:
                        num_progs = f.tree['data']['NumProgenitors'] + num_progs_cum
                    except:
                        print("You need to also request 'NumProgenitors' if requesting the 'Progenitors' field");exit()
                    num_progs_cum += np.sum(num_progs)
                    
                    
            final = np.empty(N_halos,dtype=dtypes)
            for field in fields:
                if 'Progenitors' != field:
                    final[field] = f.tree['data'][field]

            if 'Progenitors' in fields:
                final['NumProgenitors'] = num_progs
                
        else:
            new = np.empty(N_halos,dtype=dtypes)
            for field in fields:

                if field != 'Progenitors':
                    new[field] = f.tree['data'][field]    
            
                if field == 'Progenitors':
                    progs = f.tree['data']['Progenitors']
                    final_progs = np.hstack((final_progs,progs))

                    num_progs = f.tree['data']['NumProgenitors'] + num_progs_cum
                    num_progs_cum += np.sum(num_progs)

            if 'Progenitors' in fields:
                new['NumProgenitors'] = num_progs

            final = np.hstack((final,new))

    if 'Progenitors' in fields:
        return final, final_progs
    
    return final

def get_slab_halo(filenames):
    
    slabs = extract_superslab(filenames)
    n_slabs = len(slabs)
    N_halos_slabs = np.zeros(n_slabs,dtype=int)

    for i in range(len(filenames)):
        fn = filenames[i]
        print("File number %i of %i"%(i,len(filenames)-1))
        f = asdf.open(fn, lazy_load=True, copy_arrays=True)
        N_halos = f.tree['data']['HaloIndex'].shape[0]
        N_halos_slabs[i] = N_halos

    i_sort = np.argsort(slabs)
    slabs = slabs[i_sort]
    N_halos_slabs = N_halos_slabs[i_sort]

    return N_halos_slabs, slabs
