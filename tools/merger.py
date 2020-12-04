import asdf
import numpy as np
import scipy.stats as scist
import matplotlib.pyplot as plt
from astropy.table import Table

# todo: can speed up by using info about how many halos instead concatenating


def extract_superslab(fn):
    # looks like "associations_z0.100.0.asdf"
    return int(fn.split('.')[-2])

def extract_superslab_minified(fn):
    # looks like "associations_z0.100.0.asdf.minified"
    return int(fn.split('.')[-3])
    
def extract_redshift(fn):
    # looks like "associations_z0.100.0.asdf.minified" or "associations_z0.100.0.asdf"
    redshift = float('.'.join(fn.split("z")[-1].split('.')[:2]))
    return redshift


def load_merger(filenames, return_mass=False, trim=True):
    for i in range(len(filenames)):
        fn = filenames[i]
        print("File number %i of %i" % (i, len(filenames) - 1))
        f = asdf.open(fn, lazy_load=True, copy_arrays=True)
        print(f["data"].keys())
        HI = f["data"]["HaloIndex"]
        MP = f["data"]["MainProgenitor"]
        MPP = f["data"]["MainProgenitorPrec"]
        try:
            iS = f["data"]["IsPotentialSplit"]
        except:
            iS = np.zeros(len(MP), dtype=int)
            print("Working with coarsified tree")

        if return_mass:
            HM = f["data"]["HaloMass"]
            # HM = f.tree['data']['Position'][:,0]

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
            out_HI = np.hstack((out_HI, HI))
            out_MP = np.hstack((out_MP, MP))
            out_MPP = np.hstack((out_MPP, MPP))
            out_iS = np.hstack((out_iS, iS))
            if return_mass:
                out_HM = np.hstack((out_HM, HM))

    if not return_mass:
        out_HM = np.zeros(len(out_HI)) - 1
    return out_HI, out_MP, out_MPP, out_iS, out_HM


def simple_load(filenames, fields):
    if type(filenames) is str:
        filenames = [filenames]
    
    do_prog = 'Progenitors' in fields
    
    Ntot = 0
    dtypes = {}
    
    if do_prog:
        N_prog_tot = 0
        fields.remove('Progenitors')  # treat specially
    
    for fn in filenames:
        with asdf.open(fn) as af:
            # Peek at the first field to get the total length
            # If the lengths of fields don't match up, that will be an error later
            Ntot += len(af['data'][fields[0]])
            
            for field in fields:
                if field not in dtypes:
                    dtypes[field] = af['data'][field].dtype
            
            if do_prog:
                N_prog_tot += len(af['data']['Progenitors'])
                
    # Make the empty tables
    t = Table({f:np.empty(Ntot, dtype=dtypes[f]) for f in fields}, copy=False)
    if do_prog:
        p = Table({'Progenitors':np.empty(N_prog_tot, dtype=np.int64)}, copy=False)
    
    # Fill the data into the empty tables
    j = 0
    jp = 0
    for i, fn in enumerate(filenames):
        print(f"File number {i+1:d} of {len(filenames)}")
        f = asdf.open(fn, lazy_load=True, copy_arrays=True)
        fdata = f['data']
        thisN = len(fdata[fields[0]])
        
        for field in fields:
            # Insert the data into the next slot in the table
            t[field][j:j+thisN] = fdata[field]
            
        if do_prog:
            thisNp = len(fdata['Progenitors'])
            p['Progenitors'][jp:jp+thisNp] = fdata['Progenitors']
            jp += thisNp
            
        j += thisN
    
    # Should have filled the whole table!
    assert j == Ntot
    
    ret = dict(merger=t)
    if do_prog:
        ret['progenitors'] = p
        assert jp == N_prog_tot
    
    return ret


def get_halos_per_slab(filenames, minified):
    # extract all slabs
    if minified:
        slabs = np.array([extract_superslab_minified(fn) for fn in filenames])
    else:
        slabs = np.array([extract_superslab(fn) for fn in filenames])
    n_slabs = len(slabs)
    N_halos_slabs = np.zeros(n_slabs, dtype=int)

    # extract number of halos in each slab
    for i,fn in enumerate(filenames):
        print("File number %i of %i" % (i, len(filenames) - 1))
        f = asdf.open(fn, lazy_load=True, copy_arrays=True)
        N_halos = len(f["data"]["HaloIndex"])
        N_halos_slabs[i] = N_halos

    # sort in slab order
    i_sort = np.argsort(slabs)
    slabs = slabs[i_sort]
    N_halos_slabs = N_halos_slabs[i_sort]

    return dict(zip(slabs, N_halos_slabs))
