import numpy as np
#import csv

ind = 8
#ind = 12
#ind = 16

pos_u = np.load('pos_u_%d.npy'%ind)
npout_u = np.load('npout_u_%d.npy'%ind)
haloid_u = np.load('haloid_u_%d.npy'%ind)
pos_m = np.load('pos_m_%d.npy'%ind)
print(pos_u.shape,pos_m.shape)


min_dist = 1.
far = (np.abs(np.abs(pos_u[:,0])-500.)>min_dist) & (np.abs(np.abs(pos_u[:,1])-500.)>min_dist) & (np.abs(np.abs(pos_u[:,2])-500.)>min_dist)

pos_u = pos_u[far]
npout_u = npout_u[far]
haloid_u = haloid_u[far]

far = (np.abs(np.abs(pos_m[:,0])-500.)>min_dist) & (np.abs(np.abs(pos_m[:,1])-500.)>min_dist) & (np.abs(np.abs(pos_m[:,2])-500.)>min_dist)
pos_m = pos_m[far]

dist_u = np.sqrt(np.sum((pos_u-np.array([-990,-990,-990]))**2,axis=1))
dist_m = np.sqrt(np.sum((pos_m-np.array([-990,-990,-990]))**2,axis=1))
print(dist_m.min(),dist_m.mean(),dist_m.max())
print(dist_u.min(),dist_u.mean(),dist_u.max())
print(pos_u.shape)
print(dist_u)
#print(pos_u[:150])
print(haloid_u[:150])
print(npout_u[:150])
quit()


def downsample(pos,fac=1000):
    inds = np.arange(pos.shape[0])
    print(len(inds))
    np.random.shuffle(inds)
    inds = inds[::fac]
    print(len(inds))
    pos = pos[inds]
    return pos

pos_m = downsample(pos_m,fac=1000)
#pos_u = downsample(pos_u,fac=1000)
np.savetxt('pos_m.csv', pos_m, delimiter=",")
np.savetxt('pos_u.csv', pos_u, delimiter=",")
