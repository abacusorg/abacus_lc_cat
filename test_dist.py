import numpy as np
import matplotlib.pyplot as plt

# speed of light
c = 299792.458 # km/s
#plot_type = 'yz'
i_cut = 0; i_plot = 1
#i_cut = 1; i_plot = 0

Lbox = 1000.
dimensions = np.array([Lbox,Lbox,Lbox])
mini = -500
extra = 90.
#extra = 2.*Lbox

origin = (-dimensions+10.)/2

def distance_periodic(x0, x1, dimensions):
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

def distance(x0, x1):
    return np.sqrt(np.sum((x0-x1)**2,axis=1))

# solve when the crossing of the light cones occurs and the interpolated position and velocity
def solve_crossing(r1,r2,pos1,pos2,chi1,chi2):


    # TESTING periodic tuks
    '''
    #inds = np.where(delta > 0.5 * dimensions)[0]
    dist = np.sqrt(np.sum((pos1-pos2)**2,axis=1))
    #delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    #inds = np.where(delta < 4.)[0]
    inds = np.where(dist > 0.5*Lbox)[0]
    r1 = r1[inds]
    r2 = r2[inds]
    pos1 = pos1[inds]
    pos2 = pos2[inds]
    '''


    
    # official
    delta_pos = np.abs(pos1 - pos2)
    sign_delta = np.sign(pos2-pos1)
    #delta_pos = np.where(delta_pos > 0.5 * dimensions, (delta_pos - dimensions), delta_pos)
    delta_pos = sign_delta*delta_pos

    print(delta_pos.shape)

    # normal
    delta_r = np.abs(r1 - r2)
    sign_delta_r = np.sign(r2-r1)
    delta_r = delta_r*sign_delta_r
    
    # solve for eta_star, where chi = eta_0-eta
    # equation is r1+(chi1-chi)/(chi1-chi2)*(r2-r1) = chi
    # with solution chi_star = (r1(chi1-chi2)+chi1(r2-r1))/((chi1-chi2)+(r2-r1))
    chi_star = (r1*(chi1-chi2)+chi1*(delta_r))/((chi1-chi2)+delta_r)
    
    # get interpolated positions of the halos
    v_avg = delta_pos/(chi1-chi2)
    pos_star = pos1+v_avg*(chi1-chi_star[:,None])

    # interpolated velocity [km/s]
    vel_star = v_avg*c #vel1+a_avg*(chi1-chi_star)

    # mark True if closer to chi2 (this snapshot) 
    bool_star = np.abs(chi1-chi_star) >  np.abs(chi2-chi_star)

    #assert np.sum((chi_star > chi1) | (chi_star < chi2)) == 0, "Solution is out of bounds"
    
    return chi_star, pos_star, vel_star, bool_star

def plot_lines():
    plt.axvline(500.,color='k',ls='--')
    plt.axvline(-500.,color='k',ls='--')
    plt.axhline(500.,color='k',ls='--')
    plt.axhline(-500.,color='k',ls='--')


com_dist_prev_main_this_info_lc = np.load('%d.npy'%0)
com_dist_this_info_lc = np.load('%d.npy'%1)
pos_prev_main_this_info_lc = np.load('%d.npy'%2)
pos_this_info_lc = np.load('%d.npy'%3)
chi_prev = np.load('%d.npy'%4)
chi_this = np.load('%d.npy'%5)


# START TESTING

delta_pos = np.abs(pos_prev_main_this_info_lc - pos_this_info_lc)
delta_pos = np.where(delta_pos > 0.5 * dimensions, (delta_pos - Lbox), delta_pos)
delta_sign = np.sign(pos_prev_main_this_info_lc - pos_this_info_lc)


dist = np.sqrt(np.sum((pos_prev_main_this_info_lc - pos_this_info_lc)**2,axis=1))
inds = np.where(dist > 0.5 * Lbox)[0]
#inds = np.arange(pos_this_info_lc.shape[0])

# new position  I BELIEVE THIS IS WHAT MAKES THE LARGEST DIFFERENCE ACCOMPANIED WITH CHI2 > R1 > CHI1 only
pos_prev_main_this_info_lc = pos_this_info_lc + delta_sign*delta_pos
com_dist_prev_main_this_info_lc = distance(origin,pos_prev_main_this_info_lc)
com_dist_this_info_lc = distance(origin,pos_this_info_lc)


print("percentage weird = ",len(inds)*100./pos_this_info_lc.shape[0])


# isolate the problematic ones

pos_this_info_lc = pos_this_info_lc[inds]
pos_prev_main_this_info_lc = pos_prev_main_this_info_lc[inds]
com_dist_this_info_lc = com_dist_this_info_lc[inds]
com_dist_prev_main_this_info_lc = com_dist_prev_main_this_info_lc[inds]


x_min = mini
x_max = x_min+extra
    
x = pos_this_info_lc[:,i_cut]
choice = (x > x_min) & (x < x_max)
    
y = pos_this_info_lc[choice,i_plot]
z = pos_this_info_lc[choice,2]

plt.figure(1)
plot_lines()
plt.scatter(y,z,color='dodgerblue',s=0.1,label='current objects')

plt.legend()
plt.axis('equal')
plt.savefig("this.png")

x = pos_prev_main_this_info_lc[:,i_cut]
choice = (x > x_min) & (x < x_max)

y = pos_prev_main_this_info_lc[choice,i_plot]
z = pos_prev_main_this_info_lc[choice,2]

plt.figure(2)
plot_lines()
plt.scatter(y,z,color='orangered',s=0.1,label='main progenitor')

plt.legend()
plt.axis('equal')
plt.savefig("prev.png")

# END TESTING


chi_star_this_info_lc, pos_star_this_info_lc, vel_star_this_info_lc, bool_star_this_info_lc = solve_crossing(com_dist_prev_main_this_info_lc,com_dist_this_info_lc,pos_prev_main_this_info_lc,pos_this_info_lc,chi_prev,chi_this)

# TESTING
pos_star_this_info_lc = pos_star_this_info_lc[bool_star_this_info_lc]

x_min = mini
x_max = x_min+extra
    
x = pos_star_this_info_lc[:,i_cut]
choice = (x > x_min) & (x < x_max)
    
y = pos_star_this_info_lc[choice,i_plot]
z = pos_star_this_info_lc[choice,2]

plt.figure(0)
plt.scatter(y,z,color='dodgerblue',s=0.1,label='interpolated objects')

plt.legend()
plt.axis('equal')
plt.savefig("interp.png")
plt.show()
