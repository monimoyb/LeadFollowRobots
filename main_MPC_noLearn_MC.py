# Code producing the trajectory for the first comparison strategy in "Decentralized 2-Robot Transportation with Local and Indirect Sensing"
# No learning of follower seen obstacles in this code
# This code runs a bunch of simulations for varying obstacles and initial conditions and then counts failed attempts
# Monimoy Bujarbaruah
#####
import numpy as np 
import random
import matplotlib.pyplot as plt
import scipy as sio
from scipy.optimize import minimize
from scipy import interpolate
import pdb 
from shapely.geometry import Point
from shapely.geometry import Polygon 
from shapely.geometry import LineString
from shapely.geometry.polygon import LinearRing
from shapely.ops import unary_union 
from shapely.ops import cascaded_union 
from shapely.ops import nearest_points
import math
from matplotlib import animation
import warnings
warnings.filterwarnings("ignore")
random.seed(4)
### Defining a room class
class room:
    def __init__(self, xdim, ydim):
        self.xdim = xdim
        self.ydim = ydim

    # defining the seen obstacles by the leader/follower in its circular Lidar range
    def seen_obs(self, x, y, r, obsd):    
        # first defining the triangle vision field here
        phi_points = [] 
        cl_ppoints = [] # store only phi and the radial distance here
        for phi in np.linspace(0,2*np.pi,num=100):
            cloud_phiF = []
            line = LineString([(x, y), (x+r*np.cos(phi), y+r*np.sin(phi))])
            for ob in obsd.values():
                intr_ob = line.intersection(ob)  
                if not intr_ob.is_empty:
                    intr_p = nearest_points(intr_ob, Point(x,y)) 
                    cloud_phiF.append(np.linalg.norm([intr_p[0].x-x,intr_p[0].y-y], 2))
            if len(cloud_phiF):
                cl_p = min(cloud_phiF)
                if cl_p < r:
                    phi_points.append(phi)
                    cl_ppoints.append(cl_p)

        return (phi_points, cl_ppoints)

# Get the states/vel of the follower
def fol_statesVel(x,y,xdot, ydot,theta,theta_dot, l1, l2):
    xf = x-(l1+l2)*np.cos(theta)
    yf = y-(l1+l2)*np.sin(theta)
    xfdot = xdot + (l1+l2)*np.sin(theta)*theta_dot
    yfdot = ydot - (l1+l2)*np.cos(theta)*theta_dot
    return [xf, yf, xfdot, yfdot]

# Get the critical obstacle of the follower
def get_folCrOb(x,y,xdot,ydot,theta,l1,l2,r,obsdic,dcr):
    folSeenObs = r1.seen_obs(x,y,r,obsdic)
    cr_ob = []
    if not folSeenObs==():
        for i in range(len(folSeenObs[1])):
            if folSeenObs[1][i]<=dcr:
                cr_ob.append((x+folSeenObs[1][i]*np.cos(folSeenObs[0][i]), y+folSeenObs[1][i]*np.sin(folSeenObs[0][i]))) 

    if cr_ob == []:
        return cr_ob 
    else:                                      # pick the closest one 
        dis = []
        for i in range(len(cr_ob)):
            dis.append(np.linalg.norm([cr_ob[i][0]-x, cr_ob[i][1]-y], 2))
        min_d = min(dis)
        min_index = dis.index(min_d)   
    
        return [cr_ob[min_index]]         

# Define system dynamics here.   
def rob_dyn(x, u, v, l1, l2, m1, m2, mr, J, ts):
    q1 = -(l1*np.sin(x[4])* (-v[1]*l2 + u[1]*l1 + u[2] + v[2])/J + l1*np.cos(x[4])*x[5]**2) + 1/(m1+m2 + mr)*(np.cos(x[4])*(u[0]+v[0]) - np.sin(x[4])*(u[1]+v[1]))

    q2 = -(-l1*np.cos(x[4])*(-v[1]*l2 + u[1]*l1 + u[2] + v[2])/J + l1*np.sin(x[4])*x[5]**2 ) + 1/(m1+m2+mr)*(np.sin(x[4])*(u[0]+v[0]) + np.cos(x[4])*(u[1]+v[1]))

    xdot = np.array([x[1], q1, x[3], q2, x[5], (-v[1]*l2 + u[1]*l1 + u[2] + v[2])/J ])
    xnew = x + ts*xdot                              

    return xnew

# Get look ahead points. These are are S_tar states including the goal x and y
def get_lookah(gx, gy, N):                   
    # construct the reference
    for i in range(N+1):
        ref_step = np.array([[gx],[0],[gy],[0],[0],[0]]) # not using any theta reference here 
        if i == 0:
            ref = ref_step
        else:    
            ref = np.concatenate((ref, ref_step), axis=0)      
    return ref

#### Extra useful functions 
def quad_form(x, M):
    return x.T @ M @ x

## Angle between two vectors
def angle_vec(v1,v2):
    phiS = np.arcsin(([np.cross(v1,v2)])/(np.linalg.norm(v1,2)*np.linalg.norm(v2,2)))
    phiC = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1,2)*np.linalg.norm(v2,2)))

    threshold = 1.e-6   # because floating point comparison
    # Case 1: phi true between 0 and pi/2
    if (phiS <= phiC+threshold) and (phiS >= phiC-threshold):
        ang_vec = phiS
        
    # Case 2: phi true between pi/2 and pi
    elif (np.pi-phiS <= phiC+threshold) and (np.pi-phiS >= phiC-threshold):
        ang_vec = phiC
        
    # Case 3: phi true between -pi and -pi/2
    elif (-np.pi-phiS <= -phiC+threshold) and (-np.pi-phiS >= -phiC-threshold):
        ang_vec = -phiC
    
    # Case 4: phi true between -pi/2 and 0
    elif (phiS <= -phiC+threshold) and (phiS >= -phiC-threshold):
        ang_vec = phiS
    else:
        print("WARNING: something is wrong with the phi cases...")
        ang_vec = 0.
    
    return ang_vec

### Function to find the closest obstacle points from any given (x,y) coordinate
#   This is then used to put penalty later in the cost function if anything goes toward the known obstacles!
def find_clObsCost(x,y,po):
    poin_pos = Point(x,y)
    cl_obs = nearest_points(po, poin_pos)
    if np.sqrt((cl_obs[0].x-x)**2 + (cl_obs[0].y-y)**2) <= 0.3:
        cost_clOb = 6./(0.001+np.linalg.norm([cl_obs[0].x-x,cl_obs[0].y-y],2))  # this is $\ell(.)$ in the draft 
    else:
        cost_clOb = 0.1

    return cost_clOb

# write the optimization problem here for the leader  
class CostCl:

    def __init__(self, s_curx, s_cury, s_curxd, s_curyd, s_curth, s_curthd, Q, R, l1, l2, m1, m2, mr, K2, N, nx, nu, J, ref, po, dt):

        self.s_curx = s_curx
        self.s_cury = s_cury
        self.s_curxd = s_curxd
        self.s_curyd = s_curyd
        self.s_curth = s_curth
        self.s_curthd = s_curthd
        self.Q = Q
        self.R = R
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.mr = mr
        self.K2 = K2
        self.N = N
        self.nx = nx
        self.dt = dt
        self.nu = nu
        self.J = J
        self.ref = ref
        self.po = po                # all the known obstacles until that point in time. This is a shapely object. 

    def cost(self,upred):
        xpred = np.zeros([self.nx*(self.N+1),1])
        xpred[0:self.nx, 0] = np.array([self.s_curx,self.s_curxd, self.s_cury, self.s_curyd, self.s_curth, self.s_curthd])
        cosopt = 0.
        for i in range(self.N):
            # penalize leader deviation from ref 
            stateErr = self.ref[i*self.nx:(i+1)*self.nx].T[0] - xpred[i*self.nx:(i+1)*self.nx,0]
            # penalize leader-follower bar proximity to known obstacles 
            cost_clOb = 0
            for alpha in np.arange(0,1,0.05):
                cost_clOb = cost_clOb +  find_clObsCost(xpred[i*self.nx]-alpha*(self.l1+self.l2)*np.cos(xpred[i*self.nx+4]), 
                                                       xpred[i*self.nx+2]-alpha*(self.l1+self.l2)*np.sin(xpred[i*self.nx+4]), self.po)            
            cosopt = cosopt + quad_form(stateErr,self.Q) + quad_form(upred[i*self.nu:(i+1)*self.nu], self.R) + cost_clOb

            xpred[(i+1)*self.nx:(i+2)*self.nx,0] = rob_dyn(xpred[i*self.nx:(i+1)*self.nx,0],upred[i*self.nu:(i+1)*self.nu],self.K2*upred[i*self.nu:(i+1)*self.nu],self.l1,self.l2,self.m1,self.m2, self.mr, self.J, self.dt)
        
        stateErr = self.ref[self.N*self.nx:(self.N+1)*self.nx].T[0] - xpred[self.N*self.nx:(self.N+1)*self.nx,0]
        cost_clOb = 0
        for alpha in np.arange(0,1,0.1):
            cost_clOb = cost_clOb +  find_clObsCost(xpred[self.N*self.nx]-alpha*(self.l1+self.l2)*np.cos(xpred[self.N*self.nx+4]), 
                                                   xpred[self.N*self.nx+2]-alpha*(self.l1+self.l2)*np.sin(xpred[self.N*self.nx+4]), self.po)           
        cosopt = cosopt + quad_form(stateErr, self.Q)  + cost_clOb
        return cosopt

### Parameters
nx = 6              # number of states 
nu = 3              # number of inputs 
# penalty matrices
Q = np.array([[120., 0., 0., 0., 0., 0.], [0., 4., 0., 0., 0, 0.], [0., 0., 120., 0., 0., 0.], [0., 0., 0., 4., 0., 0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.01]])
R = np.array([[0.05, 0., 0.], [0., 0.05, 0.], [0., 0., 0.01]])
#
T =  90             # Simulation duration (collisions expected within this. Increase to be sure. Code will be slower) 
m1 = m2 = 0.04      # masses
mr = 0.01           # rod mass
l1 = l2 = 0.8       # lengths
Jr = 1./12.*mr*(l1+l2)**2
J = Jr + mr*(l1-l2)**2/4. + m1*l1**2 + m2*l2**2 # MOI total 
dt = 0.03           # sampling time
ddt = 0.02          # inference time for the follower
N = 3               # mpc horizon
# input constraints. Same for leader and the follower. 
K2 = 0.5                                                # follower expected gain
frac_lead = 0.5                                         # this fraction is for the leader
frac_folCr = 1-K2*frac_lead                             # this fraction is for follower to react to obstacles
# actual constraint set U
Fab = 5.
Fpb = 5.
taub = 0.5
# give the correct chunk to leader
Fab_l = 5.*frac_lead
Fpb_l = 5.*frac_lead
taub_l = 1.*frac_lead
hb = [(-Fab, Fab), (-Fpb, Fpb), (-taub, taub)]
bnds = []
for i in np.arange(0,N,1):
    bnds.append(hb[0])
    bnds.append(hb[1])
    bnds.append(hb[2])
bnds = tuple(bnds)
##### These to be chosen such that the follower does not go beyond saturation
r = 1.2                 # lookahead for both
dcr = 1.1               # critical distance. smaller or equal to than r 
# This fraction is to react to critical obstacles 
Fab_fc = 5.*frac_folCr
Fpb_fc = 5.*frac_folCr
taub_fc = 1.*frac_folCr
K1 = np.array([[Fab_fc/dcr, 0., 0.], [0., Fpb_fc/dcr, 0.], [0., 0., 0.]])      # follower gain to react to critical obs
### Write the room parameters here 
rmxdim = 10.
rmydim = 10.
r1 = room(rmxdim,rmydim)     # Define the room with dimensions here
################## Goal ##########################
gx = 3.                    # [m]
gy = 3.95                  # [m]
### RUN MULTIPLE SIMULATIONS FROM HERE ON. VARY OBSTACLES. DETECT COLLISIONS AND MARK FAILED TRIALS ######
mc_count = 100                                       # number of random runs of the algoritm
col_thres = 0.1                                      # threshold distance to flag a collision event 
col_count = 0                                        # trajectory collision count across all simulations
time_out_count = 0                                   # count time outs 
# Fixing the initial conditions 
x0 = 7.5
y0 = 7.2
theta0 = 0.1 
xDot0 = 0.
yDot0 = 0.
thetaDot0 = 0.
initCond = np.transpose(np.array([x0,xDot0,y0,yDot0, theta0, thetaDot0]))
step_2_tar = []

# Big loop
for mc in range(mc_count):
    
    #### To make obstacles from here on. Defining as rings to allow nonconvexity
    # Grid obs1 and obs2 now and shift them around to run all the simulations 
    obs1x_r = random.uniform(0.,0.2)
    obs1y_r = random.uniform(0.,0.5)
    #
    obs2x_r = random.uniform(0.,0.4)
    obs2y_r = random.uniform(0.,0.2)
    Obs1 = LinearRing([(3.8 + obs1x_r ,6.-obs1y_r),(3.9+obs1x_r,6.-obs1y_r),(3.9+obs1x_r,7.-obs1y_r),(3.8+obs1x_r,7.-obs1y_r)])
    Obs2 = LinearRing([(5.1 - obs2x_r, 4.5+obs2y_r), (5.1-obs2x_r, 5.+obs2y_r), (5.8-obs2x_r, 5.+obs2y_r), (5.8-obs2x_r,4.5+obs2y_r)])
    #
    Obs3 = LinearRing([(6.5, 3.), (6.5, 6.), (8., 6.), (8.,3)])

    obsdic = {'obst1': Obs1, 'obs2': Obs2, 'obs3': Obs3}                           # dictionary of varying obstacles here 

    ################### SIMULATION STARTS HERE ###########
    states = np.zeros([nx,T+1])
    inputs = np.zeros([2*nu,T])
    cos_cl = np.zeros([1,T])
    fol_prevInp  = np.zeros(nu)     # start follower previous input at zeros

    states[:,0] = initCond
    cr_obs_list = []
    obsxl = np.array([rmxdim,0.])
    obsyl = np.array([0.,rmydim])

    ### Start of main time loop 
    for index in range(T):
        if index>35:                                      # make the plot sparser in this case towards the end
            if index % 3.0 == 0.0:
                show_animation = True
            else:
                show_animation = False
        else:
            show_animation = True

        # first leader finds own free space
        (ang,rad) =  r1.seen_obs(states[0,index], states[2,index], r, obsdic)   # tuple of (phi, cl_r) 
        ox = np.cos(ang) * rad
        oy = np.sin(ang) * rad                             # obstacle cordinate points 
        
        # set obstacle positions (include the ones leader sees and also the ones inferred until then)
        obsxl = np.append(obsxl,states[0,index]+ox)
        obsyl = np.append(obsyl,states[2,index]+oy)        # these are the ones directly seen by the leader. Stored. 
        obsx = obsxl
        obsy = obsyl

        ### create all the obstacle unions 
        poi = []
        for i in range(len(obsx)):
            poi.append(Point(obsx[i],obsy[i])) 
        po = unary_union(poi)

        ### write a piece here that checks collsions and flags them 
        p1 = Polygon(Obs1)
        p2 = Polygon(Obs2)
        p3 = Polygon(Obs3)
        pol = []
        pol.append(p1)
        pol.append(p2)
        pol.append(p3)
        pol_union = unary_union(pol)
        linS =  LineString([(states[0,index], states[2,index]), (states[0,index]-(l1+l2)*np.cos(states[4,index]), states[2,index]-(l1+l2)*np.sin(states[4,index]))])
        interS = pol_union.intersection(linS) 

        if interS.is_empty == False: 
            print('COLLISSION DETECTED ON THE ROD! COUNTING THIS AND STOPPING THE TRAJECTORY.')
            col_count = col_count + 1 
            break

        # start position
        sx = states[0,index]   # [m] for leader
        sy = states[2,index]   # [m] for leader
        sfolx = sx-(l1+l2)*np.cos(states[4,index]) 
        sfoly = sy-(l1+l2)*np.sin(states[4,index])
        slinex = np.array([sx, sx-(l1+l2)*np.cos(states[4,index])])
        sliney = np.array([sy, sy-(l1+l2)*np.sin(states[4,index])])    
        
        # Now have to solve an optimization problem for control synthesis
        ref = get_lookah(gx, gy, N)                                            # reference generation    
        u0_guess =  np.random.rand(nu*N,1)                                     # intial solution guess


        myCost = CostCl(states[0,index],states[2,index],states[1,index],states[3,index],states[4,index],states[5,index],\
                             Q, R, l1, l2, m1, m2, mr, K2, N, nx, nu, J, ref, po, dt)
        
        sol = sio.optimize.minimize(myCost.cost, u0_guess, options={'disp': False, 'maxiter': 100}, method='SLSQP', bounds=bnds) 
        sol_array = sol.x
        inputs[0:nu,index] =  sol_array[0:nu]                                  # first mpc input is applied [Fa_l, Fp_l, tau_l]

        #### Follower's inference part is to be done here to calculate the follower input
        # sticking to leader states directly, because they can always be calculated from the follower's
        states_ddt = np.transpose(rob_dyn(states[:,index], inputs[0:nu,index], fol_prevInp, l1, l2, m1, m2, mr, J, ddt))
        # See appendix of the paper for these
        q_1_calc = (states_ddt[1]-states[1,index])/ddt                         
        q_2_calc = (states_ddt[3]-states[3,index])/ddt
        t_1_calc = (states_ddt[5]-states[5,index])/ddt
        # now calculate F_{al}, F_{pl} and tau_l from these inferred q_1, q_2 and t. 
        tmp1 = q_1_calc - l1*np.sin(states[4,index])*fol_prevInp[1]*l2/J + l1*np.sin(states[4,index])*fol_prevInp[2]/J +  l1*np.cos(states[4,index])*states[5,index]**2 \
                    -1/(m1+m2+mr)*(np.cos(states[4,index])*(fol_prevInp[0]) - np.sin(states[4,index])*fol_prevInp[1]) 

        tmp2 = q_2_calc + l1*np.cos(states[4,index])*fol_prevInp[1]*l2/J - l1*np.cos(states[4,index])*fol_prevInp[2]/J +  l1*np.sin(states[4,index])*states[5,index]**2 \
                    -1/(m1+m2+mr)*(np.sin(states[4,index])*(fol_prevInp[0]) + np.cos(states[4,index])*fol_prevInp[1]) 

        Amat = np.array([ [0, l1, 1], [1/(m1+m2+mr)*np.cos(states[4,index]), -(l1**2*np.sin(states[4,index])/J + 1/(m1+m2+mr)*np.sin(states[4,index])), -l1*np.sin(states[4,index])/J], \
                        [1/(m1+m2+mr)*np.sin(states[4,index]), (l1**2*np.cos(states[4,index])/J + 1/(m1+m2+mr)*np.cos(states[4,index])), l1*np.cos(states[4,index])/J] ])
        bmat = np.array([J*t_1_calc + fol_prevInp[1]*l2 - fol_prevInp[2], tmp1, tmp2])

        fol_inf_array = np.linalg.solve(Amat, bmat)

        ##################################################################################
        ### Will use these to apply the critical obstacle force correctly at that time step
        [xfddt, yfddt, xfdotddt, yfdotddt] = fol_statesVel(states_ddt[0],states_ddt[2],states_ddt[1],
                                              states_ddt[3],states_ddt[4],states_ddt[5], l1, l2)

        # Compute follower critical obstacles 
        fol_crOb = get_folCrOb(xfddt,yfddt,xfdotddt,yfdotddt,states_ddt[4],l1,l2,r,obsdic,dcr)

        # apply the follower inputs
        if fol_crOb == []: 
            inputs[nu:, index] = K2*fol_inf_array
        else:
            fromFol2Lead = [states_ddt[0]-xfddt, states_ddt[2]-yfddt]
            fromFol2Obs = [fol_crOb[0][0]-xfddt, fol_crOb[0][1]-yfddt]
            ang_vec = angle_vec(fromFol2Obs, fromFol2Lead)+np.pi
            cr_force =  (dcr-np.linalg.norm(fromFol2Obs, 2))*np.array([[np.cos(ang_vec)],[-np.sin(ang_vec)], [0]])
            inputs[nu:, index] = K2*fol_inf_array + np.transpose(K1@cr_force)

        fol_prevInp = inputs[nu:, index]                        # store the previous step inputs of the follower

        # Simulation step
        states[:,index+1] = np.transpose(rob_dyn(states_ddt, inputs[0:nu,index], inputs[nu:,index], l1, l2, m1, m2, mr, J, dt-ddt))

        if  fol_crOb:                                           # tracking the real critical obstacles        
            obs_b = Point(fol_crOb[0][0],fol_crOb[0][1])        
            cr_obs_list.append(obs_b)

                # Plot simulation step
        if show_animation:                              
            plt.figure(2)   
            plt.plot(obsxl, obsyl, "xr", markersize = 2)
            if fol_crOb:
                plt.plot(fol_crOb[0][0], fol_crOb[0][1], "xb", markersize = 2)      
            plt.plot(sx, sy, "or", markersize=3, mew = 1)
            plt.plot(sfolx, sfoly, "ob", markersize=3, mew =1)
            plt.plot(gx, gy, '*r', markersize=12)
            plt.plot(slinex, sliney, "-y", linewidth=1.5)
            plt.grid(True)
            plt.xlim((2.8,8.))
            plt.ylim((3.8,8.))
            plt.pause(0.001)
            plt.draw()

        print('Time Index:')
        print(index)

        # quit this if leader too close already
        if np.linalg.norm([states[0,index+1]-gx, states[2,index+1]-gy],2) <= 0.5:      
            time_out_count = time_out_count+1
            step_2_tar.append(index+1)
            break   
    
    plt.clf()
    
    # reached end of the loop but too slow and taking forever
    if index == T-1:
        if np.linalg.norm([states[0,T]-gx, states[2,T]-gy],2) >= 0.5:                                        
            time_out_count = time_out_count+1
            step_2_tar.append(T) 

    print('Simulation Count is:')
    print(mc+1)
    if step_2_tar:
       print("Array of Steps in CFT")
        print(step_2_tar)
    print('Failed Due to Collision Count is:')
    print(col_count)
    print("Failed due to time out count is")
    print(time_out_count)
    print('Successful Simulation Count:')
    print(mc+1 - col_count - time_out_count)

print("Average steps in a collision free trial is:")
print(np.mean(step_2_tar))

###############################################################################################################