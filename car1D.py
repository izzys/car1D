import math
import numpy as np
import gym
from gym import spaces, logger
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


# solver params
dt = 0.005
max_time = 30

# display params
skip_frames = 25

# car params:
body_width = 1.0
body_length = 1.0
body_color = (0.3,0.2,0.6)

suspension_width = 0.2
suspension_length = 0.7
suspension_upper_color = (0.7,0.2,0.8)
suspension_lower_color = (0.2,0.6,0.7)

wheel_radius = 0.4
wheel_color = (0.7,0.5,0.2)

m1 = 1500.0 # quarter car weight (kg)
m2 = 300.0 # wheel weight (kg)

k1 = 80000.0 # suspension stiffness
k2 = 50000.0 # tire stiffness

b1 = 1750.0 # suspension damping
b2 = 15000.0 # tire damping

# laser scanner params
scanner_width = 0.1
scanner_length = 0.1
scanner_color = (0,0,0)

N_laser_rays = 16
Dtheta_laser_rays = 2.5*math.pi/180 #deg
Theta0_laser_rays = -65.0*math.pi/180 #deg

ground_xstart = -5.0
ground_xend = 100.0
ground_ds = 0.05
ground_color1 = (.3,.1,.3)
ground_color2 = (.1,.5,.3)

MaxU_positive = np.array([1500.0])
MaxU_negative = np.array([4500.0])

Xdim = 6
z1_ind = 0
z2_ind = 1
x_ind = 2
dz1_ind = 3
dz2_ind = 4
dx_ind = 5

class _Box2D:

    def __init__(self,x0,y0,w,l,color,y_ind):

        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.l = l
        self.color = color
        self.x_ind = x_ind
        self.y_ind = y_ind

    def get_pos(self,X):

        x = self.x0 + X[self.x_ind,0]
        y = self.y0 + X[self.y_ind,0]

        return x,y


class _Cylinder2D:

    def __init__(self,x0,y0,r,color,y_ind):

        self.x0 = x0
        self.y0 = y0
        self.color = color
        self.radius = r
        self.x_ind = x_ind
        self.y_ind = y_ind

    def get_pos(self,X):

        x = self.x0 + X[self.x_ind,0]
        y = self.y0 + X[self.y_ind,0]
        return x, y


class _LaserScanner:

    def __init__(self,x0,y0,w,l,color,y_ind,hGround):

        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.l = l
        self.color = color
        self.initial_Lvec = []
        self.x_ind = x_ind
        self.y_ind = y_ind

        self.f_prev = 0.0
        self.L_prev = 0.0

        self.ground = hGround

        for i in range(N_laser_rays):
            theta = Theta0_laser_rays + Dtheta_laser_rays*(i-1)
            ray_len = math.sqrt((self.y0/math.tan(-theta))**2 + self.y0**2)
            self.initial_Lvec.append(ray_len)

    def get_pos(self,X):

        x = self.x0 + X[self.x_ind,0]
        y = self.y0 + X[self.y_ind,0]
        return x, y

    def get_laser_scan(self, X):

        x0_laser_rays, y0_laser_rays = self.get_pos(X)

        x0_laser_rays = x0_laser_rays + scanner_width
        y0_laser_rays = y0_laser_rays + scanner_length/2

        x_ground = np.zeros(N_laser_rays)
        y_ground = np.zeros(N_laser_rays)

        for i in range(N_laser_rays):

            theta_i = Theta0_laser_rays + i*Dtheta_laser_rays
            x_ground[i], y_ground[i] = self._get_laser_intersection(x0_laser_rays, y0_laser_rays, theta_i, i)

        return x0_laser_rays,y0_laser_rays,x_ground,y_ground

    def _get_laser_intersection(self, x0, y0, theta, ray_ind):

        # do a newton 's method line search:
        L0 = self.initial_Lvec[ray_ind]
        k = 1
        f_diff = 1e9
        x_diff = 1e9
        init = True

        while f_diff > 1e-6 and x_diff > 1e-6 and k < 100:

            [L, f_diff, x_diff] = self._do_newton_iteration(x0, y0, theta, L0, init)
            k = k + 1
            L0 = L
            init = False
           # print("k",k)

        if L < 0:
            L = self.initial_Lvec[ray_ind]

        x_ground = x0 + L * math.cos(theta)
        y_ground = y0 + L * math.sin(theta)

        return x_ground, y_ground

    def _do_newton_iteration(self,x0,y0,theta,L,init):

        if init:
            self.f_prev = 0.0
            self.L_prev = 0.0
            f = 1e3
            dL = 1e-3
        else:
            x_ray= np.array([x0 + L * math.cos(theta)])
            dx_ray = np.array([0.0])
            f,_ = self.ground.get_height(x_ray, dx_ray) - (y0 + L * math.sin(theta))
            dL = (L - self.L_prev) / (f - self.f_prev) * f

        L_next = L - dL

        f_diff = abs(f - self.f_prev)
        L_diff = abs(L - self.L_prev)
        self.f_prev = f
        self.L_prev = L

        return L_next,f_diff,L_diff

class _Ground:

    def __init__(self,x_start,x_end,ds):

        self.x0 = x_start
        self.xf = x_end
        self.length = x_end - x_start
        self.ds = ds

        self.reset()

        self.ground_type = 'sin' # 'sin' or 'square'

    def reset(self):

        self.f1 = 0.1 + np.random.uniform() * 15
        self.f2 = self.f1 * 3 + np.random.uniform() * 3

        self.a1 = 0.01 + np.random.uniform() * 0.2
        self.a2 = 0.001 + np.random.uniform() * 0.05

        self.bump_start = 5 + np.random.uniform()*20;
        self.bump_end = self.bump_start + 1 + np.random.uniform() * 10

    def get_height(self,x,dx):

        z = np.zeros(len(x))
        dz =  np.zeros(len(x))
        for i in range(len(x)):
            if x[i] > self.bump_start and x[i] < self.bump_end:

                if self.ground_type=='sin':
                    z[i] = self.a1 * math.cos(self.f1 * x[i]) + self.a2 * math.cos(self.f2 * x[i])
                    dz[i] = -self.a1 * self.f1 * math.sin(self.f1 * x[i]) * dx[i] - self.a2 * self.f2 * math.sin(self.f2 * x[i]) * dx[i]

                elif self.ground_type=='square':
                    z[i] = self.a1
                    dz[i] = 0.0

        return z,dz

class Car1D(gym.Env):

    """
     Description:
         A quarter car model, moving in 1 direction (Hence 1D). model consists of a mass, 2 springs, 2 dampers in series(modeling the suspension and tire), and a non smooth terrain that imposes vertical displacements.
         The horizontal movement is governed by a control law, that follows the desired speed. A laser scanner is mounted on the car and measures the ground height in front of the car.

     Observation:
         Type: Tuple(4)
             Num	Type            Observation                             Min         Max   Units
         1	    Box(1)        Car's horizontal Velocity                     0          15     [m/s]
         2      Box(1)        Car's vertical acceleration                   -25        25    [m/s^2]
         3	    Box(1)        Gas/Brakes Control cmd                        -5e3       5e3    [N]
         4      Box(16)       array of ground heights from laser scanner    -1          1      [m]

     Actions:
         Type: Box(1)
         Num	Action
         0	    Desired velocity command

     Reward:
         Reward is -acc^2+vel^2-brake_force for every time step , including the termination step.
         we give positive reward for horizontal velocity, negative reward for vertical acceleration,
         and negative reward for braking.
         if the vehicle drives backwards and reaches x=x0, we reduce -100


     Starting State:
         All observations are zero

     Episode Termination:
         Car Position is beyond x0,xf
         Episode time exceeds 30 seconds
     """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):

        x0 = -body_width/2
        y0 = wheel_radius + suspension_length
        w = body_width
        l = body_length
        color = body_color
        y_ind = z1_ind
        self.body = _Box2D(x0,y0,w,l,color,y_ind)

        x0 = -suspension_width/2
        y0 = wheel_radius
        w = suspension_width
        l = suspension_length
        color = suspension_lower_color
        y_ind = z2_ind
        self.suspension_lower = _Box2D(x0,y0,w,l,color,y_ind)

        x0 = -suspension_width*1.2/2
        y0 = wheel_radius + suspension_length/2
        w = suspension_width*1.2
        l = suspension_length
        color = suspension_upper_color
        y_ind = z1_ind
        self.suspension_upper = _Box2D(x0,y0,w,l,color,y_ind)

        x_start = ground_xstart
        x_end = ground_xend
        ds = ground_ds
        self.ground = _Ground(x_start,x_end,ds)

        x0 = body_width/2-scanner_width/2
        y0 = wheel_radius + suspension_length + body_length-scanner_length/2
        w = scanner_width
        l = scanner_length
        color = scanner_color
        y_ind = z1_ind
        self.laser_scanner = _LaserScanner(x0,y0,w,l,color,y_ind,self.ground)

        x0 = 0
        y0 = wheel_radius
        color = wheel_color
        r = wheel_radius
        y_ind = z2_ind
        self.wheel = _Cylinder2D(x0,y0,r,color,y_ind)



        # acc calculation
        self.dz1_prev = 0.0

        # Fx calculation
        self.Fx_prev = 0.0
        self.U_prev = 0.0
        self.err_prev = 0.0

        self.tau_positive = 1.0
        self.tau_negative = 0.2

        self.b1_positive = dt / self.tau_positive
        self.a1_positive = (1.0 - dt / self.tau_positive)

        self.b1_negative = dt / self.tau_negative
        self.a1_negative = (1.0 - dt / self.tau_negative)

        self.Kp = 1.8
        self.Kd = 2.5

        # dynamic state
        self.X0 = np.zeros( (Xdim,1) )

        # derivatives matrices:
        self.Minv = np.matrix([[1.0 ,  0.0  , 0.0  , 0.0     , 0.0     , 0.0],
                               [0.0 ,  1.0  , 0.0  , 0.0     , 0.0     , 0.0],
                               [0.0 ,  0.0  , 1.0  , 0.0     , 0.0     , 0.0],
                               [0.0 ,  0.0  , 0.0  , 1.0/m1  , 0.0     , 0.0],
                               [0.0 ,  0.0  , 0.0  , 0.0     , 1.0/m2  , 0.0],
                               [0.0 ,  0.0  , 0.0  , 0.0     , 0.0     , 1.0/(m1+m2)]])

        self.G = np.matrix([[0.0, 0.0,        0.0, -1.0,  0.0,       0.0],
                            [0.0, 0.0,       0.0,  0.0, -1.0,       0.0],
                            [0.0, 0.0,       0.0,  0.0,  0.0,      -1.0],
                            [ k1, -k1,       0.0,   b1,  -b1,       0.0],
                            [-k1, (k1 + k2), 0.0,  -b1,  (b1 + b2), 0.0],
                            [0.0, 0.0,       0.0,  0.0,  0.0,       0.0]])

        self.Phi = np.matrix([[0.0 , 0.0 , 0.0 ],
                              [0.0 , 0.0 , 0.0 ],
                              [0.0 , 0.0 , 0.0 ],
                              [0.0 , 0.0 , 0.0 ],
                              [k2  , b2  , 0.0 ],
                              [0.0 , 0.0 , 1.0 ]])


        """
        Two kinds of valid input:
            Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
            Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0])) # low and high are arrays of the same shape
        """

        # action space
        self.action_space = spaces.Box(low=0.0, high=15.0,shape=(1,),dtype=np.float64)

        # observation space
        self.observation_space = spaces.Tuple((spaces.Box(low=0.0, high=15.0,shape=(1,),dtype=np.float64), # "horizontal vel"
                                               spaces.Box(low=-1e2, high=1e2,shape=(1,),dtype=np.float64), # "vertical acc"
                                               spaces.Box(low=-5e3, high=5e3,shape=(1,),dtype=np.float64), # "control cmd"
                                               spaces.Box(low=np.ones(N_laser_rays,np.float64)*(-1), high=np.ones(N_laser_rays,np.float64)))) # "laser scanner"

        # state
        self.state = None

        # time
        self.t = 0.0
        self.dt = dt
        self.t_final = max_time

        # graphics
        self.hFig = None
        self.hAxes = None
        self.hTimeObj_str = None
        self.hTimeObj = None
        self.hWheel = None
        self.hUpperSuspension = None
        self.hLowerSuspension = None
        self.hBody = None
        self.hLaserScanner = None
        self.hGround = None
        self.hLaserRays = None
        self.frame = 0

        # protect against illegal calls
        self.steps_beyond_done = None

    def _get_initial_state(self):

        # Get ground:
        x = self.X0[x_ind]
        dx = self.X0[dx_ind]
        [z, dz] = self.ground.get_height(x, dx)

        # Get acc:
        dz1 = self.X0[dz1_ind,0]
        acc = self._get_acc(dz1)

        # Get laser scan:
        x0_laser_rays, y0_laser_rays, x_ground, z_ground = self.laser_scanner.get_laser_scan(self.X0)

        state = self._wrap_state(self.X0,acc,0.0,0.0,z_ground)

        return state

    def reset(self):

        self.dz1_prev = 0.0
        self.Fx_prev = 0.0
        self.U_prev = 0.0
        self.err_prev = 0.0
        self.t = 0.0
        self.frame = 0
        self.ground.reset()
        self.state = self._get_initial_state()
        self.steps_beyond_done = None
        ob = self._get_observation()
        return ob


    def step(self,action):

    #    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        X, acc, Fx, u, z_ground = self._unwrap_state(self.state)

        # Get desired_speed:
        vel_cmd = action

        # Get ground:
        x = X[x_ind]
        dx = X[dx_ind]
        [z, dz] = self.ground.get_height(x, dx)

        # Get derivatives:
        dX = self._DerivativesFcn(X, z, dz, Fx)

        # compute next state:
        Xnext = X + self.dt * dX

        # prepare for next iteation

        # Get acc:
        dz1 = X[dz1_ind]
        acc = self._get_acc(dz1)

        # Get laser scan:
        x0_laser_rays, y0_laser_rays, x_ground, z_ground = self.laser_scanner.get_laser_scan(X)

        # Get driving force:
        Fx, u = self._get_Fx(X, z, vel_cmd)

        self.state = self._wrap_state(Xnext,acc,Fx,u,z_ground)
        self.t = self.t + self.dt

        done = Xnext[x_ind] < self.ground.x0  or Xnext[x_ind] > self.ground.xf or self.t > self.t_final

        done = bool(done)

        if not done:
            reward = self._get_reward(action)
        elif self.steps_beyond_done is None:
            # finished episode
            self.steps_beyond_done = 0
            reward = self._get_reward(action)
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        observation = self._get_observation()

        return observation , reward, done, {}

    def _get_reward(self,action):

        X, acc, Fx, u, z_ground = self._unwrap_state(self.state)

        x = X[x_ind,0]
        dx = X[dx_ind,0]

        normalize_u_to_1 = 1/abs(MaxU_negative)
        brake = np.clip(u,-np.inf,0.0)*normalize_u_to_1

        normalize_acc_to_1 = 1/25.0
        acc = acc*normalize_acc_to_1

        normalize_dx_to_1 = 1/15.0
        dx = np.clip(dx,0.0,np.inf)*normalize_dx_to_1

        r = -acc**2 + dx**2 - 0.1*abs(brake)
        if x<self.ground.x0:
            r = r-100

        return r

    def _get_observation(self):

        X, acc, Fx, u, z_ground = self._unwrap_state(self.state)
        dx = X[dx_ind,0]

        # ob = self.observation_space.sample()
        # "horizontal vel": np.array(dx)
        # "vertical acc":   np.array(acc)
        # "control cmd":    np.array(u)
        # "laser scanner":  z_ground

        ob = (np.array([dx]),np.array([acc]),np.array([u]),z_ground)

        return ob

    @staticmethod
    def _unwrap_observation(ob):

        dx = ob[0]
        acc = ob[1]
        u = ob[2]
        z_ground = ob[3]

        return dx,acc,u,z_ground

    def _get_acc(self,dz1):

        ddz1 = (dz1-self.dz1_prev)/self.dt
        self.dz1_prev = dz1

        return ddz1

    def _init_display(self):

        if self.hFig==None:

          #  matplotlib.interactive(True)

            self.hFig = plt.figure()
            self.hAxes = self.hFig.add_subplot(111)#, autoscale_on=True)
            self.hAxes.set_aspect('equal')
            self.hAxes.set_frame_on(False)
            self.hAxes.set_xticks([])
            self.hAxes.set_yticks([])

            self.hAxes1 = self.hFig.add_subplot(331)#, autoscale_on=True)
            #   self.hAxes1.margins(1.5)
            self.hAxes1.grid(True)
            self.hAxes1.set_xticks([])
            self.hAxes1.set_xlim(0, 5)
            MaxAcc = 25.0
            self.hAxes1.set_ylim(-MaxAcc/9.81*3, MaxAcc/9.81*3)
            self.hAxes1.set_ylabel('Acc [g]')

            self.hAxes2 = self.hFig.add_subplot(332)#, autoscale_on=True)
            # self.hAxes2.margins(0.5)
            self.hAxes2.grid(True)
            self.hAxes2.set_xticks([])
            self.hAxes2.set_xlim(0, 5)
            MaxVel = 10.0
            self.hAxes2.set_ylim(0,MaxVel)
            self.hAxes2.set_ylabel('Vel [m/s]')


            self.hAxes3 = self.hFig.add_subplot(333)#, autoscale_on=True)
            #  self.hAxes3.margins(0.5)
            self.hAxes3.grid(True)
            self.hAxes3.set_xlim(0, N_laser_rays)
            self.hAxes3.set_ylim(-0.5, 1)
            self.hAxes3.set_xticks([])
            self.hAxes3.set_ylabel('Terrain [m]')

           # plt.show()

        else:
            self.hAxes.clear()
            self.hAxes.set_xticks([])
            self.hAxes.set_yticks([])

            self.hAxes1.clear()
            self.hAxes1.grid(True)
            self.hAxes1.set_xticks([])
            self.hAxes1.set_xlim(0, 5)
            MaxAcc = 25.0
            self.hAxes1.set_ylim(-MaxAcc/9.81*3, MaxAcc/9.81*3)
            self.hAxes1.set_ylabel('Acc [g]')

            self.hAxes2.clear()
            self.hAxes2.grid(True)
            self.hAxes2.set_xticks([])
            self.hAxes2.set_xlim(0, 5)
            MaxVel = 10.0
            self.hAxes2.set_ylim(0,MaxVel)
            self.hAxes2.set_ylabel('Vel [m/s]')

            self.hAxes3.clear()
            self.hAxes3.grid(True)
            self.hAxes3.set_xlim(0, N_laser_rays)
            self.hAxes3.set_ylim(-0.5, 1)
            self.hAxes2.set_xticks([])
            self.hAxes3.set_ylabel('Terrain [m]')

        self.hTimeObj_str = 'time = %.1fs'
        self.hTimeObj = self.hAxes.text(0.84, -0.05, '0.0s', transform=self.hAxes.transAxes)

        X0 = np.zeros((Xdim,1))

        # add car wheel
        xy = self.wheel.get_pos(X0)
        r = self.wheel.radius
        face_color = self.wheel.color
        self.hWheel = mpatches.Circle(xy, r, fc=face_color, zorder=1)
        self.hAxes.add_patch(self.hWheel)

        # add upper suspension
        xy = self.suspension_upper.get_pos(X0)
        width = self.suspension_upper.w
        height = self.suspension_upper.l
        face_color = self.suspension_upper.color
        self.hUpperSuspension = mpatches.Rectangle(xy, width, height, fc=face_color, zorder=3)
        self.hAxes.add_patch(self.hUpperSuspension)

        # add lower suspension
        xy = self.suspension_lower.get_pos(X0)
        width = self.suspension_lower.w
        height = self.suspension_lower.l
        face_color = self.suspension_lower.color
        self.hLowerSuspension = mpatches.Rectangle(xy, width, height, fc=face_color, zorder=2)
        self.hAxes.add_patch(self.hLowerSuspension)

        # add car body
        xy = self.body.get_pos(X0)
        width = self.body.w
        height = self.body.l
        face_color = self.body.color
        self.hBody = mpatches.Rectangle(xy, width, height, fc=face_color, zorder=1)
        self.hAxes.add_patch(self.hBody)

        # add laser scanner
        xy = self.laser_scanner.get_pos(X0)
        width = self.laser_scanner.w
        height = self.laser_scanner.l
        face_color = self.laser_scanner.color
        self.hLaserScanner = mpatches.Rectangle(xy, width, height, fc=face_color, zorder=1)
        self.hAxes.add_patch(self.hLaserScanner)

        # add laser rays
        x0,y0,ground_x,ground_z = self.laser_scanner.get_laser_scan(X0)

        self.hLaserRays = []
        for i in range(len(ground_z)):
            self.hLaserRays.append(mlines.Line2D([x0,ground_x[i]], [y0,ground_z[i]], lw=1., alpha=0.3, color=(1.0,0.0,0.0)))
            self.hAxes.add_line(self.hLaserRays[i])

        # add ground
        x0 = self.ground.x0
        xf = self.ground.xf
        ds = self.ground.ds

        ground_x = np.arange(x0, xf, ds)
        ground_dx = np.zeros(len(ground_x))
        ground_z,_ = self.ground.get_height(ground_x,ground_dx)

        # ind1 = i for (i,val) in ground_x if np.ceil(abs(ground_x))%2 == 0
        # ind2 = i for (i,val) in ground_x if np.floor(abs(ground_x))%2 == 0

        ind1 = np.where(np.ceil(abs(ground_x))%2 == 0)
        ind2 = np.where(np.floor(abs(ground_x))%2 == 0)

        self.hGround1 = mlines.Line2D(ground_x[ind1], ground_z[ind1], lw=5., alpha=0.3, color=ground_color1,linestyle='',marker="8")
        self.hGround2 = mlines.Line2D(ground_x[ind2], ground_z[ind2], lw=5., alpha=0.3, color=ground_color2,linestyle='',marker="8")
        self.hAxes.add_line(self.hGround1)
        self.hAxes.add_line(self.hGround2)

        self.hAxes.set_xlim(-body_length * 2,  body_length * 8)
        self.hAxes.set_ylim(-1, 5)

        # Acc display
        MaxAcc = 25.0
        self.acc_tvec = np.arange(0.0,5.0,self.dt)
        self.acc_vec = np.zeros(len(self.acc_tvec))
        self.hAccVisLine = mlines.Line2D(self.acc_tvec,self.acc_vec , color='b')
        self.hAccVisLineUpper = mlines.Line2D( self.acc_tvec, np.ones(len(self.acc_tvec)) * MaxAcc / 9.81, color='r',linestyle='--')
        self.hAccVisLineLower = mlines.Line2D( self.acc_tvec, np.ones(len(self.acc_tvec)) *(-MaxAcc) / 9.81, color='r',linestyle='--')
        self.hAxes1.add_line(self.hAccVisLine)
        self.hAxes1.add_line(self.hAccVisLineUpper)
        self.hAxes1.add_line(self.hAccVisLineLower)

        #Vel display
        MaxVel = 5.0
        self.vel_tvec = np.arange(0.0,5.0,self.dt)
        self.vel_vec = np.zeros(len(self.vel_tvec))
        self.hVelVisLine = mlines.Line2D(self.vel_tvec,self.vel_vec , color='b')
        self.hVelVisLineRef = mlines.Line2D(self.vel_tvec, np.ones(len(self.vel_tvec)) * MaxVel, color='k',linestyle='--')
        self.hAxes2.add_line(self.hVelVisLine)
        self.hAxes2.add_line(self.hVelVisLineRef)

        # Terrain display
        ds = 0.1
        self.terrain_xvec = np.arange(0,N_laser_rays)
        self.hTerrainVisLine = mlines.Line2D(self.terrain_xvec, np.zeros(len(self.terrain_xvec)), color='k',linestyle='',marker='.',markersize=5)
        self.hAxes3.add_line(self.hTerrainVisLine)
        # plt.draw()
        # plt.pause(0.1)

    def render(self, mode='human'):

        if self.hFig is None or self.frame==0:
            self._init_display()

        if not bool(self.frame % skip_frames):

            X, acc, Fx, u, z_ground = self._unwrap_state(self.state)

            self.hTimeObj.set_text(self.hTimeObj_str % self.t)

            # update car body pos:
            xy = self.body.get_pos(X)
            self.hBody.set_xy(xy)

            # update wheel pos:
            xy = self.wheel.get_pos(X)
            self.hWheel.center = xy

            # update upper suspension pos:
            xy = self.suspension_upper.get_pos(X)
            self.hUpperSuspension.set_xy(xy)

            # update lower suspension pos:
            xy = self.suspension_lower.get_pos(X)
            self.hLowerSuspension.set_xy(xy)

            # update laser scanner pos:
            xy = self.laser_scanner.get_pos(X)
            self.hLaserScanner.set_xy(xy)

            # update laser scan:
            x0, y0, ground_x, ground_z = self.laser_scanner.get_laser_scan(X)
            for i in range(len(ground_z)):
                self.hLaserRays[i].set_data([x0, ground_x[i]], [y0, ground_z[i]])

            # update ground:
            # x0 = self.ground.x0
            # xf = self.ground.xf
            # ds = self.ground.ds
            #
            # ground_x = np.arange(x0, 5, ds)
            # ground_z = np.zeros(len(ground_x))
            #
            # self.hGround.set_data(ground_x, ground_z)

         #   self.acc_vec = np.concatenate(( self.acc_vec[1:], acc/9.81 ))

         #   vel =  np.array([np.asscalar(X[dx_ind,0])])
         #   self.vel_vec = np.concatenate(( self.vel_vec[1:], vel ))

            # update acc display:
            self.hAccVisLine.set_data(self.acc_tvec, self.acc_vec)

            # update vel display:
            self.hVelVisLine.set_data(self.vel_tvec, self.vel_vec)

            # update laser scan display:
            self.hTerrainVisLine.set_data(self.terrain_xvec, ground_z)

           # x = X[x_ind,0]
          #  self.hAxes.set_xlim(x-body_length*2,x+body_length*8)

            # time
            self.hTimeObj.set_text(self.hTimeObj_str % self.t)

            # update figure
            plt.draw()
            plt.pause(1e-6)

        self.frame = self.frame + 1

    @staticmethod
    def _wrap_state(X,acc,Fx,u,z_ground):

        state = (X,acc,Fx,u,z_ground)

        return state

    @staticmethod
    def _unwrap_state(state):

        X = state[0]
        acc = state[1]
        Fx = state[2]
        u = state[3]
        z_ground = state[4]

        return X,acc,Fx,u,z_ground

    def _get_Fx(self, X, ground_z, desired_speed):

        # Control law:
        measured_speed = X[dx_ind]
        err = desired_speed - measured_speed

        if err < 0:
            U = self.Kp * err * 2.5 + self.Kd * (err - self.err_prev) / self.dt * 0.25
        else:
            U = self.Kp * err + self.Kd * (err - self.err_prev) / self.dt

        U = U * 1000
        U = max(min(U, MaxU_positive), -MaxU_negative)

        #  apply delay on force:
        if U < 0 and err < 0:
            Fx = self.b1_negative * self.U_prev + self.a1_negative * self.Fx_prev
        else:
            Fx = self.b1_positive * self.U_prev + self.a1_positive * self.Fx_prev

        self.Fx_prev = Fx
        self.U_prev = U
        self.err_prev = err

        return Fx,U

    def _DerivativesFcn(self,X, z, dz, Fx):

        U = np.array([z , dz , [Fx]])
        dX = self.Minv*(-self.G*X + self.Phi*U)

        return dX


