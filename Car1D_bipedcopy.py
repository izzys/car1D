import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef,distanceJointDef,wheelJointDef, contactListener)#)

import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
CAR_SCALE = 2


MOTORS_SPEED = 0
MOTORS_TORQUE = 0
#SPEED_HIP     = 4
#SPEED_KNEE    = 6
LIDAR_RANGE   = 300/SCALE*CAR_SCALE

INITIAL_RANDOM = 5

# car params:
body_width = 1.0
body_length = 1.0
body_color = (0.3,0.2,0.6)

suspension_width = 0.2
suspension_length = 1.0
suspension_upper_color = (0.7,0.2,0.8)
suspension_lower_color = (0.2,0.6,0.7)

wheel_radius = 0.3
wheel_color = (0.7,0.5,0.2)

#hull_length = 30

HULL_POLY =[
    (-50,+15), (+6,+15), (+54,+4),
    (+54,-18), (-50,-18)
    ]

WHEEL_RADIUS = 14/SCALE*CAR_SCALE

LEG_DOWN = -8/SCALE*CAR_SCALE
LEG_W, LEG_H = 16/SCALE*CAR_SCALE, 34/SCALE*CAR_SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE*CAR_SCALE,y/SCALE*CAR_SCALE) for x,y in HULL_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0020,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy

# LEG_FD = fixtureDef(
#                     shape=polygonShape(box=(LEG_W/2, LEG_H/2)),
#                     density=1.0,
#                     restitution=0.0,
#                     categoryBits=0x0020,
#                     maskBits=0x001)

# LOWER_FD = fixtureDef(
#                     shape=polygonShape(box=(0.8*LEG_W/2, LEG_H/2)),
#                     density=1.0,
#                     restitution=0.0,
#                     categoryBits=0x0020,
#                     maskBits=0x001)

UPPER_FD = fixtureDef(
                    shape=polygonShape(box=(0.8*LEG_W/2*CAR_SCALE, LEG_H/2*CAR_SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

LOWER_FD = fixtureDef(
                    shape=polygonShape(box=(0.8*LEG_W/2*CAR_SCALE, LEG_H/2*CAR_SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

WHEEL_FD = fixtureDef(
                    shape=circleShape(radius=WHEEL_RADIUS),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

# solver params
dt = 0.005
max_time = 30

# display params
skip_frames = 10

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
Dtheta_laser_rays = 2.5 #deg
Theta0_laser_rays = -65.0 #deg

ground_xstart = -5.0
ground_xend = 100.0
ground_ds = 0.1


MaxU_positive = 1500.0
MaxU_negative = 4500.0

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
        self.len = l
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

    def __init__(self,x0,y0,w,l,color,y_ind):

        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.len = l
        self.color = color
        self.initial_Lvec = []
        self.x_ind = x_ind
        self.y_ind = y_ind

        for i in range (1,N_laser_rays):
            theta = Theta0_laser_rays + Dtheta_laser_rays*(i-1)
            ray_len = math.sqrt((self.y0/math.tan(-theta*math.pi/180))**2 + self.y0**2)
            self.initial_Lvec.append(ray_len)

    def get_pos(self,X):

        x = self.x0 + X[self.x_ind,0]
        y = self.y0 + X[self.y_ind,0]
        return x, y

    def get_laser_intersection(self, x0, y0, theta, i):
        x_ground = 0
        y_ground = 0
        return x_ground, y_ground


class _Ground:

    def __init__(self,x_start,x_end,ds):

        self.x0 = x_start
        self.xf = x_end
        self.length = x_end - x_start
        self.ds = ds

    def get_height(self,x,dx):

        z,dz = 0.0,0.0

        return z,dz

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.hull==contact.fixtureA.body or self.env.hull==contact.fixtureB.body:
            #self.env.game_over = True
            pass

        # for leg in [self.env.legs[1], self.env.legs[3]]:
        #     if leg in [contact.fixtureA.body, contact.fixtureB.body]:
        #         leg.ground_contact = True
        for wheel in self.env.wheels:
            if wheel in [contact.fixtureA.body, contact.fixtureB.body]:
                wheel.ground_contact = True


    def EndContact(self, contact):
        # for leg in [self.env.legs[1], self.env.legs[3]]:
        #     if leg in [contact.fixtureA.body, contact.fixtureB.body]:
        #         leg.ground_contact = False
        for wheel in self.env.wheels:
            if wheel in [contact.fixtureA.body, contact.fixtureB.body]:
                wheel.ground_contact = False


class BipedalWalker(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = FRICTION)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = FRICTION,
                    categoryBits=0x0001,
                )


       # high = np.array([np.inf]*24)
       # self.action_space = spaces.Box(np.array([-1,-1,-1,-1]), np.array([+1,+1,+1,+1]))
       # self.observation_space = spaces.Box(-high, high)

        ################################## from car1D #########################
        x0 = -body_width / 2
        y0 = wheel_radius + suspension_length
        w = body_width
        l = body_length
        color = body_color
        y_ind = z1_ind
        self.body = _Box2D(x0, y0, w, l, color, y_ind)

        x0 = -suspension_width / 2
        y0 = wheel_radius
        w = suspension_width
        l = suspension_length
        color = suspension_lower_color
        y_ind = z2_ind
        self.suspension_lower = _Box2D(x0, y0, w, l, color, y_ind)

        x0 = -suspension_width * 1.2 / 2
        y0 = wheel_radius + suspension_length / 2
        w = suspension_width * 1.2
        l = suspension_length
        color = suspension_upper_color
        y_ind = z1_ind
        self.suspension_upper = _Box2D(x0, y0, w, l, color, y_ind)

        x0 = body_width / 2
        y0 = wheel_radius + suspension_length + body_length
        w = scanner_width
        l = scanner_length
        color = scanner_color
        y_ind = z1_ind
        self.laser_scanner = _LaserScanner(x0, y0, w, l, color, y_ind)

        x0 = 0
        y0 = wheel_radius
        color = wheel_color
        r = wheel_radius
        y_ind = z2_ind
        self.wheel = _Cylinder2D(x0, y0, r, color, y_ind)

        x_start = ground_xstart
        x_end = ground_xend
        ds = ground_ds
        self.ground = _Ground(x_start, x_end, ds)

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
        self.X0 = np.zeros((Xdim, 1))

        # derivatives matrices:
        self.Minv = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0 / m1, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 1.0 / m2, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / (m1 + m2)]])

        self.G = np.matrix([[0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                            [k1, -k1, 0.0, b1, -b1, 0.0],
                            [-k1, (k1 + k2), 0.0, -b1, (b1 + b2), 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.Phi = np.matrix([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [k2, b2, 0.0],
                              [0.0, 0.0, 1.0]])

        """
        Two kinds of valid input:
            Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
            Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0])) # low and high are arrays of the same shape
        """

        # action space
        self.action_space = spaces.Box(low=1.0, high=5.0, shape=(1,), dtype=np.float64)

        # observation space
        self.observation_space = spaces.Tuple(
            (spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float64),  # "horizontal vel"
             spaces.Box(low=-1e2, high=1e2, shape=(1,), dtype=np.float64),  # "vertical acc"
             spaces.Box(low=-5e3, high=5e3, shape=(1,), dtype=np.float64),  # "control cmd"
             spaces.Box(low=np.ones(N_laser_rays, np.float64) * (-1),
                        high=np.ones(N_laser_rays, np.float64))))  # "laser scanner"

        # state
        self.state = None

        # time
        self.t = 0.0
        self.dt = dt
        self.t_threshold = max_time

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
        self.frame = 1

        # protect against illegal calls
        self.steps_beyond_done = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        # for leg in self.legs:
        #     self.world.DestroyBody(leg)
        for wheel in self.wheels:
             self.world.DestroyBody(wheel)
        self.legs = []
        self.joints = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity += self.np_random.uniform(-1, 1)/SCALE   #1
                y += velocity

            elif state==PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [
                    (x,              y),
                    (x+TERRAIN_STEP, y),
                    (x+TERRAIN_STEP, y-4*TERRAIN_STEP),
                    (x,              y-4*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices=[(p[0]+TERRAIN_STEP*counter,p[1]) for p in poly]
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state==PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4*TERRAIN_STEP

            elif state==STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                poly = [
                    (x,                      y),
                    (x+counter*TERRAIN_STEP, y),
                    (x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
                    (x,                      y+counter*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

            elif state==STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        ]
                    self.fd_polygon.shape.vertices=poly
                    t = self.world.CreateStaticBody(
                        fixtures = self.fd_polygon)
                    t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                    self.terrain.append(t)
                counter = stair_steps*stair_width

            elif state==STAIRS and not oneshot:
                s = stair_steps*stair_width - counter - stair_height
                n = s/stair_width
                y = original_y + (n*stair_height)*TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter==0:
                counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
                if state==GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge)
            color = (0.3, 1.0 if i%2==0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
                 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def reset(self):

        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = TERRAIN_STEP*TERRAIN_STARTPAD/2
        init_y = TERRAIN_HEIGHT+LEG_H*2
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            fixtures = HULL_FD
                )
        self.hull.color1 = (0.5,0.4,0.9)
        self.hull.color2 = (0.3,0.3,0.5)
      #  self.hull.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        self.suspensions_lower = []
        self.suspensions_upper = []
        self.joints = []
        for i in [-1, +1]:
            upper = self.world.CreateDynamicBody(
                position=(init_x + i * LEG_H, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(0.0),
                fixtures=UPPER_FD
            )
            upper.color1 = (0.2 - i / 10., 0.6 - i / 10., 0.1 - i / 10.)
            upper.color2 = (0.2 - i / 10., 0.7 - i / 10., 0.5 - i / 10.)
            rjd = distanceJointDef(
                bodyA=self.hull,
                bodyB=upper,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
            )
            self.suspensions_upper.append(upper)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position=(init_x + i * LEG_H, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(0.0),
                fixtures=LOWER_FD
            )
            lower.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            lower.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = distanceJointDef(
                bodyA=upper,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
            )
            self.suspensions_lower.append(lower)

        self.wheels = []
        self.axles = []
        for i in [-1,+1]:
            wheel = self.world.CreateDynamicBody(
                position = (init_x + i*LEG_H, init_y-LEG_H),
                fixtures = WHEEL_FD
                )
            wheel.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            wheel.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = wheelJointDef(
                bodyA=self.suspensions_lower[min(0, i)],
                bodyB=wheel,
                localAnchorA=(i*LEG_H,-LEG_H),#(0, LEG_H),
                localAnchorB=(0,0),#(init_x + i*LEG_H, init_y - LEG_H/2),#(0,LEG_H/2),
                enableMotor=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 0.0,
                )
            self.wheels.append(wheel)
            self.axles.append(self.world.CreateJoint(rjd))


           #     self.joints.append(self.world.CreateJoint(rjd))

            # lower = self.world.CreateDynamicBody(
            #     position = (init_x, init_y - LEG_H*3/2 - LEG_DOWN),
            #     angle = (i*0.05),
            #     fixtures = LOWER_FD
            #     )
            # lower.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            # lower.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            # rjd = revoluteJointDef(
            #     bodyA=leg,
            #     bodyB=lower,
            #     localAnchorA=(0, -LEG_H/2),
            #     localAnchorB=(0, LEG_H/2),
            #     enableMotor=True,
            #     enableLimit=True,
            #     maxMotorTorque=MOTORS_TORQUE,
            #     motorSpeed = 1,
            #     lowerAngle = -1.6,
            #     upperAngle = -0.1,
            #     )
            # lower.ground_contact = False
            # self.legs.append(lower)
            # self.joints.append(self.world.CreateJoint(rjd))

        # self.wheel1 = self.world.CreateDynamicBody(
        #     position = (0, 0),
        #     fixtures = WHEEL_FD
        #         )
        # self.wheel1.color1 = (0.6,0.2,0.9)
        # self.wheel1.color2 = (0.1,0.6,0.7)
        #
        # self.wheel2 = self.world.CreateDynamicBody(
        #     position = (0, 0),
        #     fixtures = WHEEL_FD
        #        )
        # self.wheel2.color1 = (0.6,0.2,0.9)
        # self.wheel2.color2 = (0.1,0.6,0.7)

        # rjd = revoluteJointDef(
        #     bodyA=self.hull,
        #     bodyB=self.wheel,
        #     localAnchorA=(init_x, init_y),
        #     localAnchorB=(0, 0),
        #     enableMotor=True,
        #     enableLimit=False,
        #     maxMotorTorque=MOTORS_TORQUE,
        #     motorSpeed=0.01,
        #    # lowerAngle=-np.inf(),
        #   #  upperAngle=np.inf(),
        # )
        # self.wheel_axle = self.world.CreateJoint(rjd)


        self.drawlist = self.terrain + self.wheels + [self.hull] #+ [self.wheel]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0
        self.lidar = [LidarCallback() for _ in range(10)]

        self.dz1_prev = 0.0
        self.Fx_prev = 0.0
        self.U_prev = 0.0
        self.err_prev = 0.0
        self.t = 0.0
        self.frame = 1
        self.state = self._get_initial_state()
        self.steps_beyond_done = None

        ob = self._get_observation()
        return ob


        #return self.step(np.array([0,0,0,0]))[0]

    def step(self, action):
        #self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        # control_speed = False  # Should be easier as well
        # if control_speed:
        #     self.joints[0].motorSpeed = float(SPEED_HIP  * np.clip(action[0], -1, 1))
        #     self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
        #     self.joints[2].motorSpeed = float(SPEED_HIP  * np.clip(action[2], -1, 1))
        #     self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        # else:
        #     self.joints[0].motorSpeed     = float(SPEED_HIP     * np.sign(action[0]))
        #     self.joints[0].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
        #     self.joints[1].motorSpeed     = float(SPEED_KNEE    * np.sign(action[1]))
        #     self.joints[1].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
        #     self.joints[2].motorSpeed     = float(SPEED_HIP     * np.sign(action[2]))
        #     self.joints[2].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
        #     self.joints[3].motorSpeed     = float(SPEED_KNEE    * np.sign(action[3]))
        #     self.joints[3].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

        self.axles[0].motorSpeed     = float(MOTORS_SPEED     * np.sign(action))
        self.axles[0].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action), 0, 1))
        self.axles[1].motorSpeed     = float(MOTORS_SPEED    * np.sign(action))
        self.axles[1].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action), 0, 1))


        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + LEG_H + math.sin(1.5*i/10.0)*LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/10.0)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        # state = [
        #     self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
        #     2.0*self.hull.angularVelocity/FPS,
        #     0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
        #     0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,
        #     self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
        #     self.joints[0].speed / SPEED_HIP,
        #     self.joints[1].angle + 1.0,
        #     self.joints[1].speed / SPEED_KNEE,
        #     1.0 if self.legs[1].ground_contact else 0.0,
        #     self.joints[2].angle,
        #     self.joints[2].speed / SPEED_HIP,
        #     self.joints[3].angle + 1.0,
        #     self.joints[3].speed / SPEED_KNEE,
        #     1.0 if self.legs[3].ground_contact else 0.0
        #     ]


        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/FPS,
            0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,
            self.axles[0].speed,
            self.axles[1].speed,
            ]


        state += [l.fraction for l in self.lidar]
      #  assert len(state)==24

        self.scroll = pos.x - VIEWPORT_W/SCALE/5

        shaping  = 130*pos[0]/SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            done   = True
        return np.array(state), reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 0, VIEWPORT_H/SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+VIEWPORT_W/SCALE, 0),
            (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
            (self.scroll,                  VIEWPORT_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render+1) % 10
        i = self.lidar_render
        if i < 2*len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar)-i-1]
            self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

######################## from Car1D #####################

    def _get_initial_state(self):

        # Get ground:
        x = self.X0[x_ind,0]
        dx = self.X0[dx_ind,0]
        [z, dz] = self.ground.get_height(x, dx)

        # Get acc:
        dz1 = self.X0[dz1_ind,0]
        acc = self._get_acc(dz1)

        # Get laser scan:
        x0_laser_rays, y0_laser_rays, x_ground, z_ground = self._get_laser_scan(self.X0)

        state = self._wrap_state(self.X0,acc,0.0,0.0,z_ground)

        return state

    def _get_reward(self,action):

        r = 0.0
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

    # def _init_display(self):
    #
    #     self.hFig = plt.figure()
    #     self.hAxes = self.hFig.add_subplot(111, autoscale_on=True, xlim=(-5, 5), ylim=(-1, 4))
    #
    #     self.hTimeObj_str = 'time = %.1fs'
    #     self.hTimeObj = self.hAxes.text(0.05, 0.9, '0.0s', transform=self.hAxes.transAxes)
    #
    #     X0 = np.zeros((Xdim,1))
    #
    #     # add car wheel
    #     xy = self.wheel.get_pos(X0)
    #     r = self.wheel.radius
    #     face_color = self.wheel.color
    #     self.hWheel = mpatches.Circle(xy, r, fc=face_color, zorder=1)
    #     self.hAxes.add_patch(self.hWheel)
    #
    #     # add upper suspension
    #     xy = self.suspension_upper.get_pos(X0)
    #     width = self.suspension_upper.w
    #     height = self.suspension_upper.len
    #     face_color = self.suspension_upper.color
    #     self.hUpperSuspension = mpatches.Rectangle(xy, width, height, fc=face_color, zorder=3)
    #     self.hAxes.add_patch(self.hUpperSuspension)
    #
    #     # add lower suspension
    #     xy = self.suspension_lower.get_pos(X0)
    #     width = self.suspension_lower.w
    #     height = self.suspension_lower.len
    #     face_color = self.suspension_lower.color
    #     self.hLowerSuspension = mpatches.Rectangle(xy, width, height, fc=face_color, zorder=2)
    #     self.hAxes.add_patch(self.hLowerSuspension)
    #
    #     # add car body
    #     xy = self.body.get_pos(X0)
    #     width = self.body.w
    #     height = self.body.len
    #     face_color = self.body.color
    #     self.hBody = mpatches.Rectangle(xy, width, height, fc=face_color, zorder=1)
    #     self.hAxes.add_patch(self.hBody)
    #
    #     # add laser scanner
    #     xy = self.laser_scanner.get_pos(X0)
    #     width = self.laser_scanner.w
    #     height = self.laser_scanner.len
    #     face_color = self.laser_scanner.color
    #     self.hLaserScanner = mpatches.Rectangle(xy, width, height, fc=face_color, zorder=1)
    #     self.hAxes.add_patch(self.hLaserScanner)
    #
    #     # add ground
    #     x0 = self.ground.x0
    #     xf = self.ground.xf
    #     ds = self.ground.ds
    #
    #     ground_x = np.arange(x0, 5, ds)
    #     ground_z = np.zeros(len(ground_x))
    #
    #     self.hGround = lns.Line2D(ground_x, ground_z)
    #
    #     #  self.hAxes.figure.canvas.draw()
    #
    #     plt.axis('equal')
    #     plt.axis('off')
    #     #  plt.tight_layout()
    #
    #     plt.draw()
    #     plt.pause(0.1)

    # def render(self, mode='human'):
    #
    #     if self.hFig is None:
    #         self._init_display()
    #
    #     if not bool(self.frame % skip_frames):
    #
    #         X, acc, Fx, u, z_ground = self._unwrap_state(self.state)
    #
    #         self.hTimeObj.set_text(self.hTimeObj_str % self.t)
    #
    #         # update car body pos:
    #         xy = self.body.get_pos(X)
    #         self.hBody.set_xy(xy)
    #
    #         # update wheel pos:
    #         xy = self.wheel.get_pos(X)
    #         self.hWheel.center = xy
    #
    #         # update upper suspension pos:
    #         xy = self.suspension_upper.get_pos(X)
    #         self.hUpperSuspension.set_xy(xy)
    #
    #         # update lower suspension pos:
    #         xy = self.suspension_lower.get_pos(X)
    #         self.hLowerSuspension.set_xy(xy)
    #
    #         # update laser scanner pos:
    #         xy = self.laser_scanner.get_pos(X)
    #         self.hLaserScanner.set_xy(xy)
    #
    #         # update ground:
    #         x0 = self.ground.x0
    #         xf = self.ground.xf
    #         ds = self.ground.ds
    #
    #         ground_x = np.arange(x0, 5, ds)
    #         ground_z = np.zeros(len(ground_x))
    #
    #         self.hGround.set_data(ground_x, ground_z)
    #
    #         # time
    #         self.hTimeObj.set_text(self.hTimeObj_str % self.t)
    #
    #         # update figure
    #         plt.draw()
    #         plt.pause(1e-6)
    #
    #     self.frame = self.frame + 1

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

    def _get_laser_scan(self, X):

        x0_laser_rays, y0_laser_rays = self.laser_scanner.get_pos(X)

        x_ground = np.zeros(N_laser_rays)
        y_ground = np.zeros(N_laser_rays)
        for i in range(N_laser_rays):

            theta_i = Theta0_laser_rays + i*Dtheta_laser_rays
            xtmp, ytmp = self.laser_scanner.get_laser_intersection(x0_laser_rays, y0_laser_rays, theta_i, i)
            x_ground[i] = xtmp
            y_ground[i] = ytmp

        return x0_laser_rays,y0_laser_rays,x_ground,y_ground

    def _DerivativesFcn(self,X, z, dz, Fx):

        U = np.array([[z] , [dz] , [Fx]])

        dX = self.Minv*(-self.G*X + self.Phi*U)

        return dX



class BipedalWalkerHardcore(BipedalWalker):
    hardcore = True

if __name__=="__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = BipedalWalkerHardcore()
    env.reset()
    # steps = 0
    # total_reward = 0
    a = np.array([0.0])




    # STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    # SPEED = 0.29  # Will fall forward on higher speed
    # state = STAY_ON_ONE_LEG
    # moving_leg = 0
    # supporting_leg = 1 - moving_leg
    # SUPPORT_KNEE_ANGLE = +0.1
    # supporting_knee_angle = SUPPORT_KNEE_ANGLE

    while True:
        s, r, done, info = env.step(a)
        # total_reward += r
        # if steps % 20 == 0 or done:
        #     print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        #     print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
        #     print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
        #     print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        # steps += 1

        # contact0 = s[8]
        # contact1 = s[13]
        # moving_s_base = 4 + 5*moving_leg
        # supporting_s_base = 4 + 5*supporting_leg
        #
        # hip_targ  = [None,None]   # -0.8 .. +1.1
        # knee_targ = [None,None]   # -0.6 .. +0.9
        # hip_todo  = [0.0, 0.0]
        # knee_todo = [0.0, 0.0]
        #
        # if state==STAY_ON_ONE_LEG:
        #     hip_targ[moving_leg]  = 1.1
        #     knee_targ[moving_leg] = -0.6
        #     supporting_knee_angle += 0.03
        #     if s[2] > SPEED: supporting_knee_angle += 0.03
        #     supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
        #     knee_targ[supporting_leg] = supporting_knee_angle
        #     if s[supporting_s_base+0] < 0.10: # supporting leg is behind
        #         state = PUT_OTHER_DOWN
        # if state==PUT_OTHER_DOWN:
        #     hip_targ[moving_leg]  = +0.1
        #     knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
        #     knee_targ[supporting_leg] = supporting_knee_angle
        #     if s[moving_s_base+4]:
        #         state = PUSH_OFF
        #         supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
        # if state==PUSH_OFF:
        #     knee_targ[moving_leg] = supporting_knee_angle
        #     knee_targ[supporting_leg] = +1.0
        #     if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
        #         state = STAY_ON_ONE_LEG
        #         moving_leg = 1 - moving_leg
        #         supporting_leg = 1 - moving_leg
        #
        # if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
        # if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
        # if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
        # if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]
        #
        # hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
        # hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
        # knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
        # knee_todo[1] -= 15.0*s[3]
        #
        # a[0] = hip_todo[0]
        # a[1] = knee_todo[0]
        # a[2] = hip_todo[1]
        # a[3] = knee_todo[1]
        # a = np.clip(0.5*a, -1.0, 1.0)

        env.render()
        if done: break