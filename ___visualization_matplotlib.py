import matplotlib.pyplot as plt
import matplotlib.lines as lns
import numpy as np
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation

skip_frames = 10


class Visualization:

    def __init__(self,car):

        self.hCar = car
        self.Xdim = len(car.X0)

        self.init_display()

    def init_display(self):

        self.hFig = plt.figure()
        self.hAxes = self.hFig.add_subplot(111, autoscale_on=True, xlim=(-5, 5), ylim=(-1, 4))

        self.hTimeObj_str = 'time = %.1fs'
        self.hTimeObj = self.hAxes.text(0.05, 0.9, '0.0s', transform=self.hAxes.transAxes)

        X0 = np.zeros((self.Xdim,1))

        # add car wheel
        xy = self.hCar.wheel.get_pos(X0)
        r = self.hCar.wheel.radius
        face_color = self.hCar.wheel.color
        self.hWheel = mpatches.Circle(xy,r, fc = face_color,zorder=1)
        self.hAxes.add_patch(self.hWheel)

        # add upper suspension
        xy = self.hCar.suspension_upper.get_pos(X0)
        width = self.hCar.suspension_upper.w
        height = self.hCar.suspension_upper.len
        face_color = self.hCar.suspension_upper.color
        self.hUpperSuspension = mpatches.Rectangle(xy,width,height, fc = face_color,zorder=3)
        self.hAxes.add_patch(self.hUpperSuspension)

        # add lower suspension
        xy = self.hCar.suspension_lower.get_pos(X0)
        width = self.hCar.suspension_lower.w
        height = self.hCar.suspension_lower.len
        face_color = self.hCar.suspension_lower.color
        self.hLowerSuspension = mpatches.Rectangle(xy,width,height, fc = face_color,zorder=2)
        self.hAxes.add_patch(self.hLowerSuspension)

        # add car body
        xy = self.hCar.body.get_pos(X0)
        width = self.hCar.body.w
        height = self.hCar.body.len
        face_color = self.hCar.body.color
        self.hBody = mpatches.Rectangle(xy,width,height, fc = face_color,zorder=1)
        self.hAxes.add_patch(self.hBody)

        # add laser scanner
        xy = self.hCar.laser_scanner.get_pos(X0)
        width = self.hCar.laser_scanner.w
        height = self.hCar.laser_scanner.len
        face_color = self.hCar.laser_scanner.color
        self.hLaserScanner = mpatches.Rectangle(xy,width,height, fc = face_color,zorder=1)
        self.hAxes.add_patch(self.hLaserScanner)

        # add ground
        x0 = self.hCar.ground.x0
        xf = self.hCar.ground.xf
        ds = self.hCar.ground.ds

        ground_x = np.arange(x0,5,ds)
        ground_z = np.zeros(len(ground_x))

        self.hGround = lns.Line2D(ground_x,ground_z)

      #  self.hAxes.figure.canvas.draw()

        plt.axis('equal')
        plt.axis('off')
      #  plt.tight_layout()

        plt.draw()
        plt.pause(0.1)


    def render(self,frame,state,action,reward,t):

        if not bool(frame % skip_frames):

            X, acc, Fx, u, z_ground = self.hCar._unwrap_state(state)

            self.hTimeObj.set_text(self.hTimeObj_str % t)

            # update car body pos:
            xy = self.hCar.body.get_pos(X)
            self.hBody.set_xy(xy)

            # update wheel pos:
            xy = self.hCar.wheel.get_pos(X)
            self.hWheel.center = xy

            # update upper suspension pos:
            xy = self.hCar.suspension_upper.get_pos(X)
            self.hUpperSuspension.set_xy(xy)

            # update lower suspension pos:
            xy = self.hCar.suspension_lower.get_pos(X)
            self.hLowerSuspension.set_xy(xy)

            # update laser scanner pos:
            xy = self.hCar.laser_scanner.get_pos(X)
            self.hLaserScanner.set_xy(xy)

            # update ground:
            x0 = self.hCar.ground.x0
            xf = self.hCar.ground.xf
            ds = self.hCar.ground.ds

            ground_x = np.arange(x0, 5, ds)
            ground_z = np.zeros(len(ground_x))

            self.hGround.set_data(ground_x, ground_z)

            # time
            self.hTimeObj.set_text(self.hTimeObj_str % t)


            # update figure
            plt.draw()
            plt.pause(1e-6)

    def reset_display(self):

        state0 = self.hCar._get_initial_state()
        action = 0.0
        reward = 0.0
        t = 0.0

        self.render(1, state0, action, reward, t)

        return self.hBody,self.hWheel,self.hUpperSuspension,self.hLowerSuspension,self.hLaserScanner,self.hGround,self.hTimeObj

    def animate_frame(self,i,simout):

        # Xi = simout.X[i]

        #
        # z1 = Xi[0]
        # z2 = Xi[1]
        # x = Xi[2]
        # dx = Xi[5]

        state = simout.state[i]
        action = simout.action[i]
        reward = simout.reward[i]
        t = simout.t[i]

        self.render(i, state, action, reward, t)

        return self.hBody,self.hWheel,self.hUpperSuspension,self.hLowerSuspension,self.hLaserScanner,self.hGround,self.hTimeObj

    def animate_episode(self,simout):

        ani = FuncAnimation(self.hFig, self.animate_frame, np.arange(1, len(simout.X)),
                            init_func=self.reset_display, fargs=simout, interval=25, blit=True)

        plt.show()

    def get_events(self):
        pass

    def terminate(self):
        pass
