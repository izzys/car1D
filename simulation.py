from car1D import Car1D
# from ___visualization_matplotlib import Visualization
from myPolicy import VelPlanner
import numpy as np
import matplotlib.pyplot as plt



class Simout:

    def __init__(self):

        self.t = []

        self.action = []
        self.reward = []
        self.state = []

        self.X = []
        self.acc = []
        self.z_ground = []
        self.vel_cmd = []
        self.Fx = []
        self.u = []


class Simulation:

    def __init__(self,world_model):

        self._running = True
        self._display_surf = None

        self._env = world_model
        self._policy = VelPlanner()

        # some global parameters
        self.Animate = False

    def InitEpisode(self):

        if self.Animate:
            self._env._init_display()

        self._running = True
        self._env.reset()


    def RunEpisode(self):

        t_vec = np.arange(0,self._env.t_final+self._env.dt,self._env.dt)
        simout = Simout()
        observation = self._env._get_observation()

        for i in range(len(t_vec)):

            #   print(self.t)

            action = self._policy.get_action(observation)
            ob, reward, done, info  = self._env.step(action)
            state = self._env.state

            t = t_vec[i]

            simout.t.append(t)
            simout.action.append(action)
            simout.reward.append(reward)
            simout.state.append(state)
            X, acc, Fx, u, z_ground = self._env._unwrap_state(state)


            simout.X.append(X)
            simout.acc.append(acc)
            simout.z_ground.append(z_ground)
            simout.vel_cmd.append(action)
            simout.Fx.append(Fx)
            simout.u.append(u)

            if self.Animate:
                self._env.render()

            if done:
                break

        return simout

    def TerminateEpisode(self):
        pass

    def PlotEpisode(self,simout):

        hFig = plt.figure()
        hAxes1 = hFig.add_subplot(511, autoscale_on=True)
        hAxes2 = hFig.add_subplot(512, autoscale_on=True)
        hAxes3 = hFig.add_subplot(513, autoscale_on=True)
        hAxes4 = hFig.add_subplot(514, autoscale_on=True)
        hAxes5 = hFig.add_subplot(515, autoscale_on=True)

        hFig2 = plt.figure()
        hAxes2_1 = hFig2.add_subplot(211, autoscale_on=True)
        hAxes2_2 = hFig2.add_subplot(212, autoscale_on=True)

        z1 = []
        z2 = []
        x = []
        dx = []
        z = []

        for i in range(len(simout.X)):

            Xi = simout.X[i]

            z1.append(Xi[0,0])
            z2.append(Xi[1,0])
            x.append(Xi[2,0])
            dx.append(Xi[5,0])

            ztmp,_ = self._env.ground.get_height(Xi[2], np.array([0.0]))
            z.append(ztmp)

        r = simout.reward

        hAxes1.plot(simout.t,x ,'-', lw=2,label='x[m]')
        hAxes1.grid(True)
        hAxes1.set_title('car pos')
        hAxes1.set_ylabel('x (m)')
        hAxes1.legend()

        hAxes2.plot(simout.t,dx ,'-', lw=2,label='car velocity[m/s]')
        hAxes2.plot(simout.t, simout.vel_cmd, '--', lw=2,label='velocity cmd.[m/s] (action)')
        hAxes2.grid(True)
        hAxes2.set_title('car velocity/Vel cmd')
        hAxes2.set_ylabel('dx/dt (m/s)')
        hAxes2.legend()

        hAxes3.plot(simout.t,simout.u ,'-', lw=2,label='Control cmd.[N]')
        hAxes3.plot(simout.t, simout.Fx, '--', lw=2,label='Driving force [N]')
        hAxes3.grid(True)
        hAxes3.set_title('Control cmd(u)/Driving force(Fx)')
        hAxes3.set_ylabel('Force (N)')
        hAxes3.legend()

        hAxes4.plot(simout.t,simout.acc ,'-', lw=2,label='Vertical acc[m/s^2]')
        hAxes4.grid(True)
        hAxes4.set_title('vertical acc')
        hAxes4.set_ylabel('Acc. (m/s^2)')
        hAxes4.legend()

        hAxes5.plot(simout.t,z ,'-', lw=2,label='Ground height[m]')
        hAxes5.plot(simout.t,z2 ,'-', lw=2,label='Wheel height[m]')
        hAxes5.grid(True)
        hAxes5.set_title('Ground height,wheel height')
        hAxes5.set_ylabel('z (m)')
        hAxes5.set_xlabel('time (s)')
        hAxes5.legend()

        hAxes1.plot(simout.t,r,'-', lw=2,label='reward')
        hAxes1.plot(simout.t,np.cumsum(r) ,'-', lw=2,label='reward')

        hAxes1.grid(True)
        hAxes1.set_title('car pos')
        hAxes1.set_ylabel('x (m)')
        hAxes1.legend()

        hFig.subplots_adjust(top=0.95,bottom=0.07, left=0.07, right=0.97,
                            hspace=0.45, wspace=0.5)

        hFig2.subplots_adjust(top=0.95,bottom=0.07, left=0.07, right=0.97,
                            hspace=0.45, wspace=0.5)




        plt.show()




if __name__ == "__main__":

    model = Car1D()
    sim = Simulation(model)

    sim.Animate = True

    sim.InitEpisode()

    simout = sim.RunEpisode()

    sim.TerminateEpisode()

    sim.PlotEpisode(simout)
