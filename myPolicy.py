import numpy as np
import car1D
from car1D import Car1D as Env

MaxAcc = np.array(3.0)
MaxVel = np.array(5.0)
MinVel = np.array(1.0)
RoughnessDeaccFactor = np.array(0.1)

# MaxAcc = np.array(300.0)
# MaxVel = np.array(50.0)
# MinVel = np.array(1.0)
# RoughnessDeaccFactor = np.array(0.0)


class VelPlanner:

    def __init__(self):

        self.dt = car1D.dt
        self.Deacc_increment = np.array(2 * self.dt)
        self.Acc_increment = np.array(0.25 * self.dt)
        self.vel_prev = np.array(0.0)


    def reset(self):

        self.vel_prev = np.array(0.0)

    def get_action(self, observation):

        dx, acc, u, z_ground = Env._unwrap_observation(observation)

        acc_abs = abs(acc)
        GroundStd = np.std(z_ground)
        DeaccRoughness = GroundStd * RoughnessDeaccFactor

        if acc_abs > MaxAcc:
            vel = self.vel_prev - self.Deacc_increment - DeaccRoughness
            vel = max(vel, MinVel / 2.0)
        else:
            vel = np.array(self.vel_prev + self.Acc_increment - DeaccRoughness)
            vel = max(min(vel, MaxVel), MinVel)

        self.vel_prev = vel

        return np.array([vel])
