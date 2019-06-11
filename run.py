from car1D import Car1D
from myPolicy import VelPlanner
import numpy as np

env = Car1D()

n_actions = len(env.action_space.sample())
n_observations = len(env.observation_space.sample())

vel_planner = VelPlanner()

use_myPolicy = True

ob = env.reset()

t_vec = np.arange(0,env.t_final,env.dt)

for i in range(len(t_vec)*3):

    if use_myPolicy:
        a = vel_planner.get_action(ob)
    else:
        a = env.action_space.sample()

    ob, r, done, info = env.step(a)

    if done:
        print("Episode done")
        ob = env.reset()

    env.render()
