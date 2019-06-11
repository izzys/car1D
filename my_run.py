from car1D import Car1D
from simulation import Simout
from simulation import Simulation
from myPolicy import VelPlanner

env = Car1D()

n_actions = len(env.action_space.sample())
n_observations = len(env.observation_space.sample())

simout = Simout()
sim = Simulation(env)
vel_planner = VelPlanner()

use_myPolicy = True
render_ON = False

ob = env.reset()
#assert env.observation_space.contains(ob), "%r (%s) invalid" % (ob, type(ob))

for i in range(5000):

    if use_myPolicy:
        a = vel_planner.get_action(ob)
    else:
        a = env.action_space.sample()

   # print(env.t)
    ob, r, done, info = env.step(a)
    if done:
        ob = env.reset()
        break

    if render_ON:
        env.render()

    state = env.state

    simout.t.append(env.t)
    simout.action.append(a)
    simout.reward.append(r)
    simout.state.append(state)

    X, acc, Fx, u, z_ground = env._unwrap_state(state)
    simout.X.append(X)
    simout.acc.append(acc)
    simout.z_ground.append(z_ground)
    simout.vel_cmd.append(a)
    simout.Fx.append(Fx)
    simout.u.append(u)

sim.PlotEpisode(simout)
