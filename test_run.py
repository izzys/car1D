import gym
env = gym.make('BipedalWalkerHardcore-v2') # try also CartPole-v0,MountainCar-v0,MsPacman-v0,Hopper-v1
env.reset()
for _ in range(1000):
    env.render()
    ob, r, done, info = env.step(env.action_space.sample()) # take a random action
    if done:
        ob = env.reset()