import numpy as np
import matplotlib.pyplot as plt


class Agent:    
    """Abstract Agent class."""
    def reset(self):
        self.rewards = []
        
    def initialize_episode(self):
        self.sum_rewards = 0
    
    def get_action(self, state):
        return None
    
    def update(self, state, action, next_state, reward):
        self.sum_rewards = self.sum_rewards + reward
    
    def finalize_episode(self):
        self.rewards.append(self.sum_rewards)
        
    def get_name(self):
        return self.__class__.__name__

    def plot_rewards(self, ylim=(-100,0)):
        plt.figure(figsize=(18,5))
        plt.plot(self.rewards)
        plt.xlabel('episode')
        plt.ylabel('sum of rewards')
        plt.ylim(ylim)
        plt.title(self.get_name())
        


import time
from IPython.display import clear_output

def run_experiment(environment, agent, n_experiments=1, max_steps=100, render=False, sleep=0.01, n_epoch_update=1000, plot_stats=False):
    # do some bookkeeping
    stats_steps = []
    stats_rewards = []
    stats_penalties = []
    stats_reward_per_step = []
    
    for n in range(n_experiments):
        # reset environment and agent for this episode
        state = environment.reset()
        agent.initialize_episode()
        
        # render environment
        if render:
            environment.render()
            clear_output(wait=True)
            time.sleep(sleep)
        elif n % n_epoch_update ==0: # print progress
            print("Run experiment : {} / {}".format(n, n_experiments))
        
        # episode loop
        steps = 0
        penalties = 0
        done = False
        while (not done) and (steps < max_steps):
            action = agent.get_action(state)
            next_state, reward, done, info = environment.step(action)
            agent.update(state, action, next_state, reward)
            state = next_state
            steps = steps + 1
            
            if render:
                environment.render()
                clear_output(wait=True)
                time.sleep(sleep)
             
            # count penalties
            if reward==-10:
                penalties += 1
            
        agent.finalize_episode()
        stats_penalties.append(penalties)
        stats_steps.append(steps)
        stats_rewards.append(agent.sum_rewards)
        stats_reward_per_step.append(agent.sum_rewards / steps)
    
    if render:
        environment.render()
        print('')
    else:
        print('Done.\n')

    print("Average reward       : {}".format(np.mean(stats_rewards)))
    print("Average #penalties   : {}".format(np.mean(stats_penalties)))
    print("Average #steps       : {}".format(np.mean(stats_steps)))
    print("Average #reward/step : {}".format(np.mean(stats_reward_per_step)))
    
    if plot_stats:
        plt.plot(stats_steps, label='#steps')
        plt.plot(stats_penalties, label='#penalties')
        plt.xlabel('epoch')
        plt.legend()
        
