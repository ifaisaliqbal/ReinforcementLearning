import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):


        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):


        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):

        temperature = float(temperature)
        counts = np.zeros([self.mdp.nActions, self.mdp.nStates])
        Q = np.copy(initialQ)
        ep_rewards = []

        for episode in range(nEpisodes):
            s = np.copy(s0)
            discounted_rewards = 0.
            for step in range(nSteps):
                if np.random.uniform() < epsilon:
                    a = np.random.randint(self.mdp.nActions)
                else:
                    if temperature == 0.:
                        a = np.argmax(Q[:, s])
                    else:
                        prob_a = np.exp(Q[:, s] / temperature) / np.sum(np.exp(Q[:, s] / temperature))
                        a = np.argmax(np.random.multinomial(1, prob_a))

                r, next_s = self.sampleRewardAndNextState(s, a)
                discounted_rewards += self.mdp.discount**step * r

                counts[a, s] += 1.
                Q[a, s] += (1. / counts[a, s]) * (r + self.mdp.discount * np.amax(Q[:, next_s]) - Q[a, s])
                s = np.copy(next_s)
            ep_rewards.append(discounted_rewards)

        policy = np.argmax(Q, axis=0)

        return [Q,policy,np.array(ep_rewards)]    
