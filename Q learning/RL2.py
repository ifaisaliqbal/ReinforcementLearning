import numpy as np
import MDP

class RL2:
    def __init__(self,mdp,sampleReward):

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def sampleSoftmaxPolicy(self,policyParams,state):

        softmaxPolicy = np.exp(policyParams[:,state])/np.sum(np.exp(policyParams[:,state]))
        cumProb = np.cumsum(softmaxPolicy)
        action = np.where(cumProb >= np.random.rand(1))[0][0]
        return action

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

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):

        cum_rewards = np.zeros((nEpisodes))

        cumActProb = np.cumsum(np.ones(self.mdp.nActions)/self.mdp.nActions)
        freq = np.zeros([self.mdp.nActions,self.mdp.nStates,self.mdp.nStates])
        T = defaultT
        R = initialR
        model = MDP.MDP(T,R,self.mdp.discount)
        [policy,V,_] = model.policyIteration(np.zeros(model.nStates,int))
        for episId in xrange(nEpisodes):
            state = s0
            for iterId in xrange(nSteps):

                # choose action
                if epsilon > np.random.rand(1):
                    action = np.where(cumActProb >= np.random.rand(1))[0][0]
                else: 
                    action = policy[state] 

                # sample reward and next state
                [reward,nextState]=self.sampleRewardAndNextState(state,action)
                cum_rewards[episId] += (self.mdp.discount ** iterId) * reward;

                # update counts
                freq[action,state,nextState] += 1
                asFreq = freq[action,state,:].sum()

                # update transition
                T[action,state,:] = freq[action,state,:]/asFreq

                # update reward
                R[action,state] = (reward + (asFreq-1)*R[action,state])/asFreq

                # update policy
                [policy,V,_] = model.policyIteration(policy)

                state = nextState
        return [V,policy,cum_rewards]

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps):

        cum_rewards = np.zeros((nEpisodes))
        alpha = 0.01
        policyParams = initialPolicyParams
        for episId in xrange(nEpisodes):
            discountedRewards = np.zeros(nSteps)
            actions = np.zeros(nSteps,dtype=int)
            states = np.zeros(nSteps+1,dtype=int)
            states[0] = s0
            for iterId in xrange(nSteps):

                # sample action
                actions[iterId] = self.sampleSoftmaxPolicy(policyParams, states[iterId])

                # sample reward and next state
                [reward,states[iterId+1]]=self.sampleRewardAndNextState(states[iterId],actions[iterId])
                cum_rewards[episId] += (self.mdp.discount ** iterId) * reward;

                # store discounted rewards
                discountedRewards[iterId] = reward * self.mdp.discount**iterId

            cumDiscRewards = np.cumsum(discountedRewards[::-1])[::-1]
            for iterId in xrange(nSteps):
                # update policy
                logPolicyGradient = np.zeros(self.mdp.nActions)
                logPolicyGradient[actions[iterId]] = 1
                logPolicyGradient -= np.exp(policyParams[:,states[iterId]])/np.sum(np.exp(policyParams[:,states[iterId]]))
                policyParams[:,states[iterId]] += alpha*cumDiscRewards[iterId]*logPolicyGradient
            
        return [policyParams,cum_rewards]

    def epsilonGreedyBandit(self,nIterations):

        
        rewards = np.zeros(nIterations)
        cumActProb = np.cumsum(np.ones(self.mdp.nActions)/self.mdp.nActions)
        Q = np.zeros(self.mdp.nActions)
        freq = np.zeros(self.mdp.nActions)
        for iterId in xrange(nIterations):
            
            # choose action
            if 1/(iterId+1) > np.random.rand(1):
                action = np.where(cumActProb >= np.random.rand(1))[0][0]
            else: 
                if any(freq==0):
                    action = freq.argmin(0)
                else: action = (Q/freq).argmax(0) 
                
            # sample reward
            [reward,_]=self.sampleRewardAndNextState(0,action)

            # update frequencies and values
            freq[action] += 1
            Q[action] += reward
            rewards[iterId] = reward

        empiricalMeans = Q/freq
        return [empiricalMeans, rewards]

    def thompsonSamplingBandit(self,prior,nIterations,k=1):

        rewards = np.zeros(nIterations)
        beta = np.copy(prior)
        Q = np.zeros(self.mdp.nActions)
        for iterId in xrange(nIterations):
            
            # choose action
            empiricalAverages = np.zeros(self.mdp.nActions)
            for i in xrange(self.mdp.nActions):
                empiricalAverages[i] = np.random.beta(beta[i,0],beta[i,1],k).sum()/k
            action = empiricalAverages.argmax(0)
                
            # sample reward
            [reward,_]=self.sampleRewardAndNextState(0,action)

            # update beta and values
            beta[action,:] += np.array([reward,1-reward]) 
            Q[action] += reward
            rewards[iterId] = reward

        empiricalMeans = Q/(beta-prior).sum(1)
        return [empiricalMeans, rewards]

    def UCBbandit(self,nIterations):

        rewards = np.zeros(nIterations)
        freq = np.zeros(self.mdp.nActions)
        Q = np.zeros(self.mdp.nActions)
        for iterId in xrange(nIterations):
            
            # choose action
            if any(freq==0):
                action = freq.argmin(0)
            else: 
                ucb = Q/freq + np.sqrt(2*np.log(freq.sum())/freq)
                action = ucb.argmax(0)
                
            # sample reward
            [reward,_]=self.sampleRewardAndNextState(0,action)

            # update frequencies and values
            freq[action] += 1
            Q[action] += reward
            rewards[iterId] = reward

        empiricalMeans = Q/freq
        return [empiricalMeans, rewards]

