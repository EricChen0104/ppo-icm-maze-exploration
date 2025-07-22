import torch
from actor_critic_cnn import ActorCriticCNN
from ICM import ICM_CNN
from Memory import MemoryICM, Memory
from torch.optim import Adam
import torch.nn as nn

class PPOAgent:
    def __init__(self, obs_shape, action_dim, device, lr, gamma, gae_lambda, clip_epsilon, vf_coef, ent_coef, ppo_epochs):
        self.gamma = gamma
        self.device = device
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.ppo_epochs = ppo_epochs
        
        self.policy = ActorCriticCNN(obs_shape, action_dim).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.memory = Memory()

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            dist, value = self.policy(state_tensor)  
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.squeeze(), log_prob.squeeze(), value.squeeze()

    def update(self, last_val, last_done):
        states, actions, old_log_probs, rewards, dones, vals = self.memory.get_tensors()

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        vals = vals.to(self.device)

        states = states.unsqueeze(1)
        
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - last_done
                next_val = last_val
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_val = vals[t+1]
                
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - vals[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + vals 

        for _ in range(self.ppo_epochs):
            dist, new_vals = self.policy(states)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.functional.mse_loss(new_vals.squeeze(), returns)
            
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.memory.clear()

class PPOAgentWithICM(PPOAgent):
    def __init__(self, obs_shape, action_dim, device, lr, icm_lr, gamma, gae_lambda, clip_epsilon, vf_coef, ent_coef, beta, ppo_epochs):
        super().__init__(obs_shape, action_dim, device, lr, gamma, gae_lambda, clip_epsilon, vf_coef, ent_coef, ppo_epochs)
        
        self.icm = ICM_CNN(obs_shape, action_dim).to(self.device)
        self.icm_optimizer = Adam(self.icm.parameters(), lr=icm_lr)
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ppo_epochs = ppo_epochs
        self.memory = MemoryICM()
    
    def get_intrinsic_reward(self, state, next_state, action):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)
            action_tensor = torch.tensor([action]).to(self.device)
            
            pred_next_feature, next_feature, _ = self.icm(state_tensor, next_state_tensor, action_tensor)
            
            intrinsic_reward = self.mse_loss(pred_next_feature, next_feature).item()
            
        return intrinsic_reward
    
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            dist, value = self.policy(state_tensor) 
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.squeeze(), log_prob.squeeze(), value.squeeze()

    def update(self, last_val, last_done):
        states, actions, old_log_probs, rewards, dones, vals, next_states = self.memory.get_tensors()

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        vals = vals.to(self.device)
        next_states = next_states.to(self.device)
        
        states = states.unsqueeze(1)
        next_states = next_states.unsqueeze(1)
        
        pred_next_features, true_next_features, pred_action_logits = self.icm(states, next_states, actions)
        forward_loss = self.mse_loss(pred_next_features, true_next_features)
        inverse_loss = self.ce_loss(pred_action_logits, actions.long())
        icm_loss = (1.0 - 0.2) * forward_loss + 0.2 * inverse_loss
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - (last_done if t == len(rewards) - 1 else dones[t + 1])
            next_val = last_val if t == len(rewards) - 1 else vals[t + 1]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - vals[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + vals

        for _ in range(self.ppo_epochs):
            dist, new_vals = self.policy(states)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(new_vals.squeeze(), returns)
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.memory.clear()