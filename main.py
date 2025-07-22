import pygame
import torch
import numpy as np

from rescue_env import RescueGridEnv
from PPO_agent import PPOAgent
from PPO_agent import PPOAgentWithICM
from utils import plot_rewards
import time
from actor_critic_cnn import ActorCriticCNN

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"--- selected device: {device} ---")


'''
ppo + icm--- 測試結果摘要 ---
總回合數: 10
平均獎勵: -207.57 ± 72.60
成功率: 0.00%
平均探索格子數: 296.00/625 (47.36%)

ppo + lstm--- 測試結果摘要 ---
總回合數: 10
平均獎勵: -223.59 ± 40.24
成功率: 0.00%
平均探索格子數: 291.30/625 (46.61%)
'''

hp = {
    "use_icm": True,
    "total_timesteps": 50_000,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "update_timesteps": 512,
    "ppo_epochs": 10,
    "clip_epsilon": 0.4,
    "icm_lr": 3e-4,
    "beta": 0.6,
    "vf_coef": 0.5,
    "ent_coef": 0.05,
    "map_size": (25, 25),
    "num_victims": 5,
    "obstacle_ratio": 0.3,
    "view_radius": 5
}

def manual_mode():
    env = RescueGridEnv(render_mode="human", map_size=(25, 25), num_victims=5, obstacle_ratio=0.3, view_radius=5)
    
    obs, info = env.reset()
    
    print("=" * 30)
    print("Manual Control Mode Activated (Limited Vision)!")
    print("The map is shrouded in the fog of war, only explored areas are visible.")
    print("Use arrow keys to move, 'R' to reset, 'ESC' to exit.")
    print("=" * 30)
    print("Agent's Initial Vision (Observation):")
    print(obs)
    
    running = True
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_UP: action = 0
                elif event.key == pygame.K_RIGHT: action = 1
                elif event.key == pygame.K_DOWN: action = 2
                elif event.key == pygame.K_LEFT: action = 3
                elif event.key == pygame.K_r:
                    print("\n--- Env has been reset ---")
                    obs, info = env.reset()
                    print("Agent's Initial Vision (Observation):")
                    print(obs)
        
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            
            print("\n" + "-"*15)
            print(f"Action: {['Up', 'Right', 'Down', 'Left'][action]}, Reward: {reward:.2f}")
            print("Agent's Current Vision (Observation):")
            print(obs)  # Print Agent's local vision
            print(f"True Location: {info['agent_location']}, Found: {info['victims_found']}/{env.num_victims}")

            if terminated:
                print("\n*** Congratulations! All trapped victims have been found! ***")
                print("--- Environment will reset automatically ---")
                obs, info = env.reset()
    
    env.close()
    print("program closed")

def train():
    agent_class = None
    if hp["use_icm"]:
        print("--- mode: PPO + ICM ---")
        agent_class = PPOAgentWithICM
        agent_params = {
            "lr": hp["learning_rate"], "icm_lr": hp["icm_lr"], "gamma": hp["gamma"],
            "gae_lambda": hp["gae_lambda"], "clip_epsilon": hp["clip_epsilon"],
            "vf_coef": hp["vf_coef"], "ent_coef": hp["ent_coef"], "beta": hp["beta"], "ppo_epochs": hp["ppo_epochs"]
        }
    else:
        print("--- mode: PPO (標準) ---")
        agent_class = PPOAgent
        agent_params = {
            "lr": hp["learning_rate"], "gamma": hp["gamma"],
            "gae_lambda": hp["gae_lambda"], "clip_epsilon": hp["clip_epsilon"],
            "vf_coef": hp["vf_coef"], "ent_coef": hp["ent_coef"], "ppo_epochs": hp["ppo_epochs"]
        }
    
    env = RescueGridEnv(
        map_size=hp["map_size"],
        num_victims=hp["num_victims"],
        obstacle_ratio=hp["obstacle_ratio"],
        view_radius=hp["view_radius"]
    )
    
    obs_shape = (1, *env.observation_space.shape)
    action_dim = env.action_space.n
    
    agent = agent_class(obs_shape=obs_shape, action_dim=action_dim, device=device, **agent_params)
    
    state, _ = env.reset()
    agent.policy.hidden = None
    episode_ext_reward = 0

    all_episode_ext_rewards = [] 
    MAX_GRID_VAL = 5.0 
    print("--- TRAINING START ---")
    actions = []
    for timestep in range(1, hp["total_timesteps"]):
        state_normalized = state / MAX_GRID_VAL
        action, log_prob, val = agent.select_action(state)
        actions.append(action.item())
        next_state, ext_reward, terminated, truncated, _ = env.step(action.item())
        next_state_normalized = next_state / MAX_GRID_VAL

        total_reward = ext_reward
        if hp["use_icm"]:
            intrinsic_reward = agent.get_intrinsic_reward(state_normalized, next_state_normalized, action.item())
            total_reward += agent.beta * intrinsic_reward

        done = terminated or truncated
        
        agent.memory.store(
            torch.tensor(state, dtype=torch.float32), 
            action, log_prob, total_reward, done, val, 
            torch.tensor(next_state, dtype=torch.float32)
        )
        
        state = next_state
        episode_ext_reward += ext_reward
        
        if timestep % hp["update_timesteps"] == 0:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(device)
                _, last_val = agent.policy(next_state_tensor, reset_hidden=True)
            agent.update(last_val.squeeze(), done)

        if done:
            # print("actions: ", actions)
            all_episode_ext_rewards.append(episode_ext_reward)
            avg_reward = np.mean(all_episode_ext_rewards[-100:])
            print(f"Timestep: {timestep}", "/", hp["total_timesteps"], f"Ep: {len(all_episode_ext_rewards)}, Ext Reward: {episode_ext_reward:.2f}, Avg Ext Reward: {avg_reward:.2f}")
            state, _ = env.reset()
            episode_ext_reward = 0

            agent.policy.hidden = None
            actions = []

    print("--- FINISH TRAINING ---")
    
    model_name = "./model/ppo_icm_rescue.pth" if hp["use_icm"] else "./model/ppo_rescue.pth"
    icm_model_name = "icm_model.pth"
    torch.save(agent.policy.state_dict(), model_name)
    torch.save(agent.icm.state_dict(), icm_model_name)
    print(f"Model has been saved to {model_name}")

    plot_rewards(all_episode_ext_rewards, file_name="./assets/ppo_training_rewards.png")
    
    # test_agent(model_path=model_name, num_episodes=10)

    env.close()

def test_agent(model_path, num_episodes=10):
    print("\n--- START TESTING Agent ---")
    
    test_env = RescueGridEnv(
        map_size=hp["map_size"],
        num_victims=hp["num_victims"],
        obstacle_ratio=hp["obstacle_ratio"],
        view_radius=hp["view_radius"],
        render_mode="human" 
    )

    obs_shape = (1, *test_env.observation_space.shape)
    action_dim = test_env.action_space.n
    
    policy = ActorCriticCNN(obs_shape, action_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval() 

    all_rewards = []
    explored_counts = []
    success_count = 0

    actions = []
    MAX_GRID_VAL = 5.0
    mean_explore_field = 0
    for episode in range(num_episodes):
        state, _ = test_env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            test_env.render()
            # time.sleep(0.05)

            with torch.no_grad():
                state_normalized = state / MAX_GRID_VAL
                state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).unsqueeze(0)
                dist, _ = policy(state_tensor)
                print("Action probabilities:", dist.probs.cpu().numpy())
                action_tensor = dist.sample()
                action = action_tensor.item()

            actions.append(action)
            # print("action: ", action)
            state, reward, terminated, truncated, _ = test_env.step(action)
            
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1

        explored_count = np.sum(test_env._explored_map)
        explored_counts.append(explored_count)
        
        all_rewards.append(episode_reward)
        if terminated: 
            success_count += 1
        
        print(f"test episode {episode + 1}/{num_episodes}: "
              f"Reward = {episode_reward:.2f}, "
              f"Steps = {episode_steps}, "
              f"success = {'YES' if terminated else 'NO'}")
        print("actions: ", actions)
        actions = []

    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    success_rate = (success_count / num_episodes) * 100
    avg_explored = np.mean(explored_counts)
    total_cells = test_env.width * test_env.height
    explored_percentage = (avg_explored / total_cells) * 100 

    print("\n--- Test Result Summary ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Explored Cells: {avg_explored:.2f}/{total_cells} ({explored_percentage:.2f}%)")
    
    test_env.close()

if __name__ == "__main__":
    # manual_mode()

    # train()

    model_save_path = "./model/ppo_rescue.pth"
    test_agent(model_path=model_save_path, num_episodes=10)


