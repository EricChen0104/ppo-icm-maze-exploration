# Curiosity-Driven Rescue: PPO + ICM Agent for Maze Exploration

> 🚨 A smart agent navigating unknown mazes to rescue victims — guided by curiosity, not just rewards.

This project implements a **Proximal Policy Optimization (PPO)** agent augmented with an **Intrinsic Curiosity Module (ICM)**, trained to autonomously explore a maze-like environment, find hidden victims, and learn efficient navigation strategies through internal motivation.

### DEMO
![]()

---

## 🎯 Objectives

The goal is to train an intelligent agent that can:

- Efficiently explore an unknown 2D maze
- Avoid getting stuck in dead ends or revisiting known areas
- Quickly react to visible victims and rescue them

The project is designed as a stepping stone toward full **SLAM-based autonomous exploration** systems in robotics.

---

## 🧠 Core Techniques

| Module        | Method                     | Description                                       |
|---------------|----------------------------|---------------------------------------------------|
| Reinforcement Learning | PPO        | Stable, widely used policy gradient algorithm     |
| Exploration    | ICM        | Curiosity-driven intrinsic rewards based on prediction error |
| Temporal Memory| LSTM       | Helps the agent remember explored vs unexplored areas |
| Vision Input   | CNN        | Encodes local observations                        |
| Mapping        | Occupancy Grid Map (OGM) | Supports frontier-based planning and SLAM integration |

---

## 🗺️ Environment Highlights – `RescueGridEnv`

- **Grid-based environment**: customizable size, obstacle density, and visibility radius
- **Partial observability**: local vision to simulate real-world sensing
- **Dynamic victim locations**: randomized at every episode
- **Visualization**: real-time rendering with `pygame`
- **Exploration tracking**: supports visited map and pseudo-count reward shaping

---

## 🔍 Performance Snapshot

| Agent            | Avg. Explored | Success Rate | Avg. Reward |
|------------------|---------------|---------------|-------------|
| PPO (Baseline)   | 46.6%         | 0.00%         | −223.59     |
| PPO + ICM        | 47.4%         | 0.00%         | −207.57     |

> ⚠️ Although ICM improves early exploration marginally, both agents struggle with long-term planning. Upcoming work includes frontier-based shaping and hierarchical control.

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train PPO + ICM
python main.py

# Test trained agent
python test.py
```

## 📁 Project Structure
```bash
MAZE_RESCUE/
├── main.py               # Training entry point
├── test.py               # Testing script
├── PPO_agent.py          # PPO and PPO+ICM implementations
├── ICM.py                # Intrinsic Curiosity Module
├── actor_critic_cnn.py   # CNN + LSTM-based network
├── rescue_env.py         # Custom GridWorld rescue environment
├── assets/               # Visualizations and logs
└── model/                # Saved models
```

📈 What's Next
- ✅ Frontier detection + potential-based reward shaping
- ✅ Hierarchical RL (Manager-Worker subgoal architecture)
- ✅ Transition to continuous action space (v, ω)
- ✅ SLAM integration with EKF or Particle Filter
- ✅ Publication to competitions, exhibitions, or academic venues


## 🙋 About the Author
This project was developed by a high school student passionate about AI, reinforcement learning, and robotics. The long-term vision is to build a quadruped robot that can explore real-world environments, build maps, and locate victims — all learned through intelligent curiosity.

Feel free to ⭐️ the repo, fork it, or reach out if you want to collaborate!

