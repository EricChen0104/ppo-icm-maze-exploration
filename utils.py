import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, moving_avg_window=100, file_name="ppo_rewards"):
    """繪製訓練過程的獎勵曲線"""
    plt.figure(figsize=(12, 6))
    plt.title("Training Progress: Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    # 繪製每個回合的原始獎勵
    plt.plot(rewards, label='Episode Reward', color='lightblue', alpha=0.7)
    
    # 計算並繪製滑動平均獎勵
    if len(rewards) >= moving_avg_window:
        moving_avg = np.convolve(rewards, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
        plt.plot(np.arange(moving_avg_window - 1, len(rewards)), moving_avg, 
                 label=f'Moving Average ({moving_avg_window} episodes)', color='red', linewidth=2)
                 
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name) # 儲存圖片
    plt.show()