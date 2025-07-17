import ale_py.roms
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import cv2
from gymnasium.wrappers import AtariPreprocessing as GymAtariPreprocessing # 이름 충돌을 피하기 위해 별칭 사용
from gymnasium.wrappers import FrameStackObservation as FrameStackObservation # FrameStackObservation is FrameStack in newer gym

# 하이퍼파라미터
ENV_NAME = "ALE/Pong-v5"
GAMMA = 0.99
LEARNING_RATE = 0.00025
ENTROPY_COEFF = 0.01
VALUE_LOSS_COEFF = 0.5
MAX_GRAD_NORM = 0.5
NUM_STEPS = 64
NUM_EPISODES = 20000 # In this A2C setup, this is more like NUM_UPDATES or NUM_ROLLOUTS
LOG_INTERVAL = 10
RENDER_EVERY_EPISODES = 20000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AtariPreprocessing(gym.Wrapper):
    def __init__(self, env, screen_size=84):
        super().__init__(env)
        self.screen_size = screen_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(screen_size, screen_size, 1),
            dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv_out_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_h = conv_out_size(conv_out_size(conv_out_size(input_shape[1], 8, 4), 4, 2), 3, 1)
        conv_w = conv_out_size(conv_out_size(conv_out_size(input_shape[2], 8, 4), 4, 2), 3, 1)
        flattened_size = conv_h * conv_w * 64

        self.fc = nn.Linear(flattened_size, 512)
        self.actor_head = nn.Linear(512, num_actions)
        self.critic_head = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        action_logits = self.actor_head(x)
        state_value = self.critic_head(x)
        return action_logits, state_value

def main():
    env = gym.make(ENV_NAME)
    # Using the built-in GymAtariPreprocessing
    env = GymAtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False, # Keep as 0-255
        frame_skip=1 # Pong usually uses 1 or 4. Original DQN used 4.
    )
    # FrameStackObservation is now just FrameStack in gymnasium
    env = FrameStackObservation(env, 4)

    num_actions = env.action_space.n
    input_shape = env.observation_space.shape

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space} (num: {num_actions})")

    model = ActorCritic(input_shape, num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    episode_rewards = deque(maxlen=100)
    all_episode_rewards = []

    # Lists to store loss values
    all_total_losses = []
    all_actor_losses = []
    all_critic_losses = []
    all_entropy_losses = []

    print(f"Starting training for {NUM_EPISODES} update steps (rollouts)...")

    for episode_or_update_step in range(NUM_EPISODES): # Each iteration is one rollout and one update
        # state is a LazyFrames object, convert to numpy array for PyTorch
        # The reset for the *true* environment episode happens here if the previous one ended,
        # or if it's the very first step.
        # In this A2C structure, `env.reset()` is implicitly handled by the loop structure
        # if an episode truly ends. But for each of NUM_EPISODES iterations,
        # we perform a rollout of NUM_STEPS.
        # For simplicity and common A2C structure, let's assume `state` persists across
        # rollouts if an episode hasn't ended. The provided code resets state at the start of each "episode"
        # which means each "episode" is a segment from the start of an actual game episode.

        # The provided code resets the environment at the beginning of each of the NUM_EPISODES iterations.
        # This means 'episode_reward' will be the reward for that segment (up to NUM_STEPS or end of actual game episode).
        state, info = env.reset()
        current_rollout_reward = 0 # Renamed for clarity
        done_rollout = False # Flag for end of current rollout/segment

        log_probs_list = []
        values_list = []
        rewards_list = []
        entropy_list = []

        # N-step rollout (data collection for one update)
        for step_in_rollout in range(NUM_STEPS):
            state_np = np.array(state) # Convert LazyFrames to numpy array
            state_tensor = torch.FloatTensor(state_np / 255.0).unsqueeze(0).to(device)

            action_logits, state_value = model(state_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()

            next_state, reward, terminated, truncated, info = env.step(action.item())
            # `done_rollout` becomes true if the *actual environment episode* ends.
            done_rollout = terminated or truncated

            rewards_list.append(reward)
            values_list.append(state_value)
            log_probs_list.append(log_prob)
            entropy_list.append(entropy)

            state = next_state
            current_rollout_reward += reward

            if done_rollout: # If true episode ended during this rollout
                break

        # --- 학습 단계 ---
        R = 0
        if not done_rollout: # If rollout ended because NUM_STEPS was reached (not because episode terminated)
            next_state_np = np.array(next_state) # Convert LazyFrames
            next_state_tensor = torch.FloatTensor(next_state_np).unsqueeze(0).to(device)
            _, R_tensor = model(next_state_tensor)
            R = R_tensor.item() # .detach() is implicitly handled as R_tensor is not part of graph further

        returns = []
        for i in reversed(range(len(rewards_list))):
            R = rewards_list[i] + GAMMA * R
            returns.insert(0, R)

        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        values_tensor = torch.cat(values_list).squeeze()
        advantages = returns_tensor - values_tensor.detach()

        log_probs_tensor = torch.cat(log_probs_list)
        actor_loss = -(log_probs_tensor * advantages).mean()
        critic_loss = F.mse_loss(values_tensor, returns_tensor)
        entropy_loss = -torch.stack(entropy_list).mean()
        loss = actor_loss + VALUE_LOSS_COEFF * critic_loss + ENTROPY_COEFF * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        # Store losses
        all_total_losses.append(loss.item())
        all_actor_losses.append(actor_loss.item())
        all_critic_losses.append(critic_loss.item())
        all_entropy_losses.append(entropy_loss.item())

        episode_rewards.append(current_rollout_reward)
        all_episode_rewards.append(current_rollout_reward) # This is rollout reward

        if episode_or_update_step % LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            print(f"Update Step {episode_or_update_step}/{NUM_EPISODES} | "
                  f"Rollout Reward: {current_rollout_reward:.2f} | "
                  f"Avg Reward (last 100 rollouts): {avg_reward:.2f} | "
                  f"Loss: {loss.item():.4f} (Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f}, Entropy: {entropy_loss.item():.4f})")

        if RENDER_EVERY_EPISODES > 0 and episode_or_update_step % RENDER_EVERY_EPISODES == 0 and episode_or_update_step > 0 :
            print(f"Rendering episode at update step {episode_or_update_step}")
            # For rendering, use a separate env instance with human render_mode
            render_env = gym.make(ENV_NAME, render_mode="human")
            render_env = GymAtariPreprocessing(render_env, screen_size=84, grayscale_obs=True, scale_obs=False, frame_skip=1)
            render_env = FrameStackObservation(render_env, 4)

            s_render, _ = render_env.reset()
            terminated_render, truncated_render = False, False
            render_ep_reward = 0
            while not (terminated_render or truncated_render):
                s_render_np = np.array(s_render)
                s_tensor_render = torch.FloatTensor(s_render_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    act_logits_render, _ = model(s_tensor_render)
                act_probs_render = F.softmax(act_logits_render, dim=-1)
                dist_render = Categorical(act_probs_render)
                act_render = dist_render.sample().item() # Stochastic for representative behavior

                s_next_render, r_render, terminated_render, truncated_render, _ = render_env.step(act_render)
                s_render = s_next_render
                render_ep_reward += r_render
            print(f"Rendered episode reward: {render_ep_reward}")
            render_env.close()

    env.close()
    print("Training finished.")

    # --- Plotting Results ---

    # Plot Episode Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(all_episode_rewards, label='Rollout Reward')
    if len(all_episode_rewards) >= 100:
        moving_avg = np.convolve(all_episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(all_episode_rewards)), moving_avg, label='Moving Avg (100 rollouts)')
    plt.xlabel("Update Step (Rollout Number)")
    plt.ylabel("Total Reward per Rollout")
    plt.title(f"Rollout Rewards for {ENV_NAME} (A2C)")
    plt.legend()
    plt.grid(True)
    plt.savefig("pong_a2c_rewards.png")
    plt.show()

    # Plot Losses
    plt.figure(figsize=(12, 10)) # Adjusted for 4 subplots

    plt.subplot(2, 2, 1)
    plt.plot(all_total_losses, label='Total Loss', color='red')
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(all_actor_losses, label='Actor Loss', color='green')
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Actor Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(all_critic_losses, label='Critic Loss', color='blue')
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Critic Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(all_entropy_losses, label='Entropy Loss', color='purple')
    # This is -coeff * entropy. A more negative value means higher entropy was encouraged.
    plt.xlabel("Update Step")
    plt.ylabel("Loss Value")
    plt.title("Entropy Loss (encourages exploration)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Adjust subplot params for a tight layout.
    plt.savefig("pong_a2c_losses.png")
    plt.show()

if __name__ == '__main__':
    try:
        import cv2
    except ImportError:
        print("OpenCV is not installed. Please install it using 'pip install opencv-python' for the custom AtariPreprocessing wrapper.")
        print("If using gymnasium's built-in AtariPreprocessing, cv2 might not be explicitly needed in this script if it handles its own dependencies.")
    main()