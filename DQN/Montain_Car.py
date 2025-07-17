import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import imageio

LEARNING_RATE = 0.0002
GAMMA = 0.99
BUFFER_SIZE = 100000
BATCH_SIZE = 128
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQUENCY = 2000
MAX_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 500
HIDDEN_LAYER_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cpu()
        action = torch.tensor([action], dtype=torch.int64).unsqueeze(0).cpu()
        reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).cpu()
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).cpu()
        done = torch.tensor([done], dtype=torch.float32).unsqueeze(0).cpu()
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.cat(states).to(device),
                torch.cat(actions).to(device),
                torch.cat(rewards).to(device),
                torch.cat(next_states).to(device),
                torch.cat(dones).to(device))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size,
                 epsilon_start, epsilon_end, epsilon_decay, target_update_freq):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.learn_step_counter = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        current_q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)

        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

def train():
    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, HIDDEN_LAYER_SIZE, LEARNING_RATE, GAMMA,
                     BUFFER_SIZE, BATCH_SIZE, EPSILON_START, EPSILON_END,
                     EPSILON_DECAY, TARGET_UPDATE_FREQUENCY)

    episode_rewards = []
    episode_losses = []

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.store(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                total_loss += loss
                steps += 1

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        avg_loss = total_loss / steps if steps > 0 else 0
        episode_losses.append(avg_loss)

        print(f"Episode {episode+1}/{MAX_EPISODES}, Total Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}, Buffer Size: {len(agent.buffer)}")

    env.close()
    print("Training finished.")

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Total Reward per Episode during Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('total_rewards_graph.png')
    plt.show()

    max_reward = max(episode_rewards) if episode_rewards else 'N/A'
    print(f"Maximum total reward achieved: {max_reward}")


    return agent

def create_video(agent, filename="mountain_car_dqn.mp4"):
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    state, _ = env.reset()
    frames = []
    done = False
    total_reward = 0
    step_count = 0

    agent.epsilon = 0.0
    agent.q_network.eval()

    while not done and step_count < MAX_STEPS_PER_EPISODE * 2 :
        frame = env.render()
        frames.append(frame)
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        step_count += 1

    env.close()

    if frames:
        print(f"Saving video... Total reward in video: {total_reward}, Steps: {step_count}")
        with imageio.get_writer(filename, fps=30, codec='ffv1', ffmpeg_log_level='error') as writer: # 코덱 지정 예시
            for frame in frames:
                writer.append_data(frame)
        print(f"Video saved as {filename}")
    else:
        print("No frames recorded for video.")


if __name__ == '__main__':
    trained_agent = train()
    create_video(trained_agent)

    state_dim = gym.make('MountainCar-v0').observation_space.shape[0]
    action_dim = gym.make('MountainCar-v0').action_space.n
    optimal_agent = DQNAgent(state_dim, action_dim, HIDDEN_LAYER_SIZE, LEARNING_RATE, GAMMA,
                     BUFFER_SIZE, BATCH_SIZE, EPSILON_START, EPSILON_END,
                     EPSILON_DECAY, TARGET_UPDATE_FREQUENCY)
    optimal_agent.q_network.load_state_dict(torch.load('best_dqn_model.pth'))
    create_video(optimal_agent, filename="best_policy_video_2.mp4")