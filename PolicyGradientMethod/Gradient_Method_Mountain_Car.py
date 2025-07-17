import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import imageio  # 동영상 생성을 위해 추가

gamma = 0.99  # 할인 계수
lr_actor = 5e-5  # 액터 학습률
lr_critic = 2e-4  # 크리틱 학습률
max_episodes = 5000  # 최대 에피소드 수 (
max_steps_per_episode = 200  # MountainCar-v0의 기본 최대 스텝
log_interval = 50  # 로그 출력 간격
entropy_coeff = 0.005  # 엔트로피 보너스 가중치
# ==================================

# 환경 설정
env_name = 'MountainCar-v0'
env = gym.make(env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


# --- Actor 네트워크 ---
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


# --- Critic 네트워크 ---
class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


def train_actor_critic():
    """Actor-Critic 에이전트를 학습시키는 함수"""
    actor = Actor(state_size, action_size)
    critic = Critic(state_size)

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

    episode_rewards = []
    print("학습 시작...")
    print(
        f"Hyperparameters: lr_actor={lr_actor}, lr_critic={lr_critic}, entropy_coeff={entropy_coeff}, max_episodes={max_episodes}")

    for episode in range(max_episodes):
        state, _ = env.reset()
        current_episode_reward = 0
        log_probs = []
        values = []
        rewards_for_episode = []
        entropies = []  # 엔트로피 저장을 위해 추가

        for step in range(max_steps_per_episode):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            action_probs = actor(state_tensor)
            m = Categorical(action_probs)
            action = m.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # --- 보상 설계 수정 ---
            adjusted_reward = reward
            if terminated and next_state[0] >= env.goal_position:  # env.goal_position은 보통 0.5
                adjusted_reward += 100  # 목표 도달 시 큰 보상

            current_episode_reward += reward  # 원래 보상 기준으로 시각화

            log_probs.append(m.log_prob(action))
            values.append(critic(state_tensor))
            rewards_for_episode.append(torch.tensor([adjusted_reward], dtype=torch.float32))
            entropies.append(m.entropy())  # 엔트로피 저장

            state = next_state
            if done:
                break

        returns = []
        R = 0
        if not done:  # 마지막 상태가 터미널이 아니면 부트스트래핑
            with torch.no_grad():  # 부트스트래핑 시 그래디언트 흐름 방지
                R = critic(torch.FloatTensor(next_state).unsqueeze(0)).item()

        # 에피소드의 각 스텝에 대한 할인된 미래 보상(Return) 계산 (뒤에서부터)
        for r in rewards_for_episode[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        # --- Returns 표준화 ---
        if len(returns) > 1:  # 데이터 포인트가 하나만 있으면 std가 0이 될 수 있음 (오류 방지)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 1e-8은 분모 0 방지

        actor_losses = []
        critic_losses = []

        for log_prob, value, R_t, entropy_term in zip(log_probs, values, returns, entropies):
            advantage = R_t - value.item()  # Advantage 계산

            # Actor 손실: -log_prob * advantage - entropy_coeff * entropy
            actor_losses.append(-log_prob * advantage - entropy_coeff * entropy_term)  # 엔트로피 보너스 추가

            # Critic 손실: (R_t - value)^2 (MSE)
            # value.squeeze()는 [1,1] 형태의 텐서를 [1] 형태로 만들어줌
            critic_losses.append(F.mse_loss(value.squeeze(), torch.tensor([R_t])))

        # 옵티마이저 스텝
        actor_optimizer.zero_grad()
        # 리스트에 있는 텐서들을 하나의 텐서로 합치고 평균내어 손실 계산
        # requires_grad가 False인 텐서가 포함될 수 있는 경우를 대비 (일반적으로는 모두 True여야 함)
        valid_actor_losses = [al for al in actor_losses if al.requires_grad]
        if valid_actor_losses:
            actor_loss = torch.stack(valid_actor_losses).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)  # 그래디언트 클리핑 (선택적)
            actor_optimizer.step()
        else:
            actor_loss = torch.tensor(0.0)  # 손실 계산할 것이 없을 경우

        critic_optimizer.zero_grad()
        # 리스트에 있는 텐서들을 하나의 텐서로 합치고 평균내어 손실 계산
        critic_loss = torch.stack(critic_losses).mean()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)  # 그래디언트 클리핑 (선택적)
        critic_optimizer.step()

        episode_rewards.append(current_episode_reward)

        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(
                f"Episode {episode + 1}/{max_episodes}, Avg Reward (last {log_interval}): {avg_reward:.2f}, Last Reward: {current_episode_reward}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

    env.close()
    print("학습 완료.")
    return actor, critic, episode_rewards


def plot_rewards(episode_rewards, filename="actor_critic_rewards_tuned.png"):
    """에피소드별 총 보상을 그래프로 그리고 저장하는 함수"""
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Total Reward per Episode')
    # 이동 평균을 함께 표시하여 추세 파악 용이
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100) / 100, mode='valid')
        plt.plot(np.arange(99, len(episode_rewards)), moving_avg, label='Moving Average (100 episodes)', color='red')
    plt.title('Total Reward per Episode (Actor-Critic - Tuned Attempt)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"보상 그래프가 {filename} 으로 저장되었습니다.")
    plt.show()


def create_policy_video(actor_model, env_name, filename="actor_critic_policy_video_tuned.mp4", max_steps=500):
    """학습된 Actor 정책을 사용하여 동영상을 생성하는 함수"""
    print(f"\n정책 실행 동영상 생성 시작 ({filename})...")
    video_env = gym.make(env_name, render_mode="rgb_array")
    state, _ = video_env.reset()
    frames = []
    done = False
    total_reward_video = 0
    step_count = 0

    actor_model.eval()  # 평가 모드로 전환 (Dropout, BatchNorm 등 비활성화)

    for _ in range(max_steps):
        frame = video_env.render()
        frames.append(frame)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():  # 그래디언트 계산 비활성화
            action_probs = actor_model(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()  # 가장 확률 높은 행동 선택 (탐험 X)

        next_state, reward, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward_video += reward
        step_count += 1

        if done:
            break

    video_env.close()

    if frames:
        # imageio를 사용하여 동영상 저장
        try:
            with imageio.get_writer(filename, fps=30, codec='libx264', ffmpeg_log_level='error') as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"동영상이 {filename} 으로 저장되었습니다. 총 보상: {total_reward_video}, 스텝: {step_count}")
        except Exception as e:
            print(f"동영상 저장 중 오류 발생 (libx264): {e}")
            print("imageio 또는 ffmpeg 관련 문제가 있을 수 있습니다. 'ffv1' 코덱으로 재시도합니다.")
            try:
                with imageio.get_writer(filename, fps=30, codec='ffv1', ffmpeg_log_level='error') as writer:
                    for frame in frames:
                        writer.append_data(frame)
                print(f"동영상이 {filename} (ffv1 코덱)으로 저장되었습니다. 총 보상: {total_reward_video}, 스텝: {step_count}")
            except Exception as e2:
                print(f"ffv1 코덱으로도 동영상 저장 실패: {e2}")
    else:
        print("동영상 생성을 위한 프레임이 없습니다.")


if __name__ == '__main__':
    # 시드 고정은 실험 결과 재현에 도움이 됩니다.
    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # # env.seed(seed) # gymnasium 에서는 env.reset(seed=seed) 사용 권장
    # # env.action_space.seed(seed)

    trained_actor, trained_critic, rewards_history = train_actor_critic()

    # 학습된 모델 저장 (선택 사항)
    # torch.save(trained_actor.state_dict(), "actor_mountaincar_tuned.pth")
    # torch.save(trained_critic.state_dict(), "critic_mountaincar_tuned.pth")
    # print("학습된 Actor 및 Critic 모델이 저장되었습니다.")

    plot_rewards(rewards_history)
    create_policy_video(trained_actor, env_name)
