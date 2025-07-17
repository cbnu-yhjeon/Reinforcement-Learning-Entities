import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero
from dezero import optimizers
from dezero import Model
import dezero.layers as L
import dezero.functions as F
from common.gridworld import GridWorld


def one_hot(state, HEIGHT=3, WIDTH=4):
    vec = np.zeros((HEIGHT * WIDTH), dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec.reshape(1, -1)

class QNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet(100, self.action_size)
        self.optimizer = optimizers.SGD(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with dezero.no_grad():
                state_vec = one_hot(state)
                qs = self.qnet(Variable(state_vec))
            return np.array(qs.data).argmax()

    def update(self, state, action, reward, next_state, done):
        state_vec = one_hot(state)
        next_state_vec = one_hot(next_state)

        if done:
            next_q_value = 0
        else:
            with dezero.no_grad():
                 next_qs = self.qnet(Variable(next_state_vec))
                 next_q_value = np.array(next_qs.data).max(axis=1)[0]

            next_q_value = reward + self.gamma * next_q_value

        qs = self.qnet(Variable(state_vec))
        q = qs[:, action]

        target = Variable(np.array([next_q_value]))
        loss = F.mean_squared_error(target, q)
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return np.array(loss.data)

try:
    env = GridWorld()
    agent = QLearningAgent()
except NameError:
    print("GridWorld 클래스를 찾을 수 없습니다.")
    exit()

episodes = 1000
loss_history = []

print("Q-Learning 학습 시작...")
for episode in range(episodes):
    state = env.reset()
    total_loss, cnt = 0, 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        loss_data = agent.update(state, action, reward, next_state, done)

        total_loss += loss_data
        cnt += 1
        state = next_state

    average_loss = total_loss / cnt
    loss_history.append(average_loss)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes}, Avg Loss: {average_loss:.4f}")

print("Q-Learning 학습 완료.")

print("\n손실 추이 그래프 표시...")
plt.figure(figsize=(10, 5))
plt.plot(range(len(loss_history)), loss_history)
plt.xlabel('Episode')
plt.ylabel('Average Loss')
plt.title('Average Loss per Episode')
plt.grid(True)
plt.show()

print("\n학습된 Q 값 및 정책 시각화...")
Q = {}
try:
    for state in env.states():
        state_vec = one_hot(state)
        with dezero.no_grad():
            q_values_var = agent.qnet(Variable(state_vec))

        q_values_np = np.array(q_values_var.data)

        for action in env.action_space:
            Q[state, action] = float(q_values_np[0, action])

    env.render_q(Q)

except AttributeError:
    print("GridWorld 환경에 'states' 또는 'action_space' 또는 'render_q' 메서드가 정의되어 있지 않습니다.")
except Exception as e:
     print(f"Q 값 시각화 중 오류 발생: {e}")

print("스크립트 실행 완료.")