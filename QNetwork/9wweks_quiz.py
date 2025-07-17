import numpy as np
import matplotlib  # Renderer에서 사용될 수 있으므로 명시적 임포트 권장
import matplotlib.pyplot as plt  # main 함수에서 직접 사용
import random
from collections import deque, namedtuple  # 현재 코드에서는 직접 사용 안 함 (Replay Buffer 등 확장 시 필요)

# 가정: DeZero 라이브러리가 설치되어 있고 다음처럼 임포트 가능
# 실제 DeZero 프로젝트 구조에 따라 임포트 경로는 달라질 수 있습니다.
try:
    import dezero
    from dezero import Model, optimizers, Variable  # 필요한 DeZero 구성 요소들
    import dezero.functions as F
    import dezero.layers as L

    DEZERO_AVAILABLE = True
    no_grad_context = dezero.no_grad  # DeZero의 no_grad 컨텍스트 사용
    print("DeZero 라이브러리를 성공적으로 임포트했습니다.")
except ImportError:
    print("경고: DeZero 라이브러리를 찾을 수 없습니다. 이 코드는 DeZero 없이 실행되지 않습니다.")
    print("DeZero 부분을 제외하고 GridWorld 및 기본 로직만 정의됩니다.")
    DEZERO_AVAILABLE = False


    # DeZero 클래스 및 함수들을 위한 Placeholder 정의 (실행은 안됨)
    class Model:
        def __init__(self, *args, **kwargs): pass

        def __call__(self, *args, **kwargs): return None  # 실제 Variable 반환해야 함

        def cleargrads(self, *args, **kwargs): pass

        def params(self, *args, **kwargs): return []


    class Layer:
        def __init__(self, *args, **kwargs): pass


    class Linear(Layer):
        def __init__(self, out_size, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.out_size = out_size


    class Function:
        def __call__(self, *args, **kwargs): return None  # 실제 Variable 반환해야 함


    class Variable:
        def __init__(self, data, *args, **kwargs):
            self.data = data  # data는 NumPy 배열

        @property
        def ndim(self):
            return self.data.ndim if hasattr(self.data, 'ndim') else 0

        @property
        def shape(self):
            return self.data.shape if hasattr(self.data, 'shape') else ()

        def reshape(self, *shape):
            return Variable(self.data.reshape(*shape))  # 새 Variable 반환

        def backward(self, *args, **kwargs):
            pass

        # Variable 간의 연산자 오버로딩 (매우 간략화된 예시)
        def __add__(self, other):
            if isinstance(other, Variable): return Variable(self.data + other.data)
            return Variable(self.data + other)

        def __radd__(self, other):
            return self.__add__(other)

        def __mul__(self, other):
            if isinstance(other, Variable): return Variable(self.data * other.data)
            return Variable(self.data * other)

        def __rmul__(self, other):
            return self.__mul__(other)

        def __sub__(self, other):
            if isinstance(other, Variable): return Variable(self.data - other.data)
            return Variable(self.data - other)

        def __rsub__(self, other):
            return Variable(other - self.data)

        def __getitem__(self, slices):
            return Variable(self.data[slices])  # 슬라이싱 지원


    class optimizers:
        class SGD:
            def __init__(self, lr=0.01): self.lr = lr

            def setup(self, model): self.target = model

            def update(self): pass  # 실제로는 model.params()를 순회하며 업데이트


    class F:  # DeZero Functions Placeholder
        @staticmethod
        def relu(x_var): return Variable(np.maximum(0, x_var.data))

        @staticmethod
        def mean_squared_error(x0_var, x1_var):
            err = np.mean((x0_var.data - x1_var.data) ** 2)
            return Variable(np.array(err, dtype=np.float32))

        @staticmethod
        def sum(x_var, axis=None, keepdims=False):
            return Variable(np.sum(x_var.data, axis=axis, keepdims=keepdims))

        @staticmethod
        def reshape(x_var, shape): return Variable(x_var.data.reshape(shape))

        @staticmethod
        def get_item(x_var, slices): return Variable(x_var.data[slices])

        @staticmethod
        def max(x_var, axis=None, keepdims=False):  # F.max Placeholder
            return Variable(np.max(x_var.data, axis=axis, keepdims=keepdims))


    class dezero_utils:
        class no_grad:  # Placeholder context manager
            def __enter__(self): pass

            def __exit__(self, type, value, traceback): pass


    no_grad_context = dezero_utils.no_grad

# Renderer 클래스를 별도 파일에서 임포트
try:
    from quiz_render import Renderer

    RENDERER_AVAILABLE = True
    print("gridworld_render.Renderer를 성공적으로 임포트했습니다.")
except ImportError:
    print("경고: gridworld_render.py 파일을 찾을 수 없거나 Renderer 클래스가 없습니다.")
    print("시각화 기능을 사용하려면 해당 파일과 클래스가 필요합니다.")
    RENDERER_AVAILABLE = False


    class Renderer:  # Placeholder Renderer
        def __init__(self, *args, **kwargs): pass

        def render_q(self, *args, **kwargs): print("Placeholder Renderer: render_q 호출됨 (기능 없음)")

        def render_v(self, *args, **kwargs): print("Placeholder Renderer: render_v 호출됨 (기능 없음)")

# 전역 변수로 HEIGHT, WIDTH 설정
HEIGHT = 5
WIDTH = 5


# ##################################################################
# GridWorld 클래스 정의
# ##################################################################
class GridWorld:
    def __init__(self, height=HEIGHT, width=WIDTH):
        self.height = height
        self.width = width
        self.action_space_size = 4  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.actions = list(range(self.action_space_size))

        self.reward_map = np.array(
            [[0, 0, 0, -1.0, 1.0],
             [0, 0, 0, 0, 0],
             [0, None, None, None, 0],
             [0, 0, 0, -1.0, 0],
             [0, 0, 0, 0, 0]], dtype=object
        )
        self.goal_state_coord = (0, 4)
        self.start_state_coord = (4, 0)

        self.wall_coords = []
        for r_idx in range(self.height):
            for c_idx in range(self.width):
                if self.reward_map[r_idx, c_idx] is None:
                    self.wall_coords.append((r_idx, c_idx))

        self.agent_pos = self.start_state_coord  # (row, col) 튜플

    def reset(self):
        self.agent_pos = self.start_state_coord
        return self.agent_pos  # (row, col) 튜플 반환

    def step(self, action_idx):
        action_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
        move = action_moves[action_idx]

        current_r, current_c = self.agent_pos
        next_r, next_c = current_r + move[0], current_c + move[1]

        if not (0 <= next_r < self.height and 0 <= next_c < self.width) or \
                (next_r, next_c) in self.wall_coords:
            next_r, next_c = current_r, current_c

        self.agent_pos = (next_r, next_c)

        reward_val = 0.0
        if self.reward_map[next_r, next_c] is not None:
            reward_val = float(self.reward_map[next_r, next_c])

        done = (self.agent_pos == self.goal_state_coord)

        return self.agent_pos, reward_val, done

    def render_q_custom(self, q_dict_for_render, print_value=True, show_greedy_policy=True):
        if RENDERER_AVAILABLE:
            renderer = Renderer(self.reward_map, self.goal_state_coord, self.wall_coords, self.start_state_coord)
            renderer.render_q(q_dict_for_render, print_value=print_value, show_greedy_policy=show_greedy_policy)
        else:
            print("Renderer를 사용할 수 없어 render_q_custom을 실행할 수 없습니다.")

    def render_v_custom(self, v_dict=None, policy_dict=None, print_value=True):
        if RENDERER_AVAILABLE:
            renderer = Renderer(self.reward_map, self.goal_state_coord, self.wall_coords, self.start_state_coord)
            renderer.render_v(v_dict, policy_dict, print_value=print_value)
        else:
            print("Renderer를 사용할 수 없어 render_v_custom을 실행할 수 없습니다.")


# ##################################################################
# 상태 변환 함수 (원-핫 인코딩)
# ##################################################################
def one_hot(state_coord):  # state_coord는 (row, col) 튜플
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    r, c = state_coord
    idx = WIDTH * r + c
    vec[idx] = 1.0
    return vec


# ##################################################################
# DeZero Q-Network (`QNet` 클래스)
# ##################################################################
if DEZERO_AVAILABLE:
    class QNet(Model):  # DeZero Model 상속
        def __init__(self, action_size):
            super().__init__()
            # DeZero Linear 레이어는 입력 크기를 첫 번째 forward 호출 시 추론하거나,
            # 명시적으로 in_size를 지정해야 할 수 있습니다.
            # 여기서는 DeZero가 in_size 없이 out_size만으로 정의 가능하다고 가정합니다.
            self.l1 = L.Linear(100)  # 은닉층 뉴런 100개
            self.l2 = L.Linear(action_size)  # 출력층 (행동 개수만큼)

        def forward(self, x_var):  # x_var는 DeZero Variable
            # 입력 x_var가 NumPy 배열이면 Variable로 변환 (안전장치)
            if not isinstance(x_var, Variable):
                x_var = Variable(x_var)

            # 입력 x_var가 1D (feature_size,) 형태라면 (batch_size, feature_size)로 reshape
            if x_var.ndim == 1:
                x_var = F.reshape(x_var, (1, -1))  # 배치 차원 추가

            h = F.relu(self.l1(x_var))
            y = self.l2(h)
            return y
else:  # DeZero 없을 때 Placeholder QNet (실행 안됨)
    class QNet(Model):  # Model Placeholder 상속
        def __init__(self, action_size):
            super().__init__()
            self.action_size = action_size
            print("경고: DeZero QNet - 실제 DeZero 레이어가 없어 기능하지 않습니다.")

        def forward(self, x_var_np_or_var):  # NumPy 또는 Placeholder Variable 입력 가정
            data_np = x_var_np_or_var.data if isinstance(x_var_np_or_var, Variable) else x_var_np_or_var
            batch_size = data_np.shape[0] if data_np.ndim > 1 else 1
            # Placeholder Variable 반환
            return Variable(np.random.rand(batch_size, self.action_size).astype(np.float32))


# ##################################################################
# Q-Learning 에이전트 (`QLearningAgent` 클래스)
# ##################################################################
class QLearningAgent:
    def __init__(self, env,
                 learning_rate=0.01,
                 gamma=0.9,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_steps=30000):  # 에피소드 수가 아닌 스텝 수 기준이 더 일반적

        self.env = env
        self.action_size = env.action_space_size
        self.state_feature_size = env.height * env.width

        self.gamma = gamma
        self.lr = learning_rate

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        # 선형 감소를 위한 값. 매 스텝(get_action 호출 시)마다 이만큼 감소
        if epsilon_decay_steps > 0:
            self.epsilon_decay_val = (epsilon_start - epsilon_end) / epsilon_decay_steps
        else:  # 즉시 최소값으로 (또는 decay 없음)
            self.epsilon_decay_val = 0
            if epsilon_decay_steps == 0: self.epsilon = epsilon_end

        self.steps_done = 0  # 엡실론 감소용 스텝 카운터

        if DEZERO_AVAILABLE:
            self.qnet = QNet(self.action_size)
            self.optimizer = optimizers.SGD(lr=self.lr)
            self.optimizer.setup(self.qnet)
        else:
            self.qnet = QNet(self.action_size)  # Placeholder QNet
            self.optimizer = None  # Placeholder Optimizer
            print("경고: QLearningAgent - DeZero optimizer가 없어 실제 학습이 불가능합니다.")

    def get_action(self, state_coord):  # state_coord는 (row,col) 튜플
        # 매 get_action 호출 시 epsilon 감소 (또는 train_step 후에 감소)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_val
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
        self.steps_done += 1

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            if not DEZERO_AVAILABLE or self.qnet is None:
                return np.random.choice(self.action_size)

            state_one_hot_np = one_hot(state_coord)
            state_var = Variable(state_one_hot_np.reshape(1, -1))

            with no_grad_context():  # 추론 시에는 그래디언트 계산 불필요
                qs_var = self.qnet(state_var)

            qs_np = qs_var.data  # Variable에서 NumPy 배열 (.data) 추출
            return np.argmax(qs_np[0])  # 배치 차원 [0] 제거 후 argmax

    def update(self, state_coord, action, reward, next_state_coord, done):
        if not DEZERO_AVAILABLE or self.qnet is None or self.optimizer is None:
            return 0.0

        state_var = Variable(one_hot(state_coord).reshape(1, -1))
        next_state_var = Variable(one_hot(next_state_coord).reshape(1, -1))

        q_values_s_all = self.qnet(state_var)  # shape: (1, action_size) Variable

        # 선택된 행동의 Q(s,a) 값 (DeZero Variable 슬라이싱 사용)
        # 결과는 shape: (1,1) Variable (DeZero 슬라이싱 방식에 따라 다를 수 있음)
        q_s_a = q_values_s_all[:, action]

        # 타겟 Q 값 계산
        if done:
            target_q_val_scalar = np.array(reward, dtype=np.float32)
        else:
            with no_grad_context():
                q_values_s_prime_all = self.qnet(next_state_var)

            # 다음 상태의 최대 Q 값 (max_a' Q(s', a'))
            # DeZero의 F.max 함수 또는 NumPy의 .max() 사용
            if hasattr(F, 'max') and callable(F.max):  # F.max 사용 가능하면 사용
                next_q_max_var = F.max(q_values_s_prime_all, axis=1, keepdims=True)  # shape: (1,1) Variable
                target_q_val_scalar = reward + self.gamma * next_q_max_var.data[0, 0]  # .data[0,0]으로 스칼라 값 추출
            else:  # F.max 없으면 NumPy로 계산
                next_q_max_np = q_values_s_prime_all.data.max(axis=1, keepdims=True)  # shape: (1,1) NumPy 배열
                target_q_val_scalar = reward + self.gamma * next_q_max_np[0, 0]  # [0,0]으로 스칼라 값 추출

        # target_q_val도 (1,1) 형태의 Variable로 만듦
        target_q_val = Variable(np.array([[target_q_val_scalar]], dtype=np.float32))

        loss = F.mean_squared_error(q_s_a, target_q_val)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        # 손실 값을 Python 스칼라 float으로 반환
        return loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)


# ##################################################################
# 메인 학습 루프
# ##################################################################
def main():
    if not DEZERO_AVAILABLE:
        print("실제 DeZero 학습을 진행하려면 DeZero 라이브러리를 올바르게 설정해야 합니다.")
        print("현재는 Placeholder 로직으로 실행됩니다.")

    env = GridWorld()
    agent = QLearningAgent(env,
                           learning_rate=0.01,
                           gamma=0.9,
                           epsilon_start=1.0,
                           epsilon_end=0.01,
                           epsilon_decay_steps=30000)  # 스텝 수 기준 엡실론 감소

    episodes = 1000
    max_steps_per_episode = 200  # 한 에피소드당 최대 스텝 수
    loss_history = []
    reward_history = []

    print(f"DQN Training with DeZero (or Placeholders) for {episodes} episodes...")
    print(
        f"Parameters: lr={agent.lr}, gamma={agent.gamma}, epsilon_start={agent.epsilon}, epsilon_decay_steps={agent.epsilon_decay_val * agent.steps_done if agent.epsilon_decay_val != 0 else 'N/A'}")

    for episode in range(episodes):
        state_coord = env.reset()  # (row, col) 튜플
        total_loss_episode = 0.0
        total_reward_episode = 0.0
        num_steps_in_episode = 0

        for step in range(max_steps_per_episode):
            action = agent.get_action(state_coord)  # (row,col) 전달
            next_state_coord, reward, done = env.step(action)

            loss_val = agent.update(state_coord, action, reward, next_state_coord, done)

            total_loss_episode += loss_val
            total_reward_episode += reward
            num_steps_in_episode += 1
            state_coord = next_state_coord

            if done:
                break

        average_loss_episode = total_loss_episode / num_steps_in_episode if num_steps_in_episode > 0 else 0
        loss_history.append(average_loss_episode)
        reward_history.append(total_reward_episode)

        if (episode + 1) % 100 == 0:  # 100 에피소드마다 로그 출력
            print(
                f"Episode: {episode + 1}/{episodes}, Steps: {num_steps_in_episode}, Avg Loss: {average_loss_episode:.4f}, Total Reward: {total_reward_episode}, Epsilon: {agent.epsilon:.3f}")

    print("Training finished.")

    # 에피소드별 손실 및 보상 추이 시각화
    if RENDERER_AVAILABLE:  # Matplotlib 사용 가능 시
        fig_summary, axs_summary = plt.subplots(1, 2, figsize=(12, 5))
        axs_summary[0].plot(loss_history)
        axs_summary[0].set_title("Average Loss per Episode")
        axs_summary[0].set_xlabel("Episode");
        axs_summary[0].set_ylabel("Average Loss")
        axs_summary[0].grid(True)

        axs_summary[1].plot(reward_history)
        axs_summary[1].set_title("Total Reward per Episode")
        axs_summary[1].set_xlabel("Episode");
        axs_summary[1].set_ylabel("Total Reward")
        axs_summary[1].grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print("Matplotlib 또는 Renderer가 없어 학습 결과 그래프를 표시할 수 없습니다.")

    print("\n학습된 Q-Network로부터 Q 값 및 정책 시각화 중...")
    q_table_for_render = {}
    if DEZERO_AVAILABLE and RENDERER_AVAILABLE and agent.qnet is not None:
        for r_idx in range(env.height):
            for c_idx in range(env.width):
                state_c = (r_idx, c_idx)
                if state_c in env.wall_coords: continue  # 벽은 건너뜀

                state_one_hot = one_hot(state_c)
                # DeZero의 no_grad 컨텍스트 사용
                with no_grad_context():
                    state_v = Variable(state_one_hot.reshape(1, -1))  # 배치 차원 추가
                    q_vals_v = agent.qnet(state_v)  # QNet 호출

                q_vals_np = q_vals_v.data[0]  # (action_size,) 형태의 NumPy 배열로 변환

                for act_idx in range(env.action_space_size):
                    q_table_for_render[(state_c, act_idx)] = q_vals_np[act_idx]

        # GridWorld 클래스에 정의된 render_q_custom 사용
        env.render_q_custom(q_table_for_render, print_value=True, show_greedy_policy=True)
    else:
        print("DeZero 또는 Renderer를 사용할 수 없거나 QNet이 학습되지 않아 최종 시각화를 할 수 없습니다.")


if __name__ == '__main__':
    main()