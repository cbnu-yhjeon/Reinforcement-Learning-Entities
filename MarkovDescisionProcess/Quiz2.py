import numpy as np

# 1. 환경 설정

# 상태 정의 (좌표로 표현)
states = [(i, j) for i in range(3) for j in range(4)]
states.remove((1, 1))  # 벽 제거

# 행동 정의
actions = ["UP", "DOWN", "LEFT", "RIGHT"]

# 보상 함수
rewards = {
    (0, 3): 1,
    (1, 3): -1,
}

# 할인율
gamma = 0.9

# 정책 (균일 확률)
policy = {s: {a: 0.25 for a in actions} for s in states}


# 2. 상태 전이 함수 정의
def next_state(state, action):
    """주어진 상태와 행동에 대해 다음 상태를 반환합니다."""
    i, j = state
    if action == "UP":
        next_s = (max(i - 1, 0), j)
    elif action == "DOWN":
        next_s = (min(i + 1, 2), j)
    elif action == "LEFT":
        next_s = (i, max(j - 1, 0))
    elif action == "RIGHT":
        next_s = (i, min(j + 1, 3))
    else:
        raise ValueError("Invalid action")

    # 벽 처리
    if next_s == (1, 1):
        return state  # 벽에 부딪히면 현재 상태 유지
    else:
        return next_s


# 3. 벨만 기대 방정식 계산
def calculate_state_value(policy, rewards, gamma, states):
    """주어진 정책에 대한 상태 가치를 계산합니다."""
    value = {s: 0 for s in states}  # 초기 value 값 0으로 설정

    while True:
        delta = 0
        for s in states:
            old_value = value[s]
            new_value = 0
            for a in actions:
                next_s = next_state(s, a)
                reward = rewards.get(next_s, 0)  # 다음 상태에 대한 보상 가져오기 (없으면 0)
                new_value += policy[s][a] * (reward + gamma * value[next_s])
            value[s] = new_value
            delta = max(delta, abs(new_value - old_value))
        if delta < 1e-6:  # 수렴 조건
            break
    return value


# 4. 결과 출력
value = calculate_state_value(policy, rewards, gamma, states)

for s, v in value.items():
    print(f"State {s}: Value = {v:.2f}")