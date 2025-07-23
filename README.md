# 🤖 강화학습 (Reinforcement Learning) 실습 저장소

충북대학교 2025년 1학기 '강화학습' 과목을 수강하며 진행한 모든 실습 코드와 결과물을 아카이빙하는 저장소입니다. 강화학습의 기초 이론부터 주요 알고리즘까지 주제별로 디렉터리를 나누어 체계적으로 정리했습니다.

## 📚 과목 정보 (Course Information)

- **과목명**: 강화학습 (Reinforcement Learning)
- **수강 학기**: 2025년 1학기
- **주요 사용 언어**: `Python`
- **핵심 라이브러리**: `Gymnasium`, `PyTorch`, `NumPy`, `Matplotlib`

---

## 📂 실습 내용 (Lab Contents)

각 디렉터리는 강화학습의 핵심적인 알고리즘 또는 개념을 다루고 있습니다. 제목을 클릭하면 해당 실습 폴더로 이동합니다.

### 1. Week01_MDP-and-Bellman-Eq
- **설명**: 강화학습의 수학적 기반이 되는 마르코프 결정 과정(MDP)을 이해하고, 최적의 가치 함수를 찾기 위한 벨만 방정식을 학습합니다.
- **주요 개념**: `Agent`, `Environment`, `State`, `Action`, `Reward`, `Policy`, `Value Function`, `Bellman Equation`

### 2. Week02_Dynamic-Programming
- **설명**: 환경의 모델을 완벽히 아는 상황에서 최적 정책을 찾는 동적 프로그래밍 기법을 실습합니다. 정책 이터레이션과 가치 이터레이션을 직접 구현했습니다.
- **주요 개념**: `Dynamic Programming`, `Policy Iteration`, `Value Iteration`, `Model-Based RL`

### 3. Week03_Monte-Carlo-Methods
- **설명**: 모델 없이, 실제 경험(에피소드)을 통해서만 학습하는 몬테카를로 예측 및 제어 기법을 학습합니다.
- **주요 개념**: `Model-Free RL`, `Monte Carlo (MC)`, `On-Policy`, `Off-Policy`, `Episodic Task`

### 4. Week04_Temporal-Difference
- **설명**: 몬테카를로와 동적 프로그래밍의 장점을 결합한 시간차 학습(TD)을 실습합니다. 대표적인 알고리즘인 SARSA와 Q-Learning을 구현하고 비교했습니다.
- **주요 개념**: `Temporal-Difference (TD) Learning`, `SARSA`, `Q-Learning`, `Bootstrapping`

### 5. Week05_Deep-Q-Network
- **설명**: 거대한 상태 공간 문제를 해결하기 위해 신경망으로 가치 함수를 근사하는 DQN을 구현합니다. Experience Replay와 Target Network 기법의 중요성을 학습했습니다.
- **주요 개념**: `Function Approximation`, `Deep Q-Network (DQN)`, `Experience Replay`, `Target Network`

### 6. Week06_Policy-Gradient
- **설명**: 가치 함수가 아닌 정책 자체를 직접 파라미터화하여 최적화하는 정책 경사 기법을 학습합니다. 대표적인 알고리즘인 REINFORCE를 구현했습니다.
- **주요 개념**: `Policy Gradient`, `REINFORCE Algorithm`, `Baseline`

### 7. Week07_Actor-Critic
- **설명**: 정책 기반(Actor)과 가치 기반(Critic) 방법을 모두 사용하는 Actor-Critic 알고리즘을 실습합니다. A2C(Advantage Actor-Critic) 모델을 구현하며 학습 안정성을 높이는 방법을 학습했습니다.
- **주요 개념**: `Actor-Critic`, `Advantage Function`, `A2C`

---

## 🚀 실행 방법 (How to Run)

각 실습은 독립적인 스크립트로 구성되어 있습니다.

1.  프로젝트 실행에 필요한 라이브러리를 설치합니다. 가상 환경 구성을 권장합니다.
    ```bash
    pip install -r requirements.txt
    ```
    > **`requirements.txt` 예시:**
    > ```txt
    > gymnasium[classic_control]
    > torch
    > numpy
    > matplotlib
    > ```

2.  원하는 실습의 디렉터리로 이동합니다.
    ```bash
    cd <실습_폴더명>  # 예: cd Week04_Temporal-Difference
    ```

3.  폴더 내의 파이썬 스크립트(`*.py`)를 실행하여 학습을 진행하고 결과를 확인합니다.

## 📝 정리 및 후기

'강화학습' 과목을 통해 불확실한 환경 속에서 에이전트가 시행착오를 겪으며 최적의 의사결정 전략을 학습하는 과정을 직접 구현해볼 수 있었습니다. MDP부터 시작해 DQN, 정책 경사에 이르기까지 강화학습의 핵심적인 아이디어들을 코드로 옮기면서 이론에 대한 이해를 심화시킬 수 있었습니다.

---
