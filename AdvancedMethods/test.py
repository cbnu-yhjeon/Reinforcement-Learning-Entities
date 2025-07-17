import ale_py.roms  # ALE ROM 경로 등록

import gymnasium as gym

env = gym.make("ALE/Pong-v5", render_mode="human")
obs, info = env.reset()
print("환경 로딩 성공!")
env.close()
