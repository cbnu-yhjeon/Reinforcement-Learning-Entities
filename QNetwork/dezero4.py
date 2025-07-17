import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers # 옵티마이저들이 들어있다
import dezero.layers as L
import dezero.functions as F

# 데이터셋 생성
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

model = TwoLayerNet(10, 1)
optimizer = optimizers.SGD(lr).setup(model) # 옵티마이저 생성 및 모델을 옵티마이저에 등록

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y_pred, y)

    model.cleargrads()
    loss.backward()

    optimizer.update() # 옵티마이저로 매개변수 갱신
    if i % 1000 == 0:
        print(loss.data)
# 그래프로 시각화 (그림 7-12와 같음)
plt.scatter(np.array(x.data).flatten(), np.array(y.data).flatten(), s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = model(t)
# plt.plot() 줄도 추가해야 모델의 예측 선을 그릴 수 있습니다.
plt.plot(np.array(t.data).flatten(), np.array(y_pred.data).flatten(), color='r') # <-- 모델 예측선 그리기 (이전 답변에서 memoryview 오류 해결했던 코드)
plt.show()