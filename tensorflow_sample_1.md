# tensorflow 사용하기
* tensorflow을 사용해서 몸무게와 키를 기준으로 성별 판정하기

## 다층 퍼셉트론 레이어을 이용해서 성별 판정하기 샘플
```python
import tensorflow as tf
# print(tf.__version__)
import tensorflow.keras.utils as utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt
import random

# 훈련데이터 준비하기
# 훈련데이터는 [weight, height]다.
x_data = []
for i in range(100):
  x_data.append([random.randint(40, 60),random.randint(140, 170)])
  x_data.append([random.randint(60, 90),random.randint(170, 200)])

# 정답데이터 준비하기
# 정답데이터는 [weight, height, gender]다.
y_data = []
for i in range(100):
  y_data.append(1)#여
  y_data.append(0)#남

# 1. 데이터셋 준비하기
X_train = np.array([x_data])

# [[weight, height], [w, h], [w, h]]를
# {
#  [w, h].
#  [w, h],
#  [w, h]
#}로 변환한다.
X_train = X_train.reshape(200,2)

Y_train = np.array(y_data)
Y_train = Y_train.reshape(200,)

# 모형클래스 객체를 생성한다.
model = Sequential()

# model.add() : 모델에 레이어를 추가한다.
# 첫번째 레이어는 입력2개[weight, height]다, 출력은 20개다. 활성화함수는 'relu'다.
model.add(Dense(20, input_dim=2, activation='relu'))
# 두번째 레이어는 출력이 10개다. 활성화함수는 'relu'다.
model.add(Dense(10, activation='relu'))
# 세번쩨 레이어는 출력이 1개[gender]다, 활성화함수는 'sigmoid'다.
model.add(Dense(1, activation='sigmoid'))

# model.compile() : 모형을 완성한다.
# loss는 손실함수를 설정한다.
# optimizer은 최적화 알고리즘을 설정한다.
# metrics는 트레이닝 단계에서 사용할 평가지표(성능기준)을 설정한다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit() : 훈련을 실시한다.
# X_train은 훈련용 문제다
# Y_train은 훈련용 정답이다.
# epochs는 훈련횟수다.
# batch_size는 가중치를 갱신할 샘플으 갯수를 지정한다.
# validation_data는 검증용 문제와 정답이다.
# verbose는 학습 중 출력되는 문구를 설정한다.
hist = model.fit(X_train, Y_train, epochs=200, batch_size=10, validation_data=(X_train,Y_train),verbose=1)


fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
```
