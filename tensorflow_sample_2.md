# tensorflow 사용하기
* tensorflow을 사용해서 Y = 2*X 로 동작하는 모델 훈련시키기
* 입력이 [1, 2, 3, ...]일 때, 출력이 [2, 4, 6, ...]이 되는 모델에 대한 샘플이다.

## 다층 퍼셉트론 레이어을 이용해서 Y = 2*X로 동작하는 모델 훈련하기 샘플
```python
import tensorflow as tf
# print(tf.__version__)
import tensorflow.keras.utils as utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt


# 1. 데이터셋 준비하기
# 훈련데이터의 문제다.
# 훈련데이터는 문제는 1, 2, 3, 4, ...
X_train = np.array(
[
    1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
]
)
# 훈련데이터의 정답이다.
# 훈련데이터의 정답은 2, 3, 6, 8, ....
# 즉, y = 2*x
Y_train = np.array(
[
    2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18
])

# 검정용데이터의 문제다.
X_val = np.array(
[
    1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
])

# 검증용 데이터의 정답이다.
Y_val = np.array(
[
    2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18
])


# 라벨링 전환
# 문제에 대한 정답을 전환하는 작업이다.
# 입력:1 -> 정답:[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Y_train = utils.to_categorical(Y_train,19)
Y_val = utils.to_categorical(Y_val,19)

# Dense

# 모형 클래스 객체를 생성한다.
model = Sequential()

# model.add()는 레이어를 추가한다.
# 입력은 1개다.
# 출력은 38개다, 임의로 정한 값이다.
# 활성화 함수는 'elu'다.
model.add(Dense(units=38, input_dim=1, activation='elu'))
# 출력은 19개다, 출력값이 0~18까지 나올기 때문에 출력의 갯수를 19개로 지정했다.
# 활성화 함수는 'softmax'다.
model.add(Dense(units=19,  activation='softmax'))

# 학습을 어떻게 할 것인지를 지정한다.
# loss는 손실함수를 지정한다.
# loss='categorical_crossentropy'

# optimizer는 loss(예측값과 정답사이의 차이)을 감소시키는 가중치를 조절하는 방법을 지정하는 것이다.
# optimizer는 손실함수의 기울기를 0으로 만드는 방식을 지정하는 것이다.

# metrics 평가기준을 말한다. 평가기준은 정확도와 재현율 등이 있다.
# metrics=['accuracy']는 평가기준으로 정확도를 사용하도록 지정한다.
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
# epochs는 반복횟수
# batch_size=1는 한 문제 풀고, 검증한다.
# verbose는
# vaidation_data는 검증용 데이터를 지정한다.
hist = model.fit(X_train, Y_train, epochs=200, batch_size=1, verbose=1, validation_data=(X_val, Y_val))




fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

# 그래프의 loss는 이상치와 계산값의 차이다.
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
# 그래프의 acc는 정확도다.
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 사용하기
X_test = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9
])
Y_test = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

])
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=1)

print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuray : ' + str(loss_and_metrics[1]))
```
