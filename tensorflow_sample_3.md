# tensorflow 사용하기
* tensorflow을 사용해서 손글씨 판별하기

## 다층 퍼셉트론 레이어을 이용해서 손글씨를 판정하는 모델 생성, 모델 저장, 모델 불어오기 샘플 코드

### 다층 퍼셉트론 레이어을 이용해서 손글씨를 판정하는 모델 생성, 모델 저장 샘플 코드
* 프로그램을 실행하면 "my_model.h5"라는 파일명으로 모델이 생성된다.

```python
import tensorflow.keras.utils as utils
# mnist는 인공지능 학습에 활용되는 손글씨 데이터셋이다.
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

# 1. 데이터셋 준비하기

# 훈련셋과 시험셋 로딩
# 훈련용 6만개와 검증용 1만개가 로딩된다.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 훈련셋과 검증셋 분리
X_val = X_train[50000:]
Y_val = Y_train[50000:]
X_train = X_train[:50000]
Y_train = Y_train[:50000]

# 28*28인 2차원 데이터를 784*1인 1차원 데이터로 변경한다.
X_train = X_train.reshape(50000, 784).astype('float32') / 255.0
X_val = X_val.reshape(10000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0

# 훈련셋, 검증셋 고르기
#train_rand_idxs = np.random.choice(50000, 700)
#val_rand_idxs = np.random.choice(10000, 300)

#X_train = X_train[train_rand_idxs]
#Y_train = Y_train[train_rand_idxs]
#X_val = X_val[val_rand_idxs]
#Y_val = Y_val[val_rand_idxs]

# 라벨링 전환
# 입력은 784개
Y_train = utils.to_categorical(Y_train)
# 출력은 10개
Y_val = utils.to_categorical(Y_val)
Y_test = utils.to_categorical(Y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=100, input_dim=28*28, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# # 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# # 4. 모델 학습시키기
#hist = model.fit(X_train, Y_train, epochs=10, batch_size=10, validation_data=(X_val, Y_val))

# # 4. 모델 학습시키기 + 학습 조기 종료시키기
# patience 속성은 개선이 없다고 바료 종료하지 않고, 개선이 없는 에포크를 얼마나 기다려 줄 것인가를 지정한다.
# 만약, 10이라고 지정하면 개선이 엇는 에포크가 10번째 지속될 경우 학습을 종료한다.
early_stopping = EarlyStopping(patience=5) # 조기종료 콜백함수 정의
hist = model.fit(X_train, Y_train, epochs=1000, batch_size=10, verbose=1, validation_data=(X_val, Y_val), callbacks=[early_stopping])

# 6. 모델 저장하기
model.save('my_model.h5')

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
```

### 다층 퍼셉트론 레이어을 이용해서 손글씨를 판정하는 저장된 모델을 불러와서 사용하기 샘플코드
* "my_model.h5"로 저장된 모델을 불러와서 모델을 사용한다.
```python
from tensorflow.keras.datasets import mnist
import tensorflow.keras.utils as utils
import numpy as np
# 1. 실무에 사용할 데이터 준비하기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_test = utils.to_categorical(y_test)

# 훈련용 데이터에서 랜덤하게 5개의 훈련용 데이터만 가져오기
xhat_idx = np.random.choice(x_test.shape[0], 5)
xhat = x_test[xhat_idx]

# 2. 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')

# 3. 모델 사용하기
yhat = model.predict_classes(xhat)

for i in range(5):
    print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
...
