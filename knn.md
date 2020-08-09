# 최근접 이웃 알고리즘
* kNN(k Nearest Neighbors) : 최근접 이웃 알고리즘
* 새로운 데이터의 분류를 알기 위해 사용  

## 몸무게와 키로 성별을 분류하는 KNN 지도학습 알고리즘 예제
```python
import random
import numpy as np

femaleData = [] # female 1
maleData = []   # male 0

# 지도학습에 필요한 데이터를 생성한다.
# [몸무게, 키, 성별] 데이터다
# 학습데이터는 [[weight, height, gender], [weight, height, gender], ....]
for i in range(50):
    femaleData.append([random.randint(40, 70), random.randint(140, 180), 1])
    maleData.append([random.randint(60, 90), random.randint(160, 200), 0])
    
# 두 값 사이의 거리를 계산하는 함수다.
# newData는 셩별 판정이 필요한 [weight, height] 다.
# dataItem는 newData와 거리 판정에 사용할 학습데이터의 n번째 리스트다.
# 실행시 전달되는 값 : distance([weight, height], [weight, height, gender])
def distance(newData, dataItem):
    # 두 점 사이의 거리를 구하는 함수
    return np.sqrt(pow((newData[0]-dataItem[0]), 2) + pow((newData[1]-dataItem[1]), 2))

# 성별을 판정하는 함수다.
# newData는 [weight, height]다. 성별을 판정할 데이터다.
# dataItems는 위에서 생성한 학습데이터 전체 리스트다.
# k는 분류결정에 사용할 데이터의 갯수다.
def knn(newData, dataItems, k):
    result = []
    count = 0
    # 학습데이터의 i번째 값과 newData의 거리를 계산하고, 그 결과를 result에 저장한다.
    for i in range(len(dataItems)):
        # result는 newData와의 거리와 성별이 포함되어 있다.
        result.append([distance(newData, dataItems[i]), dataItems[i][2]])
    result.sort()
    #print(result)
    
    # 거리판정이 완료된 result에서 k번째까지의 데이터를 조사한다.
    # 성별이 여자(1)로 판정된 값의 갯수를 모두 합한다.
    for i in range(k):
        if (result[i][1] == 1):
            count += 1
    # k갯수만큼 조사했을 때 여성으로 판정된 것이 k개의 절반보다 많으면 여자로 판정하고,
    # 아니면 남성으로 판정한다.
    if (count < (k/2)):
        print("female")
    else:
        print("male")
```
## KNN 지도학습 알고리즘을 활용해서 몸무게와 키로 성별 판정해보기
```python
weight = input("몸무게: ")
height = input("키: ")
k = input("k값")
newData = [int(weight), int(height)]
knn(newData, femaleData + maleData, int(k))
```

## KKN 지도학습 알고리즘을 활용해서 임의의 값으로 몸무게와 키로 성별을 판정하고, 그 결과를 그래프로 표시하기
```python
import matplotlib.pyplot as plt

femaleArray = np.array(femaleData)
maleArray = np.array(maleData)

for i, j in femaleArray[:, :2]:
    plt.plot(i, j, 'or')
for i, j in maleArray[:, :2]:
    plt.plot(i, j, 'ob')
plt.show()
```
