# 퍼셉트론을 활용해서 AND, OR, NAND, XOR 게이트 구현하기
## 퍼셉트론을 활용해서 입력에 대해서 AND, OR, NAND 게이트을 구현한 함수
* 가중치와 절편을 numpy의 램덤함수를 사용해서 구하는 샘플이다.

```python
import numpy as np

def AND(x1, x2):
    # 문제와 답을 제공한다.
    data = [[0,0,0], [0,1,0],[1,0,0], [1,1,1]]
    # 정답을 알고 있는 무제를 담은 넘피배열
    x = np.array([x1, x2])
    
    # 위에서 제시한 모든 답을 찾을 때 까지 반복문을 반복한다.
    while True:
        # 가중치를 램덤하게 구한다.
        w = np.array([np.random.normal(), np.random.normal()])
        # 절편을 램던하게 구한다.
        b = np.random.normal()
        
        # cnt는 문제에 대한 정답을 맞힌 갯수다.
        cnt = 0
        # 문제와 정답을 제시한다.
        # i의 값은 [0, 0, 1]과 같은 값이다.
        for i in data:
            # 가중치와 절편을 계산한다.
            tmp = i[0]*w[0] + i[1]*w[1] + b
            # 계산된 결과로 퍼셉트론 출력결과를 판정한다.
            if tmp <= 0:
                result = 0
            else:
                result = 1
            # 퍼셉트론 출력결과가 정답과 일치하는 확인한다.
            if result == i[2]:
                # 퍼셉트론 출력결과와 정답이 일치하는 경우 cnt를 증가시킨다.
                cnt += 1
        # data에서 제시한 모든 정답을 만족하는 경우 반복문을 탈출한다.
        if (cnt == 4):
            print(str(x), " -> ", "w:", str(w), "b:", str(b))
            break;
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0 # 흐르지 않는다.
    else:
        return 1 # 흐른다.    
    
def OR(x1, x2):
    # 문제와 답을 제공한다.
    data = [[0,0,0], [0,1,1],[1,0,1], [1,1,1]]
    # 정답을 알고 있는 무제를 담은 넘피배열
    x = np.array([x1, x2])
    
    # 위에서 제시한 모든 답을 찾을 때 까지 반복문을 반복한다.
    while True:
        # 가중치를 램덤하게 구한다.
        w = np.array([np.random.normal(), np.random.normal()])
        # 절편을 램던하게 구한다.
        b = np.random.normal()
        
        # cnt는 문제에 대한 정답을 맞힌 갯수다.
        cnt = 0
        # 문제와 정답을 제시한다.
        # i의 값은 [0, 0, 1]과 같은 값이다.
        for i in data:
            # 가중치와 절편을 계산한다.
            tmp = i[0]*w[0] + i[1]*w[1] + b
            # 계산된 결과로 퍼셉트론 출력결과를 판정한다.
            if tmp <= 0:
                result = 0
            else:
                result = 1
            # 퍼셉트론 출력결과가 정답과 일치하는 확인한다.
            if result == i[2]:
                # 퍼셉트론 출력결과와 정답이 일치하는 경우 cnt를 증가시킨다.
                cnt += 1
        # data에서 제시한 모든 정답을 만족하는 경우 반복문을 탈출한다.
        if (cnt == 4):
            print(str(x), " -> ", "w:", str(w), "b:", str(b))
            break;
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0 # 흐르지 않는다.
    else:
        return 1 # 흐른다.   
    
def NAND(x1, x2):
    # 문제와 답을 제공한다.
    data = [[0,0,1], [0,1,1],[1,0,1], [1,1,0]]
    # 정답을 알고 있는 무제를 담은 넘피배열
    x = np.array([x1, x2])
    
    # 위에서 제시한 모든 답을 찾을 때 까지 반복문을 반복한다.
    while True:
        # 가중치를 램덤하게 구한다.
        w = np.array([np.random.normal(), np.random.normal()])
        # 절편을 램던하게 구한다.
        b = np.random.normal()
        
        # cnt는 문제에 대한 정답을 맞힌 갯수다.
        cnt = 0
        # 문제와 정답을 제시한다.
        # i의 값은 [0, 0, 1]과 같은 값이다.
        for i in data:
            # 가중치와 절편을 계산한다.
            tmp = i[0]*w[0] + i[1]*w[1] + b
            # 계산된 결과로 퍼셉트론 출력결과를 판정한다.
            if tmp <= 0:
                result = 0
            else:
                result = 1
            # 퍼셉트론 출력결과가 정답과 일치하는 확인한다.
            if result == i[2]:
                # 퍼셉트론 출력결과와 정답이 일치하는 경우 cnt를 증가시킨다.
                cnt += 1
        # data에서 제시한 모든 정답을 만족하는 경우 반복문을 탈출한다.
        if (cnt == 4):
            print(str(x), " -> ", "w:", str(w), "b:", str(b))
            break;
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0 # 흐르지 않는다.
    else:
        return 1 # 흐른다. 
 ```
 
## 위에서 구현한 함수를 사용하기
```python
print()
print("AND 연산")
for i in [(0,0), (0,1), (1,0), (1,1)]:
    print(AND(i[0], i[1]))
    
print()
print("OR 연산")
for i in [(0,0), (0,1), (1,0), (1,1)]:
    print(OR(i[0], i[1]))
    
print()
print("NAND 연산")
for i in [(0,0), (0,1), (1,0), (1,1)]:
    print(NAND(i[0], i[1]))
    
print()
print("XOR 연산")
for i in [(0,0), (0,1), (1,0), (1,1)]:
    s1  = NAND(i[0],i[1])
    s2  = OR(i[0],i[1])
    result   = AND(s1, s2)
    print(result)
```
