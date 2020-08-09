# 퍼셉트론을 활용해서 AND, OR, NAND, XOR 게이트 구현하기

## 퍼셉트론을 활용해서 입력에 대해서 AND, OR, NAND 게이트을 구현한 함수
```python
# 논리회로 AND, OR, NAMD

# w1*x1 + w2*x2 + b > 0   흐른다.
# w1*x1 + w2*x2 + b <= 0  흐르지 않는다.

import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0 # 흐르지 않는다.
    else:
        return 1 # 흐른다.
    
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0 # 흐르지 않는다.
    else:
        return 1 # 흐른다.
    
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0 # 흐르지 않는다.
    else:
        return 1 # 흐른다.
```

## 위에서 구현된 함수를 사용하기
```python
print("AND 연산")
for item in [(0,0), (0,1), (1,0), (1,1)]:
    result = AND(item[0], item[1])
    print(str(item), " -> ", str(result))

print()
print("OR 연산")
for item in [(0,0), (0,1), (1,0), (1,1)]:
    result = OR(item[0], item[1])
    print(str(item), " -> ", str(result))
    
print()
print("NAND 연산")
for item in [(0,0), (0,1), (1,0), (1,1)]:
    result = NAND(item[0], item[1])
    print(str(item), " -> ", str(result))
    
print()
print("XOR 연산")
for item in [(0,0), (0,1), (1,0), (1,1)]:
    result1 = NAND(item[0], item[1])
    result2 = OR(item[0], item[1])
    result = AND(result1, result2)
    print(str(item), " -> ", str(result))
```
