# 딥러닝 실습하기

## 아나콘다 가상환경 만들기
1. anaconda prompt
2. conda create -n 환경이름 python=3.7
3. pip install 모듈명  
   **필요한 라이브러리** : tensorflow, matplotlib

## 아나콘다 가상환경 활성화/비활성화
- 활성화 : conda activate 환경이름
- 비활성화 : conda deactivate

## PyCharm에서 아나콘다 가상환경을 인터프리터로 사용하기
 - New Project -> Existing interpreter 체크 -> 더보기 -> Conda Environmen -> (Conda 가상환경의) Interpreter 선택  
 - 새 프로젝트에서 tensorflow 모듈 설치 성공여부 확인하기  
 ```python
 import tensorflow as tf
 print(tf.__version__) # 버전번호가 2.3.x로 출력되면 성공  
 ```
