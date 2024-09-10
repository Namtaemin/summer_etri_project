import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf


#%% 평균제곱오차(MSE)

# =============================================================================
# 
# 평균제곱오차 손실함수 구현하기!
# 수식 : MSE = 1/N * i번째까지 합 ((ei)^2)
# 의미 : (예측값 - 목표값)^2을 i번째까지 모두 합산 한 뒤 전체 개수로 나누어 평균값 도출
# 
# =============================================================================

def MSE(y, t):
    # 전체 원소 개수
    N = t.size
    # MSE계
    return np.sum((y-t)**2)/N

# 예측값 : 
y = np.array([0.5, 1.0, 1.5, 2.0])
# yy = tf.convert_to_tensor(y, dtype = tf.float32)
# 실제값
t = np.array([1.0, 2.0, 3.0, 4.0])
# tt = tf.convert_to_tensor(t, dtype = tf.float32)

MSEReult = MSE(y,t)
# tf.keras.losses.MeanSquaredError()

print()
print("## 평균 제곱 오차 손실 함수 ##")
print("y :", y)
print("t :", t)
print("MST :", MSEReult)

#%% 경사하강법(gradient decent)


# 평균 제곱 오차
def MSE(y, t):
    # 전체 원소 개수
    N = t.size
    # MSE계
    return np.sum((y-t)**2)/N

# 1~12까지의 값이 들어간 array생성
x = np.arange(12)
t = np.arange(12)

# 가중치
w = 0.5
# 바이어스
b = 0
# 학습률
lr = 0.001

lossList = []
for epoch in range(201) :
    # 단순 선형 모델식
    y = w * x + b 
    # 변화량 계산
    dW = np.sum((y - t) * x) / (2 * x.size)
    dB = np.sum((y - t)) / (2 * x.size)
    
    # 값 업데이트
    w = w - lr * dW
    b = b - lr * dB
    
    # 결과 도출
    y = w * x + b
    # 평균 제곱 오차값 계산 후 추가
    loss = MSE(y, t)
    lossList.append(loss)
    
    if not epoch % 10 :
        print("epoch={}:\t w={:>8.4f}.\t b={:>8.4f},\t loss={}".format(epoch, w, b, loss))
    
plt.plot(lossList)
plt.show()

#%% 확률적 경사하강법(SGD)



# 평균 제곱 오차
def MSE(y, t):
    # 전체 원소 개수
    N = t.size
    # MSE계
    return np.sum((y-t)**2)/N

# 1~12까지의 값이 들어간 array생성
x = np.arange(12)
t = np.arange(12)

# 가중치
w = 0.5
# 바이어스
b = 0
# 학습률
lr = 0.001

trainSize = t.size
batchSize = 4
K = trainSize // batchSize

lossList = []
for epoch in range(101) :
    loss = 0
    for step in range(K) :
        mask = np.random.choice(trainSize, batchSize)
        xBatch = x[mask]
        tBatch = t[mask]
    
        # 단순 선형 모델식
        y = w * xBatch + b 
        # 변화량 계산
        dW = np.sum((y - tBatch) * xBatch) / (2 * batchSize)
        dB = np.sum((y - tBatch)) / (2 * batchSize)
        
        # 값 업데이트
        w = w - lr * dW
        b = b - lr * dB
    
        # 결과 도출
        y = w * xBatch + b
        test =  MSE(y, t)
        print("테스트ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ",  test)
        # 평균 제곱 오차값 계산 후 추가
        loss += MSE(y, t)
    loss /= K
    lossList.append(loss)
    
    if not epoch % 10 :
        print("epoch={}:\t w={:>8.4f}.\t b={:>8.4f},\t loss={}".format(epoch, w, b, loss))
    
plt.plot(lossList)
plt.show()

















































