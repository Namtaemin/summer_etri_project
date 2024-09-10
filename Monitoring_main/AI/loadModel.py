import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
from tensorflow import keras
import joblib


import requests
# db값 반환하는 API를 통해 201개의 값을 받음
# 200개 : 예측을 위한 데이터, 1개 : 실제 값
data ={
  "year": 2021,
  "month": 8,
  "day" : 3,
  "hour": 6,
  "min" : 0,
  "sec": 0,
  "iter" : 1
}
res = requests.post('http://192.168.0.2:5000/getIntervalValueForLSTM', json=data)
data = res.text


# 정규화에 사용되었던 평균/표준편차값
meanT = 15.053684090161944
stdT  = 9.61201460854714

meanH = 63.67404628160599
stdH  = 20.392235502122855

meanI = 1015.3450066463703
stdI  = 7.642627412799711


import json
t = json.loads(data)
# 데이터를 json 형태로 변환 후 각 값을 나눔
dataT = t['temperature']
dataH = t['humidity']
dataI = t['illuminance']

# 값 정규화
testT = []
testH = []
testI = []
for i in range(1, 201):
    testT.append((dataT[i]-meanT)/stdT)
    testH.append((dataH[i]-meanH)/stdH)
    testI.append((dataI[i]-meanI)/stdI)

# 입력형태에 맞는 구조로 변환
testT = np.reshape(testT, (1, 200, 1))
testH = np.reshape(testH, (1, 200, 1))
testI = np.reshape(testI, (1, 200, 1))

# 모델 불러오기
modelT = joblib.load('model_T.pkl')
modelH = joblib.load('model_H.pkl')
modelI = joblib.load('model_I.pkl')

# 예측값과 실제 값 뽑아내기
preT = modelT.predict(testT)
reT = preT*stdT+meanT

preH = modelH.predict(testH)
reH = preH*stdH+meanH

preI = modelI.predict(testI)
reI = preI*stdI+meanI


# 예측값과 실제값 출력
d = [ ["temperature", reT, dataT[201], abs(reT-dataT[201])],
     ["humidity", reH, dataH[201], abs(reH-dataH[201])],
     ["illuminance", reI, dataI[201], abs(reI-dataI[201])]]

from tabulate import tabulate
print(tabulate(d, headers=['Type','Prediction','Real Value', 'Error']))



# print("reT : ", reT)
# print("realT : ", dataT[201], "    loss : ", abs(reT-dataT[201]))

# print("reH : ", reH)
# print("realH : ", dataH[201], "    loss : ", abs(reH-dataH[201]))

# print("reI : ", reI)
# print("realI : ", dataI[201], "    loss : ", abs(reH-dataI[201]))



#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
from tensorflow import keras
import joblib


interval = 1

data ={
  "iter" : 1,
  "len" : 0
}
res = requests.post('http://192.168.0.2:5000/getAllIntervalValue', json=data)


# 데이터 가져오기 및 json 형태로 변환
res_ = res.text
res_ = res_.replace(" GMT", "")

import json
a = json.loads(res_)
inputSize = np.size(a["temperature"])
a["temperature"] = np.reshape(a["temperature"], (inputSize))
a["humidity"] = np.reshape(a["humidity"], (inputSize))
a["illuminance"] = np.reshape(a["illuminance"], (inputSize))
data = pd.DataFrame(a)

data["dataTime"] = pd.to_datetime(data["dataTime"], format="['%a, %d %b %Y %H:%M:%S']")


uni_data_T = data['temperature']
uni_data_T.index = data['dataTime']

uni_data_H = data['humidity']
uni_data_H.index = data['dataTime']

uni_data_I = data['illuminance']
uni_data_I.index = data['dataTime']

# <class 'pandas.core.series.Series'>

print("======data Standardization======")
# 온도 데이터에 대해서, 평균을 빼고 표준편차로 나누어줌으로써 표준화를 진행합니다.
TRAIN_SPLIT = int(data['temperature'].size * 0.8)
print("TRAIN SIZE : ", TRAIN_SPLIT, "VAL : ", data["temperature"].size - TRAIN_SPLIT)

uni_data_T = uni_data_T.values
uni_train_mean_T = uni_data_T[:TRAIN_SPLIT].mean()
print("meanT : ", uni_train_mean_T)
uni_train_std_T = np.std(uni_data_T[:TRAIN_SPLIT])
print("stdT  : ", uni_train_mean_T)
uni_data_T = (uni_data_T - uni_train_mean_T) / uni_train_std_T  # Standardization
print("Temperature Standardization ok")
print(uni_data_T)


uni_data_H = uni_data_H.values
uni_train_mean_H = uni_data_H[:TRAIN_SPLIT].mean()
print("meanH : ", uni_train_mean_H)
uni_train_std_H = uni_data_H[:TRAIN_SPLIT].std()
print("stdH  : ", uni_train_std_H)
uni_data_H = (uni_data_H - uni_train_mean_H) / uni_train_std_H  # Standardization
print("Humidity Standardization ok")
print(uni_data_H)

uni_data_I = uni_data_I.values
uni_train_mean_I = uni_data_I[:TRAIN_SPLIT].mean()
print("meanI : ", uni_train_mean_I)
uni_train_std_I = uni_data_I[:TRAIN_SPLIT].std()
print("stdI  : ", uni_train_std_I)
uni_data_I = (uni_data_I - uni_train_mean_I) / uni_train_std_I  # Standardization
print("Illuminance Standardization ok")
print(uni_data_I)


# 우선 20개의 온도 관측치를 입력하면 다음 시간 스텝의 온도를 예측하도록 합니다.
# dataset : 표준화된 데이터
# history_size : 확인할 이전 데이터
# target_size : 예측해야할 레이블
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    # 훈련데이터와 예측 데이터 간의 분리
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

univariate_past_history = 200
univariate_future_target = 0

print("=====create data set=====")
x_train_uni_T, y_train_uni_T = univariate_data(uni_data_T, 0, TRAIN_SPLIT,univariate_past_history,univariate_future_target)
x_val_uni_T, y_val_uni_T = univariate_data(uni_data_T, TRAIN_SPLIT, None, univariate_past_history, univariate_future_target)
print("Temperature create ok")

x_train_uni_H, y_train_uni_H = univariate_data(uni_data_H, 0, TRAIN_SPLIT,univariate_past_history,univariate_future_target)
x_val_uni_H, y_val_uni_H = univariate_data(uni_data_H, TRAIN_SPLIT, None, univariate_past_history, univariate_future_target)
print("Humidity create ok")

x_train_uni_I, y_train_uni_I = univariate_data(uni_data_I, 0, TRAIN_SPLIT,univariate_past_history,univariate_future_target)
x_val_uni_I, y_val_uni_I = univariate_data(uni_data_I, TRAIN_SPLIT, None, univariate_past_history, univariate_future_target)
print("Illuminance create ok")


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.axis('auto')
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

# 과거 20개의 데이터를 받아서 그 평균을 반환
def baseline(history):
    return np.mean(history)

BATCH_SIZE = 256
BUFFER_SIZE = 10000

print("=====batch=====")
train_univariate_T = tf.data.Dataset.from_tensor_slices((x_train_uni_T, y_train_uni_T))
train_univariate_T = train_univariate_T.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_univariate_T = tf.data.Dataset.from_tensor_slices((x_val_uni_T, y_val_uni_T))
val_univariate_T = val_univariate_T.batch(BATCH_SIZE).repeat()
print("Temperature batch ok")

train_univariate_H = tf.data.Dataset.from_tensor_slices((x_train_uni_H, y_train_uni_H))
train_univariate_H = train_univariate_H.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_univariate_H = tf.data.Dataset.from_tensor_slices((x_val_uni_H, y_val_uni_H))
val_univariate_H = val_univariate_H.batch(BATCH_SIZE).repeat()
print("Humidity batch ok")

train_univariate_I = tf.data.Dataset.from_tensor_slices((x_train_uni_I, y_train_uni_I))
train_univariate_I = train_univariate_I.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_univariate_I = tf.data.Dataset.from_tensor_slices((x_val_uni_I, y_val_uni_I))
val_univariate_I = val_univariate_I.batch(BATCH_SIZE).repeat()
print("Illuminance batch ok")


# LSTM 모델 구성
print("=====LSTM Model Setting=====")

interval = 100
import joblib
path = "../AI/model_"
# 모델 불러오기
modelT = joblib.load(path+"t_"+str(interval)+".pkl")
modelH = joblib.load(path+"h_"+str(interval)+".pkl")
modelI = joblib.load(path+"i_"+str(interval)+".pkl")


simple_lstm_model_T = modelT
print("Temperature Setting ok")

simple_lstm_model_H = modelH
print("Humidity Setting ok")

simple_lstm_model_I = modelI
print("Illuminance Setting ok")


def printLoss(hist, title):
    print("loss", hist.history["loss"])
    print("acc", hist.history["acc"])
    print("val_loss", hist.history["val_loss"])
    print("val_acc", hist.history["val_acc"])
    
    
    # 모델 학습 과정 표시
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')
    
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    
    plt.title(title)
    plt.show()
    


# 모델 훈련
EVALUATION_INTERVAL = 200
EPOCHS = 10

histT = simple_lstm_model_T.fit(train_univariate_T, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate_T, validation_steps=50)
# 예측 결과 확인
for x, y in val_univariate_T.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      simple_lstm_model_T.predict(x)[0]], 0, 'Simple LSTM model : T')
    plot.show()


plt.title("Compare future and prediction data : T")
plt.plot(y_val_uni_T, label="input")
plt.plot(simple_lstm_model_T.predict(x_val_uni_T), label="prediction")
plt.legend(loc='best')
plot.show()

printLoss(histT, "Loss & Accuracy : T")

histH = simple_lstm_model_H.fit(train_univariate_H, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate_H, validation_steps=50)
for x, y in val_univariate_H.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      simple_lstm_model_H.predict(x)[0]], 0, 'Simple LSTM model : H')
    plot.show()
    

plt.title("Compare future and prediction data : H")
plt.plot(y_val_uni_H, label="input")
plt.plot(simple_lstm_model_H.predict(x_val_uni_H), label="prediction")
plt.legend(loc='best')
plot.show()
    
printLoss(histH, "Loss & Accuracy : H")
    
histI = simple_lstm_model_I.fit(train_univariate_I, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate_I, validation_steps=50)
for x, y in val_univariate_I.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      simple_lstm_model_I.predict(x)[0]], 0, 'Simple LSTM model : I')
    plot.show()
    
plt.title("Compare future and prediction data : I")
plt.plot(y_val_uni_I, label="input")
plt.plot(simple_lstm_model_I.predict(x_val_uni_I), label="prediction")
plt.legend(loc='best')
plot.show()

printLoss(histI, "Loss & Accuracy : I")

import joblib
# 학습시킨 모델을 현재 경로에 knn_model.pkl 파일로 저장합니다.
joblib.dump(simple_lstm_model_T, './model_t_100.pkl')
joblib.dump(simple_lstm_model_H, './model_h_100.pkl')
joblib.dump(simple_lstm_model_I, './model_i_100.pkl')






