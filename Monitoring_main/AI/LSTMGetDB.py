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
# data ={
#   "year": 2021,
#   "month": 8,
#   "day" : 3,
#   "hour": 6,
#   "min" : 0,
#   "sec": 0,
#   "iter" : 3
# }
# res = requests.post('http://192.168.0.2:5000/getIntervalValueForLSTM', json=data)
data ={
  "iter" : 8,
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

# 출력 결과 : univariate_data() 함수가 만든 20개의 과거 온도 데이터와 1개의 목표 예측 온도를 나타냅니다.
# print('Single window of past history')
# print(x_train_uni_T[0])
# print('\n Target temperature to predict')
# print(y_train_uni_T[0])



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

# 파란 마커 : 20개의 과거 온도 데이터
# 빨간 마커 : 예측해야 할 미래의 온도 데이터
# show_plot([x_train_uni_T[0], y_train_uni_T[0]], 0, 'Sample T').show()
# show_plot([x_train_uni_H[0], y_train_uni_H[0]], 0, 'Sample H').show()
# show_plot([x_train_uni_I[0], y_train_uni_I[0]], 0, 'Sample I').show()


# 과거 20개의 데이터를 받아서 그 평균을 반환
def baseline(history):
    return np.mean(history)

# 녹색 마커 : 과거 온도 데이터의 평균값을 이용해서 예측한 지점
# show_plot([x_train_uni_T[0], y_train_uni_T[0], baseline(x_train_uni_T[0])], 0, 'Sample TT').show()
# show_plot([x_train_uni_H[0], y_train_uni_H[0], baseline(x_train_uni_H[0])], 0, 'Sample HH').show()
# show_plot([x_train_uni_I[0], y_train_uni_I[0], baseline(x_train_uni_I[0])], 0, 'Sample II').show()

# 이터셋을 shuffle, batch, cache하는 작업
# =============================================================================
# cache()
# 데이터셋을 캐시, 즉 메모리 또는 파일에 보관합니다. 따라서 두번째 이터레이션부터는 캐시된 데이터를 사용합니다.
# 
# shuffle()
# 데이터셋을 임의로 섞어줍니다. BUFFER_SIZE개로 이루어진 버퍼로부터 임의로 샘플을 뽑고, 뽑은 샘플은 다른 샘플로 대체합니다. 완벽한 셔플을 위해서 전체 데이터셋의 크기에 비해 크거나 같은 버퍼 크기가 요구됩니다.
# 
# batch()
# 데이터셋의 항목들을 하나의 배치로 묶어줍니다.
# =============================================================================
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
simple_lstm_model_T = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni_T.shape[-2:]),
    tf.keras.layers.Dense(1)
])
print("Temperature Setting ok")

simple_lstm_model_H = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni_H.shape[-2:]),
    tf.keras.layers.Dense(1)
])
print("Humidity Setting ok")

simple_lstm_model_I = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni_I.shape[-2:]),
    tf.keras.layers.Dense(1)
])
print("Illuminance Setting ok")


from tensorflow.keras.optimizers import SGD
print("=====LSTM Model Compile=====")
simple_lstm_model_T.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='mae', metrics=["acc"])
print("Temperature Compile ok")
simple_lstm_model_H.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='mae', metrics=["acc"])
print("Humidity Compile ok")
simple_lstm_model_I.compile(optimizer=SGD(lr=0.001, momentum=0.9),  loss='mae', metrics=["acc"])
print("Illuminance Compile ok")


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
EVALUATION_INTERVAL = 1000
EPOCHS = 50

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
joblib.dump(simple_lstm_model_T, './model_t_8.pkl')
joblib.dump(simple_lstm_model_H, './model_h_8.pkl')
joblib.dump(simple_lstm_model_I, './model_i_8.pkl')