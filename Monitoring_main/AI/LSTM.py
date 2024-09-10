#%% 이게 진짜 찐 최종임;;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
from tensorflow import keras

print("=====data load=====")

# 온도
#dataT = pd.read_csv("./data/Temperature.csv", encoding='cp949')
dataT = pd.read_csv("./dataSet/yangsan3.csv", encoding='cp949')
dataT["시간"] = pd.to_datetime(dataT["시간"], format="%Y-%m-%d %H:%M:%S")
print("Temperature.csv load")
print(type(dataT))
# 습도
#dataH = pd.read_csv("./data/Humidity.csv", encoding='cp949')
dataH = pd.read_csv("./dataSet/yangsan3.csv", encoding='cp949')
dataH["시간"] = pd.to_datetime(dataT["시간"], format="%Y-%m-%d %H:%M:%S")
print("Humidity.csv load")
# 조도
#dataI = pd.read_csv("./data/Illuminance.csv", encoding='cp949')
dataI = pd.read_csv("./dataSet/yangsan3.csv", encoding='cp949')
dataI["시간"] = pd.to_datetime(dataT["시간"], format="%Y-%m-%d %H:%M:%S")
print("Illuminance.csv load")


print("=====data plt=====")
plt.figure(figsize=(10,5))

plt.subplot(3, 1, 1)
plt.title("temperature", fontsize=15)
plt.plot(dataT["시간"], dataT["온도"], "-", color='red', label=str("temperature"))
plt.grid()
plt.legend(fontsize=13)
plt.xticks(rotation=90)
print("temperature : plt")

plt.subplot(3, 1, 2)
plt.title("Humidity", fontsize=15)
plt.plot(dataH["시간"], dataH["습도"], "-", color='red', label=str("temperature"))
plt.grid()
plt.legend(fontsize=13)
plt.xticks(rotation=90)
print("Humidity : plt")

plt.subplot(3, 1, 3)
plt.title("Illuminance", fontsize=15)
plt.plot(dataI["시간"], dataI["기압"], "-", color='red', label=str("temperature"))
plt.grid()
plt.legend(fontsize=13)
plt.xticks(rotation=90)
print("Illuminance : plt")

plt.show()


print("=====data convert=====")
uni_data_T = dataT['온도']
uni_data_T.index = dataT['시간']

uni_data_H = dataH['습도']
uni_data_H.index = dataH['시간']

uni_data_I = dataI['기압']
uni_data_I.index = dataI['시간']


print("======data Standardization======")
# 온도 데이터에 대해서, 평균을 빼고 표준편차로 나누어줌으로써 표준화를 진행합니다.
TRAIN_SPLIT = int(dataT['온도'].size * 0.8)
print("TRAIN SIZE : ", TRAIN_SPLIT, "VAL : ", dataT["온도"].size - TRAIN_SPLIT)

uni_data_T = uni_data_T.values
uni_train_mean_T = uni_data_T[:TRAIN_SPLIT].mean()
print("meanT : ", uni_train_mean_T)
uni_train_std_T = uni_data_T[:TRAIN_SPLIT].std()
print("stdT  : ", type(uni_train_std_T))
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
# joblib.dump(simple_lstm_model_T, './model_T.pkl')
# joblib.dump(simple_lstm_model_H, './model_H.pkl')
# joblib.dump(simple_lstm_model_I, './model_I.pkl')

# simple_lstm_model_H.save("model_H")    
# simple_lstm_model_I.save("model_I")    
# simple_lstm_model_T.save("model_T")

#%%

import tensorflow as tf
import numpy as np

x = np.arange(1,500,1)
y = 0.4 * x + 30
plt.plot(x,y)

trainx, testx = x[0:int(0.8*(len(x)))], x[int(0.8*(len(x))):]
trainy, testy = y[0:int(0.8*(len(y)))], y[int(0.8*(len(y))):]
train = np.array(list(zip(trainx,trainy)))
test = np.array(list(zip(testx,testy)))


# sample : 관측치 수 , 즉 데이터 개수 
# lookback : LSTM model에서 과거 어디까지 볼 것인지에 대한 것이다. 
# features : 현재 인풋으로 사용할 개수
def create_dataset(n_X, look_back):
    dataX, dataY = [], []
    # 전체 중 내가 되 돌아볼 만큼까지
    for i in range(len(n_X)-look_back):
        # n ~ n+look_back 데이터 리스트를
        a = n_X[i:(i+look_back), ]
        # 입력값에 추가
        dataX.append(a)
        # n+look_back 이후 데이터 값 '하나'를 결과값에 추가
        dataY.append(n_X[i + look_back, ])
        
        # a = np.arange(1, 20, 1)
        # b = a[1:10,] => [ 2  3  4  5  6  7  8  9 10]
        # c = a[10,]   => 11
        
    return np.array(dataX), np.array(dataY)

# 되돌아 볼 수
look_back = 1
# 학습 데이터 생성
trainx,trainy = create_dataset(train, look_back)
# 테스트 데이터 생성
testx,testy = create_dataset(test, look_back)

# 형태 변환
trainx = np.reshape(trainx, (trainx.shape[0], 1, 2))
# 형태변환
testx = np.reshape(testx, (testx.shape[0], 1, 2))



from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(256, return_sequences = True, input_shape = (trainx.shape[1], 2)))
model.add(LSTM(128,input_shape = (trainx.shape[1], 2)))
model.add(Dense(2))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(trainx, trainy, epochs = 2000, batch_size = 10, verbose = 2, shuffle = False)
model.save_weights('LSTMBasic1.h5')

model.load_weights('LSTMBasic1.h5')
predict = model.predict(testx)

plt.plot(testx.reshape(398,2)[:,0:1], testx.reshape(398,2)[:,1:2])
plt.plot(predict[:,0:1], predict[:,1:2])

#%%
a = np.arange(1, 500, 1)
b = a[1:400,]
c = a[10,]

print(b)
print(c)
print()
print()

b = np.array(b)
c = np.array(c)
print(np.reshape(b, (b.shape[0], 1, 2)))

#%%
###########################
# 라이브러리 사용
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
 
###########################
# 데이터를 준비합니다.
data = pd.read_csv("./dataSet/제주특별자치도개발공사_기상 정보_20201231.csv", encoding='cp949')
data.head()
# 종속변수, 독립변수
# temp = []
# for i in range(47183):
#     temp.append(i//60)
#     i+=1

# temp = np.reshape(temp, (47183, 1))
# 독립 = temp
독립 = data['평균기온(섭씨)']
종속 = data['최대기온(섭씨)']

leng = data['평균기온(섭씨)'].size
trainX = []
trainY = []
testX = []
testY = []

cut = int(leng*0.8)
for i in range(leng) :
    if(i >= cut) :
        testX.append(독립[i])
        testY.append(종속[i])
    else :
        trainX.append(독립[i])
        trainY.append(종속[i])
        
print(len(trainX))
print(len(testX))

trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

trainX = np.reshape(trainY, (292, 1))
trainY = np.reshape(trainY, (292, 1))
testX = np.reshape(testX, (74, 1))
testY = np.reshape(testY, (74, 1))

print(trainX.shape, trainY.shape)
print(testX.shape, testX.shape)



###########################
# 모델을 만듭니다.
X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')
 
###########################
# 모델을 학습시킵니다. 
model.fit(trainX, trainY, epochs=1000, verbose=0)
 
###########################
# 모델을 이용합니다. 
print(model.predict(독립))
print(model.predict([[15]]))


pred = model.predict(독립)

plt.figure(figsize=(12, 9))
plt.plot(np.asarray(종속), label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()


