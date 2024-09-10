import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
 
f = open('./data/Temperature.csv','r', encoding='cp949')
rdr = csv.reader(f)
 
cnt = 0
temper = []
temp = []
for line in rdr:
    if cnt == 0 :
        print(line)
        cnt += 1
        continue
    if cnt == 100 :
        break
    print(line)
    #temp.append(line[1])
    temp.append(line[2])
    temp.append(line[3])
    temp.append(line[8])
    temper.append(temp)
    temp = []
    
    cnt +=1
    
iris = sns.load_dataset("iris")    # 붓꽃 데이터
titanic = sns.load_dataset("titanic")    # 타이타닉호 데이터
tips = sns.load_dataset("tips")    # 팁 데이터
flights = sns.load_dataset("flights")    # 여객운송 데이터

print(type(iris))
print(type(temper))
convert = pd.DataFrame(temper, columns = ['l', 'r', 't'])
print(type(convert))

print(convert)

# sns.jointplot(x='l', y='r', data=convert)
# plt.suptitle("fesf", y=1.02)
# plt.show()

sns.pairplot(convert)
plt.title("Iris Data의 Pair Plot")
plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./data/Temperature.csv", encoding='cp949')
#2021-11-01 00:01:20
df["시간"] = pd.to_datetime(df["시간"], format="%Y-%m-%d %H:%M:%S")

plt.figure(figsize=(10,5))
plt.title("temperature", fontsize=15)
print(df)
plt.plot(df["시간"], df["온도"], "-", color='red', label=str("temperature"))
plt.grid()
plt.legend(fontsize=13)
plt.xticks(rotation=90)
plt.show()

#%%
import numpy as np
import csv
import random
 
f = open('./data/Temperature.csv','r', encoding='cp949')
rdr = csv.reader(f)
 
DATASIZE = 80000

cnt = 0
temper = []
time = []
for line in rdr:
    if cnt == 0 :
        cnt += 1
        continue
    if cnt == DATASIZE :
        break
    time.append(line[1])
    temper.append(line[8])
    temp = []
    
    cnt +=1
    
#길이
perch_length = np.array(time)
perch_length.size
for i in range(0,perch_length.size) :
    temp = perch_length[i]
    temp = temp[11:]
    temp = temp[:-3]
    temp = temp.replace(":","")
    temp2 = (int(temp)/1000)*1000
    perch_length[i] = temp2
    
#무게
perch_weight = np.array(temper)

from sklearn.model_selection import train_test_split

# 훈련 세트와 테스트 세트로 나눕니다
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, test_size=0.2, random_state=10000)
# 훈련 세트와 테스트 세트를 2차원 배열로 바꿉니다
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)


#산점도를 그리기위한 라이브러리
import matplotlib.pyplot as plt

ti = train_input.tolist()
tt = train_target.tolist()
plt.plot(100,300)
plt.axis([-5, 60, 10, 50])
plt.style.use('ggplot')
np.random.seed(0)

number_of_colors = DATASIZE
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(DATASIZE)]
ti.sort()
tt.sort()
plt.scatter(ti, tt, s=3, c="#9932CC", alpha=0.1, cmap='Spectral')
plt.xlabel('length')
plt.ylabel('weight')

plt.show()


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
tri = train_input.tolist()
trt = train_target.tolist()
lr.fit(tri, trt)

# print(lr.predict([[19]]))
# print(lr.coef_, lr.intercept_)

# 훈련 세트의 산점도를 그립니다
plt.scatter(tri, trt)
# 15에서 50까지 1차 방정식 그래프를 그립니다
plt.plot([-5, 60], [10, 50])
# 50cm 농어 데이터
plt.scatter(53, 141.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(tri, trt))
print(lr.score(ti, tt))

#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
from tensorflow import keras

print("=====data load=====")

# 온도
dataT = pd.read_csv("./data/Temperature.csv", encoding='cp949')
dataT["시간"] = pd.to_datetime(dataT["시간"], format="%Y-%m-%d %H:%M:%S")
print("Temperature.csv load")
# 습도
dataH = pd.read_csv("./data/Humidity.csv", encoding='cp949')
dataH["시간"] = pd.to_datetime(dataT["시간"], format="%Y-%m-%d %H:%M:%S")
print("Humidity.csv load")
# 조도
dataI = pd.read_csv("./data/Illuminance.csv", encoding='cp949')
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
plt.plot(dataI["시간"], dataI["조도"], "-", color='red', label=str("temperature"))
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

uni_data_I = dataI['조도']
uni_data_I.index = dataI['시간']



print("======data Standardization======")
# 온도 데이터에 대해서, 평균을 빼고 표준편차로 나누어줌으로써 표준화를 진행합니다.
TRAIN_SPLIT = int(dataI['조도'].size * 0.8)
print("TRAIN SIZE : ", TRAIN_SPLIT, "VAL : ", dataI["조도"].size - TRAIN_SPLIT)

uni_data_T = uni_data_T.values
uni_train_mean_T = uni_data_T[:TRAIN_SPLIT].mean()
uni_train_std_T = uni_data_T[:TRAIN_SPLIT].std()
uni_data_T = (uni_data_T - uni_train_mean_T) / uni_train_std_T  # Standardization
print("Temperature Standardization ok")
print(uni_data_T)


uni_data_H = uni_data_H.values
uni_train_mean_H = uni_data_H[:TRAIN_SPLIT].mean()
uni_train_std_H = uni_data_H[:TRAIN_SPLIT].std()
uni_data_H = (uni_data_H - uni_train_mean_H) / uni_train_std_H  # Standardization
print("Humidity Standardization ok")
print(uni_data_H)

uni_data_I = uni_data_I.values
uni_train_mean_I = uni_data_I[:TRAIN_SPLIT].mean()
uni_train_std_I = uni_data_I[:TRAIN_SPLIT].std()
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
show_plot([x_train_uni_T[0], y_train_uni_T[0]], 0, 'Sample T').show()
show_plot([x_train_uni_H[0], y_train_uni_H[0]], 0, 'Sample H').show()
show_plot([x_train_uni_I[0], y_train_uni_I[0]], 0, 'Sample I').show()


# 과거 20개의 데이터를 받아서 그 평균을 반환
def baseline(history):
    return np.mean(history)

# 녹색 마커 : 과거 온도 데이터의 평균값을 이용해서 예측한 지점
show_plot([x_train_uni_T[0], y_train_uni_T[0], baseline(x_train_uni_T[0])], 0, 'Sample TT').show()
show_plot([x_train_uni_H[0], y_train_uni_H[0], baseline(x_train_uni_H[0])], 0, 'Sample HH').show()
show_plot([x_train_uni_I[0], y_train_uni_I[0], baseline(x_train_uni_I[0])], 0, 'Sample II').show()

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
BATCH_SIZE = 1024
BUFFER_SIZE = 10000

print("=====batch=====")
train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni_T, y_train_uni_T))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni_T, y_val_uni_T))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
print("Temperature batch ok")

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni_H, y_train_uni_H))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni_H, y_val_uni_H))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
print("Humidity batch ok")

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni_I, y_train_uni_I))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni_I, y_val_uni_I))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
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



print("=====LSTM Model Compile=====")
simple_lstm_model_T.compile(optimizer='adam', loss='mae')
print("Temperature Compile ok")
simple_lstm_model_H.compile(optimizer='adam', loss='mae')
print("Humidity Compile ok")
simple_lstm_model_I.compile(optimizer='adam', loss='mae')
print("Illuminance Compile ok")

# 모델 훈련
EVALUATION_INTERVAL = 5000
EPOCHS = 100

simple_lstm_model_T.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)
# 예측 결과 확인
for x, y in val_univariate.take(5):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      simple_lstm_model_T.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()



simple_lstm_model_H.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)
for x, y in val_univariate.take(5):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      simple_lstm_model_H.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()
    
    
    
simple_lstm_model_I.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)
for x, y in val_univariate.take(5):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      simple_lstm_model_I.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()
    
simple_lstm_model_H.save("model_H")    
simple_lstm_model_I.save("model_I")    
simple_lstm_model_T.save("model_T")

#%%
def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


model = get_model()

# Train the model.
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
plt.plot(test_input, test_target)
plt.show()
model.fit(test_input, test_target)

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("my_model")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.fit(test_input, test_target)

plt.plot(test_input, test_target)
plt.show()