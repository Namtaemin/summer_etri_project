from dataclasses import replace
from flask import Flask, request, jsonify, render_template
import random
import _json
import pymysql
import config
from datetime import datetime
import numpy as np
import pandas as pd

t = 0
h = 0
i = 0

# Flask 객체 인스턴트 생성
app = Flask(__name__)

# 접속 URL 설정
@app.route('/')
def index():
    #return 'Hello, Home!'
    return render_template('index.html')

@app.route('/home')
def home():
    return 'Hello, Home!'

@app.route('/user')
def user():
    return 'Hello, User!'



@app.route('/echo_call/<param>') #get echo api
def get_echo_call(param):
    params = request.get_json()
    print(params['param'])
    return jsonify({"param": param})

@app.route('/echo_call', methods=['POST']) #post echo api
def post_echo_call():
    param = request.get_json()
    return jsonify(param)

@app.route('/create', methods=['POST'])
def create():
    print(request.is_json)
    params = request.get_json()
    print(params['user_id'])
    print(params['user_name'])
    return 'ok'


# HTTP 요청을 통해 전달받은 정보를 데이터 베이스에 저장함
def insert_data(params):
    db = pymysql.connect(host="192.168.0.5", user="root", passwd="4321", db="sensor_db", charset="utf8")
    cur = db.cursor()
    #time = str(datetime.today().strftime("%%Y-%m-%d %H:%M:%S"))
    time = str(params['time'])
    time_ = datetime.strptime(time, "%Y-%m-%d %H:%M")
    t = str(params['temperature'])
    h = str(params['humidity'])
    i = str(params['illuminance'])
    
    sql = 'INSERT INTO sensor_db2(collect_time, temperature, humidity, illuminance) VALUES(\"'+ str(time_) + '\", ' + t + ', ' + h + ', ' + i + ')'
    cur.execute(sql)
    db.commit()
    
# 데이터 기준일로부터 원하는 텀을 기준으로 데이터 201개를 가져온다.
def getIntervalData(year_, month_, day_, hour_, min_, sec_, iter_):
  db = pymysql.connect(host="192.168.0.5", user="root", passwd="4321", db="sensor_db", charset="utf8")
  cur = db.cursor()
  year = year_
  mm = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  month = month_
  day = day_
  hour = hour_
  min = 0
  sec = 0
  interval = iter_

  timeData = []
  temperature = []
  humidity = []
  illuminance = []
  
  cnt = 0
  len = 202
  while (cnt < len) :
    if hour - interval < 1 :
      if day - 1 == 0 :
        if month - 1 == 0 :
          year -= 1
          month = 12
          day = mm[month]
        else :
          month -= 1
          day = mm[month]
      else :
        day -= 1
        hour = 24 - abs(hour - interval)
    else :
      hour -= interval
    # print("{}\t : {}/{}/{} {}:{}:{}".format(cnt+1, year, month, day, hour, min, sec))
    # 1년 전을 기준으로 데이터 200개 수집
    sql = "SELECT * from sensor_db2 where Collect_time = DATE_SUB(\'" \
        + str(year) + "-" + str(month) + "-" + str(day) + " " + str(hour) + ":" + str(min) + ":" + str(sec) \
        + "\',INTERVAL 1 HOUR) ORDER BY num DESC LIMIT 202"
    # print(sql)
    cur.execute(sql)
    data_list = cur.fetchall()
    data_list = list(data_list)
    if "datetime.datetime" in str(data_list) :
      cnt+=1
      timeData.append(data_list[0][0])
      temperature.append(data_list[0][1])
      humidity.append(data_list[0][3])
      illuminance.append(data_list[0][2])
  return timeData, temperature, humidity, illuminance

# 데이터 기준일로부터 원하는 텀을 기준으로 데이터 201개를 가져온다.
def getAllIntervalData(it, len=0):
    db = pymysql.connect(host="192.168.0.5", user="root", passwd="4321", db="sensor_db", charset="utf8")
    cur = db.cursor()
    
    getTime = "SELECT Collect_time AS HOUR "
    getTemp = "SELECT temperature AS HOUR "
    getHumi = "SELECT humidity AS HOUR "
    getIllu = "SELECT illuminance AS HOUR "
    
    if it < 1:
        it = 1
    if it > 12:
        it = 12
    
    mainSQL = "FROM sensor_db2 WHERE HOUR(Collect_time) % " + str(it) + " = 0 "
    
    limit = ""
    if len != 0 :
        limit = "LIMIT " + str(len)
    
    timeData = ""
    temperature = ""
    humidity = ""
    illuminance = ""
    
    cur.execute(getTime+mainSQL+limit)
    data_list = cur.fetchall()
    timeData = data_list
    
    cur.execute(getTemp+mainSQL+limit)
    data_list = cur.fetchall()
    temperature = data_list
    
    cur.execute(getHumi+mainSQL+limit)
    data_list = cur.fetchall()
    humidity = data_list
    
    cur.execute(getIllu+mainSQL+limit)
    data_list = cur.fetchall()
    illuminance = data_list
    
    return timeData, temperature, humidity, illuminance


# 라즈베리 파이의 값을 받아옴
@app.route('/raspberry', methods=['POST'])
def raspberry():
    #print(request.is_json)
    params = request.get_json()
    #print("수신 데이터 : ", params)
    
    print("수신 시간 : ", params['dataTime'])
    print("수신 온도 : ", params['temperature'])
    print("수신 습도 : ", params['humidity'])
    print("수신 조도 : ", params['illuminance'])
    #insert_data(params)
    return 'ok'


# csv 파일 내용 DB 저장
@app.route('/csvToDB', methods=['POST'])
def csvtodb():
    #print(request.is_json)
    params = request.get_json()
    print("수신 시간 : ", params['time'])
    print("수신 온도 : ", params['temperature'])
    print("수신 습도 : ", params['humidity'])
    print("수신 조도 : ", params['illuminance'])
    insert_data(params)
    return 'ok'

# 랜덤(온도/조도/습도 반환)
@app.route('/getRandom', methods=['POST'])
def getRandom():
    t = random.randint(26,38)
    h = random.randint(20,60)
    i = random.randint(10,80)
    json_string = {
        "dateTime": "datetime.datetime(2021, 7, 7, 17, 0)",
        "temperature": t,
        "humidity": h,
        "illuminance": i
    }
    
    print("전송 데이터 : ", json_string)
    return jsonify(json_string)


# 랜덤(온도/조도/습도 반환)
@app.route('/getRandom2', methods=['POST'])
def getRandom2():
    t = random.randint(26,38)
    h = random.randint(20,60)
    i = random.randint(10,80)
    json_string = {
        "dateTime": "datetime.datetime(2021, 7, 7, 17, 0)",
        "temperature": t,
        "humidity": h,
        "illuminance": i
    }
    print("전송 데이터 : ", json_string)
    return jsonify(json_string)

# 라즈베리 파이의 값을 전송함
@app.route('/getValue', methods=['POST'])
def getValue():
    db = pymysql.connect(host="192.168.0.5", user="root", passwd="4321", db="sensor_db", charset="utf8")
    cur = db.cursor()
    sql = "SELECT * FROM sensor_db2 ORDER BY num DESC LIMIT 1"
    cur.execute(sql)
    data_list = cur.fetchall()
    data_list = list(data_list)
    
    json_string = {
        "dataTime" : data_list[0][0],
        "temperature": data_list[0][1],
        "humidity": data_list[0][3],
        "illuminance": data_list[0][2]
    }
    
    print("전송 데이터 : ", json_string)
    return jsonify(json_string)


# 라즈베리 파이의 값을 LSTM을 위해 200개 획득
@app.route('/getValueForLSTM', methods=['POST'])
def getValueForLSTM():
    db = pymysql.connect(host="192.168.0.5", user="root", passwd="4321", db="sensor_db", charset="utf8")
    cur = db.cursor()
    sql = "SELECT * FROM sensor_db2 ORDER BY num DESC LIMIT 202"
    cur.execute(sql)
    data_list = cur.fetchall()
    data_list = list(data_list)
    
    time = []
    temp = []
    humi = []
    illu = []
    for i in range(202):
        time.append(data_list[i][0])
        temp.append(data_list[i][1])
        humi.append(data_list[i][2])
        illu.append(data_list[i][3])
    print(np.shape(time))
    json_string = {
        "dataTime" : time,
        "temperature": temp,
        "humidity": humi,
        "illuminance": illu
    }
    
    print("전송 데이터 : ", json_string)
    return jsonify(json_string)



# 라즈베리 파이의 값을 원하는 interval로 전체 반환
@app.route('/getAllIntervalValue', methods=['POST'])
def getAllIntervalValue():
    params = request.get_json()
    time, temp, humi, illu = getAllIntervalData(int(params['iter']), int(params['len']))

    json_string = {
        "dataTime" : time,
        "temperature": temp,
        "humidity": humi,
        "illuminance": illu
    }
    
    #print("전송 데이터 : ", json_string)
    return jsonify(json_string)
    

# 라즈베리 파이의 값을 LSTM을 위해 200개 획득
@app.route('/getIntervalValueForLSTM', methods=['POST'])
def getIntervalValueForLSTM():
    params = request.get_json()
    # year_, month_, day_, hour_, min_, sec_, iter_
    # timeData, temperature, humidity, illuminance
    time, temp, humi, illu = getIntervalData(int(params['year']), int(params['month']), int(params['day']), 
                                             int(params['hour']), int(params['min']), int(params['sec']), int(params['iter']))
    json_string = {
        "dataTime" : time,
        "temperature": temp,
        "humidity": humi,
        "illuminance": illu
    }
    
    #print("전송 데이터 : ", json_string['dataTime'])
    return jsonify(json_string)


def getData(interval):
    from datetime import datetime
    now = datetime.now()
    # year_, month_, day_, hour_, min_, sec_, iter_
    # timeData, temperature, humidity, illuminance
    inputHour = int(str(now.hour)) + interval + 2
    if inputHour > 23 :
        inputHour = 23
        
    time, temp, humi, illu = getIntervalData(int(str(now.year)), int(str(now.month)), int(str(now.day))-1, 
                                             int(inputHour), int(str(now.minute)), int(str(now.second)), int(str(interval)))
    
    
    import  json
    json_string = {
        "dataTime" : time,
        "temperature": temp,
        "humidity": humi,
        "illuminance" : illu
    }
    
    json_string = str(json_string).replace("datetime.datetime(", "\"")
    json_string = str(json_string).replace(")", "\"")
    json_string = str(json_string).replace("\\", "")
    return json.dumps(json_string)
    
    

# 라즈베리 파이의 값을 LSTM을 위해 200개 획득
@app.route('/getPrediction', methods=['POST'])
def getPrediction():
    params = request.get_json()

    interval = int(params['iter'])
    if(interval < 1):
        interval = 1
    if(interval > 12):
        interval = 12
    if(interval >= 9 or interval == 7) :
        interval = 1

    data = getData(interval)

    # 정규화에 사용되었던 평균/표준편차값
    meanListT =[0, 15.053684090161944, 15.071946826758147, 15.075860738326975, 15.070062310638539, 15.060954863492933, 15.028854398902418, 7, 14.993860050308712, 9, 10, 11, 12]
    stdListT=[0, 9.61201460854714, 15.071946826758147, 15.075860738326975, 15.070062310638539, 15.060954863492933, 15.028854398902418, 7, 14.993860050308712, 9, 10, 11, 12]
    
    meanListH =[0, 63.67404628160599, 63.81444539736992, 63.81619002701196, 63.89820499628423, 63.82233502538071, 63.88243868976162, 7, 64.36873999542648, 9, 10, 11, 12]
    stdListH=[0, 20.392235502122855, 20.39348453023059, 20.352355353113452, 20.42426383007355, 20.239124485697847, 20.622298180702046, 7, 20.324600635201165, 9, 10, 11, 12]
    
    meanListI =[0, 1015.3450066463703, 1015.3174070897655, 1015.3172705055097, 1015.31412565026, 1015.3317876251887, 1015.2948893843252, 7, 1015.3146009604391, 9, 10, 11, 12]
    stdListI=[0, 7.642627412799711, 7.7253670772751075, 7.732552113859898, 7.7210505647301, 7.725728058270826, 7.701605644158868, 7, 7.723022100845774, 9, 10, 11, 12]
    
    meanT = meanListT[interval]
    stdT  = stdListT[interval]

    meanH = meanListH[interval]
    stdH  = stdListH[interval]

    meanI = meanListI[interval]
    stdI  = stdListI[interval]

    import json
    t = json.loads(data)
    
    # 데이터를 json 형태로 변환 후 각 값을 나눔
    dataT = eval(t)['temperature']
    dataH = eval(t)['humidity']
    dataI = eval(t)['illuminance']
    
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
    

    import joblib
    path = "../AI/model_"
    # 모델 불러오기
    modelT = joblib.load(path+"t_"+str(interval)+".pkl")
    modelH = joblib.load(path+"h_"+str(interval)+".pkl")
    modelI = joblib.load(path+"i_"+str(interval)+".pkl")

    import tensorflow as tf
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
    
    json_string = {
        "dataTime" : str(0000),
        "temperature": str(reT[0][0]),
        "humidity": str(reH[0][0]),
        "illuminance": str(reI[0][0])
    }
    
    return jsonify(json_string)

    
    
@app.route('/result',methods = ['POST'])
def result():
    # result URL 로 접근했을 때, form 데이터는 request 객체를 이용해 전달받고
    # 이를 다시 8_sending_form_data_result.html 렌더링하면서 넘겨준다.
        
    result = request.form
    print(result)
        
        
    interval = int(result['button-num'])
    if(interval < 1):
        interval = 1
    if(interval > 12):
        interval = 12
    if(interval >= 9 or interval == 7) :
        interval = 1

    data = getData(interval)

    # 정규화에 사용되었던 평균/표준편차값
    # meanListT =[0, 15.053684090161944, 15.071946826758147, 15.075860738326975, 15.070062310638539, 15.060954863492933, 15.028854398902418, 7, 14.993860050308712, 9, 10, 11, 12]
    # stdListT=[0, 9.61201460854714, 15.071946826758147, 15.075860738326975, 15.070062310638539, 15.060954863492933, 15.028854398902418, 7, 14.993860050308712, 9, 10, 11, 12]
    
    # meanListH =[0, 63.67404628160599, 63.81444539736992, 63.81619002701196, 63.89820499628423, 63.82233502538071, 63.88243868976162, 7, 64.36873999542648, 9, 10, 11, 12]
    # stdListH=[0, 20.392235502122855, 20.39348453023059, 20.352355353113452, 20.42426383007355, 20.239124485697847, 20.622298180702046, 7, 20.324600635201165, 9, 10, 11, 12]
    
    # meanListI =[0, 1015.3450066463703, 1015.3174070897655, 1015.3172705055097, 1015.31412565026, 1015.3317876251887, 1015.2948893843252, 7, 1015.3146009604391, 9, 10, 11, 12]
    # stdListI=[0, 7.642627412799711, 7.7253670772751075, 7.732552113859898, 7.7210505647301, 7.725728058270826, 7.701605644158868, 7, 7.723022100845774, 9, 10, 11, 12]

    # meanT = meanListT[interval]
    # stdT  = stdListT[interval]

    # meanH = meanListH[interval]
    # stdH  = stdListH[interval]

    # meanI = meanListI[interval]
    # stdI  = stdListI[interval]

    import json
    t = json.loads(data)
    
    # 데이터를 json 형태로 변환 후 각 값을 나눔
    dateT = eval(t)['dataTime']
    dataT = eval(t)['temperature']
    dataH = eval(t)['humidity']
    dataI = eval(t)['illuminance']
    
    meanT = np.mean(dataT)
    stdT  = np.std(dataT)
    
    meanH = np.mean(dataH)
    stdH  = np.std(dataH)
    
    meanI = np.mean(dataI)
    stdI  = np.std(dataI)
    
    
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
    

    import joblib
    path = "../AI/model_"
    # 모델 불러오기
    modelT = joblib.load(path+"t_"+str(interval)+".pkl")
    modelH = joblib.load(path+"h_"+str(interval)+".pkl")
    modelI = joblib.load(path+"i_"+str(interval)+".pkl")

    import tensorflow as tf
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
    
    json_string = {
        "예측 미래" : str(interval) + "시간 후",
        "기준 데이터" : str(dateT[0]).replace(", ", "-"),
        "온도": str(reT[0][0]),
        "습도": str(reH[0][0]),
        "기압": str(reI[0][0])
    }
    
        
    return render_template("result.html",result = json_string)

    

if __name__ == '__main__':
    # 코드 수정 시 자동 반영 
    app.run(debug=True, host="192.168.0.5", port=5000)