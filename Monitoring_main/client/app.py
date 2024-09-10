from dataclasses import replace
import requests
import json
import random
import time

###############


# 온도/습도/조도를 읽어들이는 코드 위치


###############


import pymysql

# def getIntervalData(year_, month_, day_, hour_, min_, sec_, iter_):
#   db = pymysql.connect(host="192.168.0.3", user="root", passwd="4321", db="sensor_db", charset="utf8")
#   cur = db.cursor()
#   year = year_
#   mm = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#   month = month_
#   day = day_
#   hour = hour_
#   min = 0
#   sec = 0
#   interval = iter_

#   timeData = []
#   temperature = []
#   humidity = []
#   illuminance = []
  
#   cnt = 0
#   while (cnt < 201) :
#     if hour - interval < 1 :
#       if day - 1 == 0 :
#         if month - 1 == 0 :
#           year -= 1
#           month = 12
#           day = mm[month]
#         else :
#           month -= 1
#           day = mm[month]
#       else :
#         day -= 1
#         hour = 24 - abs(hour - interval)
#     else :
#       hour -= interval
#     # print("{}\t : {}/{}/{} {}:{}:{}".format(cnt+1, year, month, day, hour, min, sec))
#     # 1년 전을 기준으로 데이터 200개 수집
#     sql = "SELECT * from sensor_db where Collect_time = DATE_SUB(\'" \
#         + str(year) + "-" + str(month) + "-" + str(day) + " " + str(hour) + ":" + str(min) + ":" + str(sec) \
#         + "\',INTERVAL 12 MONTH);"
#     print(sql)
#     cur.execute(sql)
#     data_list = cur.fetchall()
#     data_list = list(data_list)
#     if len(data_list) > 0 :
#       cnt+=1
#   return timeData, temperature, humidity, illuminance

# td, temp, humi, illu = getIntervalData(2019, 1, 2, 3, 2, 19, 2)       



### db에 csv 파일의 모든 내용 전송
# import csv
# f = open('../AI/dataSet/yangsan.csv', 'r')
# rdr = csv.reader(f)

# cnt = 0
# with open("../AI/dataSet/yangsan.csv", 'r') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#       data ={
#         "time": dict(row)['시간'],
#         "temperature": dict(row)['온도'],
#         "humidity": dict(row)['습도'],
#         "illuminance" : dict(row)['기압']
#       }
#       res = requests.post('http://192.168.0.2:5000/csvToDB', json=data)
      
# with open("../AI/dataSet/yangsan2.csv", 'r') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#       data ={
#         "time": dict(row)['시간'],
#         "temperature": dict(row)['온도'],
#         "humidity": dict(row)['습도'],
#         "illuminance" : dict(row)['기압']
#       }
#       res = requests.post('http://192.168.0.2:5000/csvToDB', json=data)



# requests.adapters.DEFAULT_RETRIES = 5 # increase retries number
# s = requests.session()
# s.keep_alive = False # disable keep alive

# while(True) :

#   # 센서로 입력받은 온도/습도/조도를 최종적으로 저장
#   # 현재는 랜덤값이지만, 위에서 받아온 온도/습도/조도 값으로 변경
#   inputTemperature = random.randint(100, 380) / 10
#   inputHumidity = random.randint(200,600) / 10
#   illuminance = random.randint(10000,10009) / 10
#   errorT = random.randint(1,4)
#   errorH = random.randint(5,15)
  
#   # 전송 데이터 생성
#   print("")
#   print("<====== request ======>")
#   data ={
#     "temperature": inputTemperature,
#     "humidity": inputHumidity,
#     "illuminance" : illuminance,
#     "errorT" : errorT,
#     "errorH" : errorH
#   }
#   print("Send Data : ", data, end="\n\n")

#   # 데이터 전송 및 수신 값 출력
#   # 전송이 올바른 경우 : Sever State(200), Response Data(ok)
#   res = requests.post('http://192.168.0.2:5000/setEnv', json=data)
#   print("<====== response ======>")
#   print("Sever State : ", res.status_code)
#   print("Response Data : ", res.text)
  
#   res = requests.post('http://192.168.0.2:5000/getControl', json="")
#   print("<====== response ======>")
#   print("Sever State : ", res.status_code)
#   print("Response Data : ", res.text)
#   time.sleep(10)
 
# # 서버로 부터 데이터 가져오기
# # 전송이 올바른 경우 : Sever State(200), Response Data(Response Data :  { "humidity": 39, "illuminance": 74, "temperature": 27 })
# res = requests.post('http://192.168.0.2:5000/getValue', json="")
# print("<====== response ======>")
# print("Sever State : ", res.status_code)
# print("Response Data : ", res.text)

# # 전송 데이터 생성
# print("")
# print("<====== request ======>")
# data ={
#   "year": 2022,
#   "month": 1,
#   "day" : 2,
#   "hour": 3,
#   "min" : 0,
#   "sec": 0,
#   "iter" : 3
# }
# print("Send Data : ", data, end="\n\n")

# # 데이터 전송 및 수신 값 출력
# # 전송이 올바른 경우 : Sever State(200), Response Data(ok)
# res = requests.post('http://192.168.0.2:5000/getIntervalValueForLSTM', json=data)
# print("<====== response ======>")
# print("Sever State : ", res.status_code)
# print("Response Data : ", res.text)

# def getAllIntervalData(it, len=0):
#     db = pymysql.connect(host="192.168.0.3", user="root", passwd="4321", db="sensor_db", charset="utf8")
#     cur = db.cursor()
    
#     getTime = "SELECT Collect_time AS HOUR "
#     getTemp = "SELECT temperature AS HOUR "
#     getHumi = "SELECT humidity AS HOUR "
#     getIllu = "SELECT illuminance AS HOUR "
    
#     if it < 1:
#         it = 1
#     if it > 12:
#         it = 12
    
#     mainSQL = "FROM sensor_db WHERE HOUR(Collect_time) % " + str(it) + " = 0 "
    
#     limit = ""
#     if len != 0 :
#         limit = "LIMIT " + str(len)
    
#     timeData = []
#     temperature = []
#     humidity = []
#     illuminance = []
    
#     cur.execute(getTime+mainSQL+limit)
#     data_list = cur.fetchall()
#     timeData = list(data_list)
    
#     cur.execute(getTemp+mainSQL+limit)
#     data_list = cur.fetchall()
#     temperature = list(data_list)
    
#     cur.execute(getHumi+mainSQL+limit)
#     data_list = cur.fetchall()
#     humidity = list(data_list)
    
#     cur.execute(getIllu+mainSQL+limit)
#     data_list = cur.fetchall()
#     illuminance = list(data_list)
    
#     return timeData, temperature, humidity, illuminance
  
# tt, t, h, i = getAllIntervalData(3, 10)

# tt = str(tt).replace("(datetime.datetime","")
# tt = str(tt).replace("),)",")")

# t = str(t).replace(",)","")
# t = str(t).replace("(","")

# h = str(h).replace(",)","")
# h = str(h).replace("(","")

# i = str(i).replace(",)","")
# i = str(i).replace("(","")

# print(tt)
# print(t)
# print(h)
# print(i)


# 예측 데이터 요청
# 전송 데이터 생성
# print("")
# print("<====== request ======>")
# data ={
#   "iter": 1
# }
# print("Send Data : ", data, end="\n\n")

# import time
# start = time.time()  # 시작 시간 저장

# # 데이터 전송 및 수신 값 출력
# # 전송이 올바른 경우 : Sever State(200), Response Data(ok)
# res = requests.post('http://192.168.0.2:5000/getPrediction', json=data)
# print("<====== response ======>")
# print("Sever State : ", res.status_code)
# print("Response Data : ",  res.text)
# print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# def get_prediction():
#   print("")
#   print("<====== request ======>")
#   data ={
#     "iter": 1
#   }
#   print("Send Data : ", data, end="\n\n")

#   import time
#   start = time.time()  # 시작 시간 저장

#   # 데이터 전송 및 수신 값 출력
#   # 전송이 올바른 경우 : Sever State(200), Response Data(ok)
#   res = requests.post('http://192.168.0.2:5000/getPrediction', json=data)
#   print("<====== response ======>")
#   print("Sever State : ", res.status_code)
#   print("Response Data : ",  res.text)
#   print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간


import requests
import json
from datetime import datetime

f = open("./KEY.txt", 'r')
key = f.readline()
f.close()

# year = 2010
month = [0, "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
day = [0, "31", "28", "31", "30", "31", "30", "31", "29", "30", "31", "30", "31"]

for year in range(2022,2023) :
  print(year)
  for i in range(8,9) :
    print(i)
    curT = str(year)+month[i]+day[i]
    yesT = str(year)+month[i]+"26"
    # # GET
    req = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey=' + key + '&numOfRows=800&pageNo=1&dataType=JSON&dataCd=ASOS&dateCd=HR&startDt='+yesT+'&startHh=01&endDt='+curT+'&endHh=23&stnIds=257'
    res = requests.get(req)
          
    dict_string = res.text
    print(dict_string)
    dict = json.loads(dict_string)
    value = dict["response"]["body"]["items"]["item"]
    t = []
    h = []
    i = []

    print(len(value))
    for n in range(0, len(value)) :
      #print(value[n]['tm'], value[n]['ta'], value[n]['hm'], value[n]['pa'])
        # print(value[n]['tm'])
        # print(value[n]['ta'])
        # print(value[n]['hm'])
        # print(value[n]['pa'])
        
      if str(value[n]['ta']) == "" :
        cnt = n
        while(cnt>0) :
          if(str(value[cnt]['ta']) != ""):
            t.append(value[cnt]['ta'])
            break
          cnt -= 1  
        cnt = n
        while(cnt<len(value)-1) :
          if(str(value[cnt]['ta']) != ""):
            t.append(value[cnt]['ta'])
            break
          cnt += 1  
      else :
        t.append(value[n]['ta'])
              
            
      if str(value[n]['hm']) == "" :
        cnt = n
        while(cnt>0) :
          if(str(value[cnt]['hm']) != ""):
            h.append(value[cnt]['hm'])
            break
          cnt -= 1
        cnt = n
        while(cnt<len(value)-1) :
          if(str(value[cnt]['hm']) != ""):
            h.append(value[cnt]['hm'])
            break
          cnt += 1  
      else :
        h.append(value[n]['hm'])
              
            
      if str(value[n]['pa']) == "" :
        cnt = n
        while(cnt>0) :
          if(str(value[cnt]['pa']) != ""):
            i.append(value[cnt]['pa'])
            break
          cnt -= 1
              
        cnt = n
        while(cnt<len(value)-1) :
          if(str(value[cnt]['pa']) != ""):
            i.append(value[cnt]['pa'])
            break
          cnt += 1  
      else :
          i.append(value[n]['pa'])
            
    for n in range(0, len(value)) :
      data ={
        "time": value[n]['tm'],
        "temperature": t[n],
        "humidity": h[n],
        "illuminance" : i[n]
      }
      res = requests.post('http://192.168.0.5:5000/csvToDB', json=data)
