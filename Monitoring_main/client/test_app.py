import requests
from app import get_prediction


def test_get_prediction():
  print("")
  print("<====== request ======>")
  data ={
    "iter": 1
  }
  print("Send Data : ", data, end="\n\n")

  import time
  start = time.time()  # 시작 시간 저장

  # 데이터 전송 및 수신 값 출력
  # 전송이 올바른 경우 : Sever State(200), Response Data(ok)
  res = requests.post('http://192.168.0.2:5000/getPrediction', json=data)
  print("<====== response ======>")
  print("Sever State : ", res.status_code)
  print("Response Data : ",  res.text)
  print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간