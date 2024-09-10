# sensor data insert to RDS
import RPi.GPIO as GPIO
import sys
import time
import Adafruit_DHT
import pymysql

sensor = Adafruit_DHT.DHT11
conn=pymysql.connect(host="localhost",user="sensor_db",passwd="0000",db="sensor_db")
pin = 2
try:
	with conn.cursor() as cur :
		sql="insert into sensor_data values(%s,%s,%s,%s)"
		while True:
			humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
			if humidity is not None and temperature is not None:
				print('Temp=%0.1f*C Humidity=%0.1f'%(temperature,humidity))
				cur.execute(sql,('DHT11',time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),temperature, humidity))
				conn.commit()
			else:
				print("Failed to get reading.")
			time.sleep(600)
except KeyboardInterrupt:
	exit()
finally:
	conn.close()
