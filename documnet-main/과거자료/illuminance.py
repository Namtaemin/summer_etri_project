# sensor data insert to RDS
import RPi.GPIO as GPIO
import sys
import time
import smbus
import threading
import pymysql

I2C_CH =1
BH1750_DEV_ADDR = 0x48

CONT_H_RES_MODE = 0x10
CONT_H_RES_MODE2 = 0x11
CONT_L_RES_MODE = 0x13
ONETIME_H_RES_MODE = 0x20
ONETIME_H_RES_MODE2 = 0x21
ONETIME_L_RES_MODE = 0x23

conn=pymysql.connect(host="localhost",user="sensor_db",passwd="0000",db="sensor_db")

def readIlluminance():
  i2c = smbus.SMBus(I2C_CH)
  luxBytes = i2c.read_i2c_block_data(BH1750_DEV_ADDR, CONT_H_RES_MODE, 2)
  lux = int.from_bytes(luxBytes, byteorder='big')
  i2c.close()
  return lux

Illuminance = readIlluminance()


try:
	with conn.cursor() as cur :
		sql="insert into illumination_sensor values(%s,%s,%s)"
		while True:
			readIlluminance()
			if Illuminance is not None:
				print('Illuminance=%0.1flx'%(Illuminance))
				cur.execute(sql,('Illuminance',time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),readIlluminance()))
				conn.commit()
			else:
				print("Failed to get reading.")
			time.sleep(180)
except KeyboardInterrupt:
	exit()
finally:
	conn.close()

