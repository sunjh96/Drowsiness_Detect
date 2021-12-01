import RPi.GPIO as GPIO
from time import sleep

buzzer=22 #핀번호 22
led=20    #핀번호 20
GPIO.setmode(GPIO.BCM) 
GPIO.setup(buzzer,GPIO.OUT) #GPIO핀을 out용도로 지정
GPIO.setwarnings(False)


def bellAlert(drowsy):
  if drowsy==1:
      GPIO.output(buzzer,GPIO.HIGH) #해당핀 출력
      sleep(10.0)
      print("일어나")
      

def lampAlert(drowsy):
  if drowsy==1:
    for i in range(4):
      GPIO.output(led,GPIO.HIGH)
      sleep(2.0)
      GPIO.output(led,GPIO.LOW)
      sleep(0.5)
      
