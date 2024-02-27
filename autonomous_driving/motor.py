import RPi.GPIO as GPIO
import time
import sys

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

DCpin1 = 32
DCpin2 = 31
DCpin3 = 33
DCpin4 = 29

GPIO.setup(DCpin1, GPIO.OUT)
GPIO.setup(DCpin3, GPIO.OUT)


pwm1 = GPIO.PWM(DCpin1, 100)
pwm2 = GPIO.PWM(DCpin3, 100)
pwm1.stop()
pwm2.stop()

def Forward(speed):
    pwm1.ChangeDutyCycle(speed)
    pwm2.ChangeDutyCycle(speed)

def Right(speed) :
    pwm1.ChangeDutyCycle(speed-10)
    pwm2.ChangeDutyCycle(speed)

def Left(speed) :
    pwm1.ChangeDutyCycle(speed)
    pwm2.ChangeDutyCycle(speed-10)

def Stop():
    pwm2.ChangeDutyCycle(0)
    pwm1.ChangeDutyCycle(0)

if __name__ == '__main__':
    try:
        while True:
            Stop()
 
 
    except KeyboardInterrupt:
        print("Measurement stopped by User")
        pwm1.stop()
        pwm2.stop()
        GPIO.cleanup()
        sys.exit()
        

