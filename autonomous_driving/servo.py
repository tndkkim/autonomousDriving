#import RPi.GPIO as GPIO
import time

pin =11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(pin, GPIO.OUT)
p= GPIO.PWM(pin, 50)
p.start(7)
cnt = 0

def HardTurnright():
    return p.ChangeDutyCycle(8.9)

def Turnright():
    return p.ChangeDutyCycle(8.3)

def Go():
    return p.ChangeDutyCycle(7.8)

def Turnleft():
    return p.ChangeDutyCycle(7.2)

def HardTurnleft():
    return p.ChangeDutyCycle(6.7)

if __name__ == '__main__':
    try:
        while True:
            p.ChangeDutyCycle(12.5)
            time.sleep(10)
            
    except KeybordInterrupt:
         p.stop()
         GPIO.cleanup()
