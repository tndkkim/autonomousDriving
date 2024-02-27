import RPi.GPIO as GPIO
import time
import motor

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

GPIO_TRIGGER = 11
GPIO_ECHO = 8

GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

def distance():
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.1)
    GPIO.output(GPIO_TRIGGER, False)

    StartTime = time.time()
    StopTime = time.time()

    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()

    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()


    TimeElapsed = StopTime - StartTime

    distance = (TimeElapsed * 34300) / 2

    return distance

if __name__ == '__main__':
    try:
        while True:

            dist = distance()
    #             
            if dist > 7:
                print ("Measured Distance = %.1f cm" % dist)
                print("Foward")
                motor.Forward(12)

            elif dist <= 7:
                print ("Measured Distance = %.1f cm" % dist)
                print("stop")
                motor.Stop()



    except KeyboardInterrupt:
        print("Measurement stopped by User")
        GPIO.cleanup()
