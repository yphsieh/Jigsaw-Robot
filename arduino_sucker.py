import sys
from time import sleep
import serial

class Arduino_Sucker:

    def __init__(self):
        self.ser = None

    def connect(self):
        COM_PORT = '/dev/ttyACM0'  
        BAUD_RATES = 9600
        
        print("Connected")
        self.ser = serial.Serial(COM_PORT, BAUD_RATES)
        self.suck()
           
    def suck(self):
        print("Sucking.")
        self.ser.write(b'LED_OFF\n') # suck
        sleep(0.5)

    def release(self):
        print("Releasing.")
        self.ser.write(b'LED_ON\n')  # release
        sleep(0.5)