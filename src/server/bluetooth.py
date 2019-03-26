import serial
import os 
import time

bluetooth =  serial.Serial('/dev/tty.HC-05-DevB', 9600, timeout=10)
# ser.open()
print(bluetooth.write(b'1'))
time.sleep(0.3)
print(bluetooth.write(b'1'))
time.sleep(0.3)
print(bluetooth.write(b'1'))
time.sleep(0.3)
print(bluetooth.write(b'1'))
time.sleep(0.3)
print(bluetooth.write(b'1'))
time.sleep(0.3)
print(bluetooth.write(b'3'))
time.sleep(0.3)
print(bluetooth.write(b'3'))
bluetooth.close()
