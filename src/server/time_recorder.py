import sys
import time
import os

filename = "../data/file.txt"

while(1):
	f = open(filename, "a")
	key=input("Press 'Enter' to record start time: ")
	f.write((str)(time.time())+',')
	key=input("Press 'Enter' to record stop time: ")
	f.write((str)(time.time())+'\n')
	f.close()