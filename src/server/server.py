import argparse
import math
import csv
import time

from pythonosc import dispatcher
from pythonosc import osc_server


# TODO: semaphore for different handlers.
# TODO: scale up the raw EEG data

'''
    Write EEG data collected from four channels to a csv file.
'''
def csv_write_eeg(ID, ch1, ch2, ch3, ch4):
    with open('/Users/jeanluo/Desktop/temp/eeg_data.csv', 'w') as csvfile:
        fieldnames = ['timestamp', 'ch1', 'ch2','ch3','ch4']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        writer.writerow({'timestamp': ID, 'ch1': ch1, 'ch2': ch2,'ch3': ch3,'ch4': ch4})

'''
    Write quantization data collected from four channels to a csv file. 
'''
def csv_write_quantization(ID, ch1, ch2, ch3, ch4):
    with open('/Users/jeanluo/Desktop/temp/quantization_data.csv', 'w') as csvfile:
        fieldnames = ['timestamp', 'ch1', 'ch2','ch3','ch4']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        writer.writerow({'timestamp': ID, 'ch1': ch1, 'ch2': ch2,'ch3': ch3,'ch4': ch4})

'''
    Get raw EEG data and write to a csv file.
'''
def eeg_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    print("EEG (uV) per channel: ", ch1, ch2, ch3, ch4)
    csv_write_eeg(time.time(),ch1,ch2,ch3,ch4)

'''
    Get EEG quantization data and write to a csv file.
'''
def quantization_handler(unused_addr, args, ch1, ch2, ch3, ch4):
	print("quantization: ", ch1, ch2, ch3, ch4)
	csv_write_quantization(time.time(),ch1,ch2,ch3,ch4)

'''
    Run as main().
'''
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
                        default="127.0.0.1",
                        help="The ip to listen on")
    parser.add_argument("--port",
                        type=int,
                        default=5000,
                        help="The port to listen on")
    args = parser.parse_args()

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/debug", print)
    dispatcher.map("/muse/eeg", eeg_handler, "EEG")
    dispatcher.map("/muse/eeg/quantization", quantization_handler, "Quantization")

    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()

