import argparse
import math
import csv
import time
import signal
import sys

from threading import Lock
from pythonosc import dispatcher
from pythonosc import osc_server


# TODO: semaphore for different handlers.
# TODO: scale up the raw EEG data. Done!

# Check http://forum.choosemuse.com/t/quantization/327 to see why we need this
quant_count = 0

quant_ch1 = 1.0
quant_ch2 = 1.0
quant_ch3 = 1.0
quant_ch4 = 1.0

eeg_write_lock = Lock()

csvfile = None

'''
    Get raw EEG data and write to a csv file.
'''
def eeg_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    print("EEG (uV) per channel: ", ch1, ch2, ch3, ch4)
    global quant_count

    eeg_write_lock.acquire()
    if quant_count == 0:
        args[1].writerow({'timestamp':time.time(),'ch1':ch1,'ch2':ch2,'ch3':ch3,'ch4':ch4})
    else:
        args[1].writerow({'timestamp':time.time(),'ch1':ch1*quant_ch1,'ch2':ch2*quant_ch2,'ch3':ch3*quant_ch3,'ch4':ch4*quant_ch4})

    eeg_write_lock.release()
    quant_count = (quant_count+1)%17


'''
    Get EEG quantization data and write to a csv file.
'''
def quantization_handler(unused_addr, args, q_ch1, q_ch2, q_ch3, q_ch4):
    print("quantization: ", q_ch1, q_ch2, q_ch3, q_ch4)
    global quant_ch1
    global quant_ch2
    global quant_ch3
    global quant_ch4

    eeg_write_lock.acquire()
    quant_ch1 = q_ch1
    quant_ch2 = q_ch2
    quant_ch3 = q_ch3
    quant_ch4 = q_ch4
    eeg_write_lock.release()

def exit_pro(sig, frame):
    print()
    if csvfile:
        csvfile.close()
        print('Close csv file writer!')
    print("exit")
    sys.exit(0)

signal.signal(signal.SIGINT, exit_pro)
signal.signal(signal.SIGTERM, exit_pro)

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

    csvfile = open('../data/eeg_data.csv', 'a')
    fieldnames = ['timestamp', 'ch1', 'ch2','ch3','ch4']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/debug", print)
    dispatcher.map("/muse/eeg", eeg_handler, "EEG", writer)
    dispatcher.map("/muse/eeg/quantization", quantization_handler, "Quantization")

    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()

