import argparse
import math
import numpy as np
import csv
import time
import signal
import sys
import queue
import time
import serial

from threading import Lock, Thread
from pythonosc import dispatcher
from pythonosc import osc_server
from utility import DTModelBinary
from utility import DTModelMulti
from utility import SVMModelBinary
from utility import SVMModelMulti
from sender import BluetoothSender

# Check http://forum.choosemuse.com/t/quantization/327 to see why we need this
quant_count = 0

quant_ch1 = 1.0
quant_ch2 = 1.0
quant_ch3 = 1.0
quant_ch4 = 1.0

alpha_absolute = queue.Queue(400)
beta_absolute = queue.Queue(400)
delta_absolute = queue.Queue(400)
theta_absolute = queue.Queue(400)
gamma_absolute = queue.Queue(400)

# Get relative data, five types of signals.
alpha_relative = queue.Queue(400)
beta_relative = queue.Queue(400)
delta_relative = queue.Queue(400)
theta_relative = queue.Queue(400)
gamma_relative = queue.Queue(400)

# Get session data, five types of signals.
alpha_session_score = queue.Queue(400)
beta_session_score = queue.Queue(400)
delta_session_score = queue.Queue(400)
theta_session_score = queue.Queue(400)
gamma_session_score = queue.Queue(400)


eeg_write_lock = Lock()

csvfile_eeg = None
csvfile_power = None

power_receiver = None
bluetooth = None
sender = None


'''
    Get raw EEG data and write to a csv file.
'''
def eeg_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    print("EEG (uV) per channel: ", ch1, ch2, ch3, ch4)
    global quant_count

    eeg_write_lock.acquire()
    if quant_count == 0:
        args[1].writerow({'timestamp': time.time(), 'ch1': ch1, 'ch2': ch2, 'ch3': ch3, 'ch4': ch4})
    else:
        args[1].writerow({'timestamp': time.time(), 'ch1': ch1 * quant_ch1, 'ch2': ch2 * quant_ch2, 'ch3': ch3 * quant_ch3, 'ch4': ch4 * quant_ch4})

    eeg_write_lock.release()
    quant_count = (quant_count + 1) % 17


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


# Absolute Band Power data handler
def alpha_absolute_handler(unused_addr, args, alpha_abs_ch1, alpha_abs_ch2, alpha_abs_ch3, alpha_abs_ch4):
    global alpha_absolute
    alpha_absolute.put(alpha_abs_ch1)
    alpha_absolute.put(alpha_abs_ch2)
    alpha_absolute.put(alpha_abs_ch3)
    alpha_absolute.put(alpha_abs_ch4)

def beta_absolute_handler(unused_addr, args, beta_abs_ch1, beta_abs_ch2, beta_abs_ch3, beta_abs_ch4):
    global beta_absolute
    beta_absolute.put(beta_abs_ch1)
    beta_absolute.put(beta_abs_ch2)
    beta_absolute.put(beta_abs_ch3)
    beta_absolute.put(beta_abs_ch4)

def delta_absolute_handler(unused_addr, args, delta_abs_ch1, delta_abs_ch2, delta_abs_ch3, delta_abs_ch4):
    global delta_absolute
    delta_absolute.put(delta_abs_ch1)
    delta_absolute.put(delta_abs_ch2)
    delta_absolute.put(delta_abs_ch3)
    delta_absolute.put(delta_abs_ch4)

def theta_absolute_handler(unused_addr, args, theta_abs_ch1, theta_abs_ch2, theta_abs_ch3, theta_abs_ch4):
    global theta_absolute
    theta_absolute.put(theta_abs_ch1)
    theta_absolute.put(theta_abs_ch2)
    theta_absolute.put(theta_abs_ch3)
    theta_absolute.put(theta_abs_ch4)

def gamma_absolute_handler(unused_addr, args, gamma_abs_ch1, gamma_abs_ch2, gamma_abs_ch3, gamma_abs_ch4):
    global gamma_absolute
    gamma_absolute.put(gamma_abs_ch1)
    gamma_absolute.put(gamma_abs_ch2)
    gamma_absolute.put(gamma_abs_ch3)
    gamma_absolute.put(gamma_abs_ch4)

# Relative Band Powers data handler
def alpha_relative_handler(unused_addr, args, alpha_rel_ch1, alpha_rel_ch2, alpha_rel_ch3, alpha_rel_ch4):
    global alpha_relative
    alpha_relative.put(alpha_rel_ch1)
    alpha_relative.put(alpha_rel_ch2)
    alpha_relative.put(alpha_rel_ch3)
    alpha_relative.put(alpha_rel_ch4)

def beta_relative_handler(unused_addr, args, beta_rel_ch1, beta_rel_ch2, beta_rel_ch3, beta_rel_ch4):
    global beta_relative
    # print(beta_rel_ch1, beta_rel_ch2, beta_rel_ch3, beta_rel_ch4)
    beta_relative.put(beta_rel_ch1)
    beta_relative.put(beta_rel_ch2)
    beta_relative.put(beta_rel_ch3)
    beta_relative.put(beta_rel_ch4)

def delta_relative_handler(unused_addr, args, delta_rel_ch1, delta_rel_ch2, delta_rel_ch3, delta_rel_ch4):
    global delta_relative
    delta_relative.put(delta_rel_ch1)
    delta_relative.put(delta_rel_ch2)
    delta_relative.put(delta_rel_ch3)
    delta_relative.put(delta_rel_ch4)

def theta_relative_handler(unused_addr, args, theta_rel_ch1, theta_rel_ch2, theta_rel_ch3, theta_rel_ch4):
    global theta_relative
    theta_relative.put(theta_rel_ch1)
    theta_relative.put(theta_rel_ch2)
    theta_relative.put(theta_rel_ch3)
    theta_relative.put(theta_rel_ch4)

def gamma_relative_handler(unused_addr, args, gamma_rel_ch1, gamma_rel_ch2, gamma_rel_ch3, gamma_rel_ch4):
    global gamma_relative
    gamma_relative.put(gamma_rel_ch1)
    gamma_relative.put(gamma_rel_ch2)
    gamma_relative.put(gamma_rel_ch3)
    gamma_relative.put(gamma_rel_ch4)

# session score data handler
def alpha_session_score_handler(unused_addr, args, alpha_sess_ch1, alpha_sess_ch2, alpha_sess_ch3, alpha_sess_ch4):
    # print(alpha_sess_ch1, alpha_sess_ch2, alpha_sess_ch3,alpha_sess_ch4)
    global alpha_session_score
    alpha_session_score.put(alpha_sess_ch1)
    alpha_session_score.put(alpha_sess_ch2)
    alpha_session_score.put(alpha_sess_ch3)
    alpha_session_score.put(alpha_sess_ch4)

def beta_session_score_handler(unused_addr, args, beta_sess_ch1, beta_sess_ch2, beta_sess_ch3, beta_sess_ch4):
    global beta_session_score
    beta_session_score.put(beta_sess_ch1)
    beta_session_score.put(beta_sess_ch2)
    beta_session_score.put(beta_sess_ch3)
    beta_session_score.put(beta_sess_ch4)

def delta_session_score_handler(unused_addr, args, delta_sess_ch1, delta_sess_ch2, delta_sess_ch3, delta_sess_ch4):
    global delta_session_score
    delta_session_score.put(delta_sess_ch1)
    delta_session_score.put(delta_sess_ch2)
    delta_session_score.put(delta_sess_ch3)
    delta_session_score.put(delta_sess_ch4)

def theta_session_score_handler(unused_addr, args, theta_sess_ch1, theta_sess_ch2, theta_sess_ch3, theta_sess_ch4):
    global theta_session_score
    theta_session_score.put(theta_sess_ch1)
    theta_session_score.put(theta_sess_ch2)
    theta_session_score.put(theta_sess_ch3)
    theta_session_score.put(theta_sess_ch4)

def gamma_session_score_handler(unused_addr, args, gamma_sess_ch1, gamma_sess_ch2, gamma_sess_ch3, gamma_sess_ch4):
    global gamma_session_score
    gamma_session_score.put(gamma_sess_ch1)
    gamma_session_score.put(gamma_sess_ch2)
    gamma_session_score.put(gamma_sess_ch3)
    gamma_session_score.put(gamma_sess_ch4)

class PowerBandsReceiver(Thread):
    def __init__(self, fileWriter):
        Thread.__init__(self)
        self.shoudStop = False
        self.writer = fileWriter
        self.model = SVMModelBinary("../model/")
        self.segment_size = 3

    def stop(self):
        self.shoudStop = True

    def run(self):
        global alpha_absolute
        global beta_absolute
        global delta_absolute
        global theta_absolute
        global gamma_absolute

        global alpha_relative
        global beta_relative
        global delta_relative
        global theta_relative
        global gamma_relative

        global alpha_session_score
        global beta_session_score
        global delta_session_score
        global theta_session_score
        global gamma_session_score

        data = []

        while not self.shoudStop:
            # print("here")
            a_a_ch1 = round(alpha_absolute.get(), 9)
            a_a_ch2 = round(alpha_absolute.get(), 9)
            a_a_ch3 = round(alpha_absolute.get(), 9)
            a_a_ch4 = round(alpha_absolute.get(), 9)

            b_a_ch1 = round(beta_absolute.get(), 9)
            b_a_ch2 = round(beta_absolute.get(), 9)
            b_a_ch3 = round(beta_absolute.get(), 9)
            b_a_ch4 = round(beta_absolute.get(), 9)

            d_a_ch1 = round(delta_absolute.get(), 9)
            d_a_ch2 = round(delta_absolute.get(), 9)
            d_a_ch3 = round(delta_absolute.get(), 9)
            d_a_ch4 = round(delta_absolute.get(), 9)

            t_a_ch1 = round(theta_absolute.get(), 9)
            t_a_ch2 = round(theta_absolute.get(), 9)
            t_a_ch3 = round(theta_absolute.get(), 9)
            t_a_ch4 = round(theta_absolute.get(), 9)

            g_a_ch1 = round(gamma_absolute.get(), 9)
            g_a_ch2 = round(gamma_absolute.get(), 9)
            g_a_ch3 = round(gamma_absolute.get(), 9)
            g_a_ch4 = round(gamma_absolute.get(), 9)

            a_r_ch1 = round(alpha_relative.get(), 9)
            a_r_ch2 = round(alpha_relative.get(), 9)
            a_r_ch3 = round(alpha_relative.get(), 9)
            a_r_ch4 = round(alpha_relative.get(), 9)

            b_r_ch1 = round(beta_relative.get(), 9)
            b_r_ch2 = round(beta_relative.get(), 9)
            b_r_ch3 = round(beta_relative.get(), 9)
            b_r_ch4 = round(beta_relative.get(), 9)

            d_r_ch1 = round(delta_relative.get(), 9)
            d_r_ch2 = round(delta_relative.get(), 9)
            d_r_ch3 = round(delta_relative.get(), 9)
            d_r_ch4 = round(delta_relative.get(), 9)

            t_r_ch1 = round(theta_relative.get(), 9)
            t_r_ch2 = round(theta_relative.get(), 9)
            t_r_ch3 = round(theta_relative.get(), 9)
            t_r_ch4 = round(theta_relative.get(), 9)

            g_r_ch1 = round(gamma_relative.get(), 9)
            g_r_ch2 = round(gamma_relative.get(), 9)
            g_r_ch3 = round(gamma_relative.get(), 9)
            g_r_ch4 = round(gamma_relative.get(), 9)

            a_s_ch1 = round(alpha_session_score.get(), 9)
            a_s_ch2 = round(alpha_session_score.get(), 9)
            a_s_ch3 = round(alpha_session_score.get(), 9)
            a_s_ch4 = round(alpha_session_score.get(), 9)

            b_s_ch1 = round(beta_session_score.get(), 9)
            b_s_ch2 = round(beta_session_score.get(), 9)
            b_s_ch3 = round(beta_session_score.get(), 9)
            b_s_ch4 = round(beta_session_score.get(), 9)

            d_s_ch1 = round(delta_session_score.get(), 9)
            d_s_ch2 = round(delta_session_score.get(), 9)
            d_s_ch3 = round(delta_session_score.get(), 9)
            d_s_ch4 = round(delta_session_score.get(), 9)

            t_s_ch1 = round(theta_session_score.get(), 9)
            t_s_ch2 = round(theta_session_score.get(), 9)
            t_s_ch3 = round(theta_session_score.get(), 9)
            t_s_ch4 = round(theta_session_score.get(), 9)

            g_s_ch1 = round(gamma_session_score.get(), 9)
            g_s_ch2 = round(gamma_session_score.get(), 9)
            g_s_ch3 = round(gamma_session_score.get(), 9)
            g_s_ch4 = round(gamma_session_score.get(), 9)

            # print("g_s_ch1:",g_s_ch1)

            if not self.shoudStop:
                self.writer.writerow({'timestamp':time.time(),
                                    'alpha_absolute_ch1':a_a_ch1,'alpha_absolute_ch2':a_a_ch2,'alpha_absolute_ch3':a_a_ch3,'alpha_absolute_ch4':a_a_ch4,
                                    'beta_absolute_ch1':b_a_ch1,'beta_absolute_ch2':b_a_ch2,'beta_absolute_ch3':b_a_ch3,'beta_absolute_ch4':b_a_ch4,
                                    'delta_absolute_ch1':d_a_ch1,'delta_absolute_ch2':d_a_ch2,'delta_absolute_ch3':d_a_ch3,'delta_absolute_ch4':d_a_ch4,
                                    'theta_absolute_ch1':t_a_ch1,'theta_absolute_ch2':t_a_ch2,'theta_absolute_ch3':t_a_ch3,'theta_absolute_ch4':t_a_ch4,
                                    'gamma_absolute_ch1':g_a_ch1,'gamma_absolute_ch2':g_a_ch2,'gamma_absolute_ch3':g_a_ch3,'gamma_absolute_ch4':g_a_ch4,
                                    'alpha_relative_ch1':a_r_ch1, 'alpha_relative_ch2':a_r_ch2,'alpha_relative_ch3':a_r_ch3,'alpha_relative_ch4':a_r_ch4,
                                    'beta_relative_ch1':b_r_ch1,'beta_relative_ch2':b_r_ch2,'beta_relative_ch3':b_r_ch3,'beta_relative_ch4':b_r_ch4,
                                    'delta_relative_ch1':d_r_ch1, 'delta_relative_ch2':d_r_ch2,'delta_relative_ch3':d_r_ch3,'delta_relative_ch4':d_r_ch4,
                                    'theta_relative_ch1':t_r_ch1, 'theta_relative_ch2':t_r_ch2, 'theta_relative_ch3':t_r_ch3, 'theta_relative_ch4':t_r_ch4,
                                    'gamma_relative_ch1':g_r_ch1, 'gamma_relative_ch2':g_r_ch2, 'gamma_relative_ch3':g_r_ch3, 'gamma_relative_ch4':g_r_ch4,
                                    'alpha_session_score_ch1':a_s_ch1,'alpha_session_score_ch2':a_s_ch2,'alpha_session_score_ch3':a_s_ch3,'alpha_session_score_ch4':a_s_ch4,
                                    'beta_session_score_ch1':b_s_ch1,'beta_session_score_ch2':b_s_ch2,'beta_session_score_ch3':b_s_ch3,'beta_session_score_ch4':b_s_ch4,
                                    'delta_session_score_ch1':d_s_ch1, 'delta_session_score_ch2':d_s_ch2, 'delta_session_score_ch3':d_s_ch3, 'delta_session_score_ch4':d_s_ch4,
                                    'theta_session_score_ch1':t_s_ch1,'theta_session_score_ch2':t_s_ch2,'theta_session_score_ch3':t_s_ch3,'theta_session_score_ch4':t_s_ch4,
                                    'gamma_session_score_ch1':g_s_ch1,'gamma_session_score_ch2':g_s_ch2,'gamma_session_score_ch3':g_s_ch3,'gamma_session_score_ch4':g_s_ch4})
                data.append([a_a_ch1,a_a_ch2,a_a_ch3,a_a_ch4,
                             b_a_ch1,b_a_ch2,b_a_ch3,b_a_ch4,
                             d_a_ch1,d_a_ch2,d_a_ch3,d_a_ch4,
                             t_a_ch1,t_a_ch2,t_a_ch3,t_a_ch4,
                             g_a_ch1,g_a_ch2,g_a_ch3,g_a_ch4,
                             # a_r_ch1,a_r_ch2,a_r_ch3,a_r_ch4,
                             # b_r_ch1,b_r_ch2,b_r_ch3,b_r_ch4,
                             # d_r_ch1,d_r_ch2,d_r_ch3,d_r_ch4,
                             # t_r_ch1,t_r_ch2,t_r_ch3,t_r_ch4,
                             # g_r_ch1,g_r_ch2,g_r_ch3,g_r_ch4,
                             a_s_ch1,a_s_ch2,a_s_ch3,a_s_ch4,
                             b_s_ch1,b_s_ch2,b_s_ch3,b_s_ch4,
                             d_s_ch1,d_s_ch2,d_s_ch3,d_s_ch4,
                             t_s_ch1,t_s_ch2,t_s_ch3,t_s_ch4,
                             g_s_ch1,g_s_ch2,g_s_ch3,g_s_ch4
                            ])
                if len(data) == self.segment_size:
                    feature = np.sum(data, axis=0)
                    feature = np.expand_dims(feature, axis=0)  # sklearn needs a 2-D input data shape
                    bt = time.time()
                    y = self.model.predict(feature)
                    st = time.time()
                    if y == 0:
                        print("stop,%f" % (st - bt))
                        sender.send(b'0')
                    elif y == 1:
                        print("left,%f" % (st - bt))
                        sender.send(b'1')
                    elif y == 2:
                        print("right,%f" % (st - bt))
                        sender.send(b'2')
                    elif y == 3:
                        print("forward,%f" % (st - bt))
                        sender.send(b'3')
                    else:
                        print("none,%f" % (st - bt))
                        sender.send(b'0')

                    data = []

                # self.writer.writerow({'timestamp':time.time(),
                #                     'alpha_absolute_ch1':a_a_ch1,'alpha_absolute_ch2':a_a_ch2,'alpha_absolute_ch3':a_a_ch3,'alpha_absolute_ch4':a_a_ch4,
                #                     'beta_absolute_ch1':b_a_ch1,'beta_absolute_ch2':b_a_ch2,'beta_absolute_ch3':b_a_ch3,'beta_absolute_ch4':b_a_ch4,
                #                     'delta_absolute_ch1':d_a_ch1,'delta_absolute_ch2':d_a_ch2,'delta_absolute_ch3':d_a_ch3,'delta_absolute_ch4':d_a_ch4,
                #                     'theta_absolute_ch1':t_a_ch1,'theta_absolute_ch2':t_a_ch2,'theta_absolute_ch3':t_a_ch3,'theta_absolute_ch4':t_a_ch4,
                #                     'gamma_absolute_ch1':g_a_ch1,'gamma_absolute_ch2':g_a_ch2,'gamma_absolute_ch3':g_a_ch3,'gamma_absolute_ch4':g_a_ch4,
                #                     'alpha_relative_ch1':a_r_ch1, 'alpha_relative_ch2':a_r_ch2,'alpha_relative_ch3':a_r_ch3,'alpha_relative_ch4':a_r_ch4,
                #                     'beta_relative_ch1':b_r_ch1,'beta_relative_ch2':b_r_ch2,'beta_relative_ch3':b_r_ch3,'beta_relative_ch4':b_r_ch4,
                #                     'delta_relative_ch1':d_r_ch1, 'delta_relative_ch2':d_r_ch2,'delta_relative_ch3':d_r_ch3,'delta_relative_ch4':d_r_ch4,
                #                     'theta_relative_ch1':t_r_ch1, 'theta_relative_ch2':t_r_ch2, 'theta_relative_ch3':t_r_ch3, 'theta_relative_ch4':t_r_ch4,
                #                     'gamma_relative_ch1':g_r_ch1, 'gamma_relative_ch2':g_r_ch2, 'gamma_relative_ch3':g_r_ch3, 'gamma_relative_ch4':g_r_ch4,
                #                     'alpha_session_score_ch1':a_s_ch1,'alpha_session_score_ch2':a_s_ch2,'alpha_session_score_ch3':a_s_ch3,'alpha_session_score_ch4':a_s_ch4,
                #                     'beta_session_score_ch1':b_s_ch1,'beta_session_score_ch2':b_s_ch2,'beta_session_score_ch3':b_s_ch3,'beta_session_score_ch4':b_s_ch4,
                #                     'delta_session_score_ch1':d_s_ch1, 'delta_session_score_ch2':d_s_ch2, 'delta_session_score_ch3':d_s_ch3, 'delta_session_score_ch4':d_s_ch4,
                #                     'theta_session_score_ch1':t_s_ch1,'theta_session_score_ch2':t_s_ch2,'theta_session_score_ch3':t_s_ch3,'theta_session_score_ch4':t_s_ch4,
                #                     'gamma_session_score_ch1':g_s_ch1,'gamma_session_score_ch2':g_s_ch2,'gamma_session_score_ch3':g_s_ch3,'gamma_session_score_ch4':g_s_ch4})


def exit_pro(sig, frame):
    # global alpha_absolute
    # global beta_absolute
    # global delta_absolute
    # global theta_absolute
    # global gamma_absolute

    # global alpha_relative
    # global beta_relative
    # global delta_relative
    # global theta_relative
    # global gamma_relative

    # global alpha_session_score
    # global beta_session_score
    # global delta_session_score
    # global theta_session_score
    # global gamma_session_score

    print()
    power_receiver.stop()

    list(map(alpha_absolute.put, [0, 0, 0, 0]))
    list(map(beta_absolute.put, [0, 0, 0, 0]))
    list(map(delta_absolute.put, [0, 0, 0, 0]))
    list(map(theta_absolute.put, [0, 0, 0, 0]))
    list(map(gamma_absolute.put, [0, 0, 0, 0]))

    list(map(alpha_relative.put, [0, 0, 0, 0]))
    list(map(beta_relative.put, [0, 0, 0, 0]))
    list(map(delta_relative.put, [0, 0, 0, 0]))
    list(map(theta_relative.put, [0, 0, 0, 0]))
    list(map(gamma_relative.put, [0, 0, 0, 0]))

    list(map(alpha_session_score.put, [0, 0, 0, 0]))
    list(map(beta_session_score.put, [0, 0, 0, 0]))
    list(map(delta_session_score.put, [0, 0, 0, 0]))
    list(map(theta_session_score.put, [0, 0, 0, 0]))
    list(map(gamma_session_score.put, [0, 0, 0, 0]))

    if csvfile_eeg:
        csvfile_eeg.close()
        print('Close eeg csv file writer!')
    if csvfile_power:
        csvfile_power.close()
        print('Close power csv file writer!')
    print("exit")
    power_receiver.join()

    if bluetooth:
        print("Close bluetooth")
        bluetooth.write(b'0')
        bluetooth.close()
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
    parser.add_argument("--filename_eeg",
                        default="eeg_data.csv",
                        help="the name of the file to save data")
    parser.add_argument("--filename_power",
                        default="power_data.csv",
                        help="the name of the file to save data")
    args = parser.parse_args()

    # EEG File writers
    # csvfile_eeg = open('../data/'+args.filename_eeg, 'a')
    # fieldnames_eeg = ['timestamp', 'ch1', 'ch2','ch3','ch4']
    # writer_eeg = csv.DictWriter(csvfile_eeg, fieldnames=fieldnames_eeg)
    # writer_eeg.writeheader()

    csvfile_power = open('../data/' + args.filename_power, 'a')
    fieldnames_power = ['timestamp',
                        'alpha_absolute_ch1','alpha_absolute_ch2','alpha_absolute_ch3','alpha_absolute_ch4',
                        'beta_absolute_ch1','beta_absolute_ch2','beta_absolute_ch3','beta_absolute_ch4',
                        'delta_absolute_ch1','delta_absolute_ch2','delta_absolute_ch3','delta_absolute_ch4',
                        'theta_absolute_ch1','theta_absolute_ch2','theta_absolute_ch3','theta_absolute_ch4',
                        'gamma_absolute_ch1','gamma_absolute_ch2','gamma_absolute_ch3','gamma_absolute_ch4',
                        'alpha_relative_ch1', 'alpha_relative_ch2','alpha_relative_ch3','alpha_relative_ch4',
                        'beta_relative_ch1','beta_relative_ch2','beta_relative_ch3','beta_relative_ch4',
                        'delta_relative_ch1', 'delta_relative_ch2','delta_relative_ch3','delta_relative_ch4',
                        'theta_relative_ch1', 'theta_relative_ch2', 'theta_relative_ch3', 'theta_relative_ch4',
                        'gamma_relative_ch1', 'gamma_relative_ch2', 'gamma_relative_ch3', 'gamma_relative_ch4',
                        'alpha_session_score_ch1','alpha_session_score_ch2','alpha_session_score_ch3','alpha_session_score_ch4',
                        'beta_session_score_ch1','beta_session_score_ch2','beta_session_score_ch3','beta_session_score_ch4',
                        'delta_session_score_ch1', 'delta_session_score_ch2', 'delta_session_score_ch3', 'delta_session_score_ch4',
                        'theta_session_score_ch1','theta_session_score_ch2','theta_session_score_ch3','theta_session_score_ch4',
                        'gamma_session_score_ch1','gamma_session_score_ch2','gamma_session_score_ch3','gamma_session_score_ch4']
    writer_power = csv.DictWriter(csvfile_power, fieldnames=fieldnames_power)
    # writer_power.writeheader()

    print("connecting to robot car ...")
    bluetooth = serial.Serial('/dev/tty.HC-05-DevB', 9600, timeout=10)
    print("robot car connected!")
    sender = BluetoothSender(bluetooth, 5)

    power_receiver = PowerBandsReceiver(writer_power)  # writer_power
    power_receiver.start()

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/debug", print)
    # dispatcher.map("/muse/eeg", eeg_handler, "EEG", writer_eeg)
    # dispatcher.map("/muse/eeg/quantization", quantization_handler, "Quantization")

    # Get absolute data, five types of signals.
    dispatcher.map("/muse/elements/alpha_absolute", alpha_absolute_handler, "alpha_absolute")
    dispatcher.map("/muse/elements/beta_absolute", beta_absolute_handler, "beta_absolute")
    dispatcher.map("/muse/elements/delta_absolute", delta_absolute_handler, "delta_absolute")
    dispatcher.map("/muse/elements/theta_absolute", theta_absolute_handler, "theta_absolute")
    dispatcher.map("/muse/elements/gamma_absolute", gamma_absolute_handler, "gamma_absolute")

    # Get relative data, five types of signals.
    dispatcher.map("/muse/elements/alpha_relative", alpha_relative_handler, "alpha_relative")
    dispatcher.map("/muse/elements/beta_relative", beta_relative_handler, "beta_relative")
    dispatcher.map("/muse/elements/delta_relative", delta_relative_handler, "delta_relative")
    dispatcher.map("/muse/elements/theta_relative", theta_relative_handler, "theta_relative")
    dispatcher.map("/muse/elements/gamma_relative", gamma_relative_handler, "gamma_relative")

    # Get session data, five types of signals.
    dispatcher.map("/muse/elements/alpha_session_score", alpha_session_score_handler, "alpha_session_score")
    dispatcher.map("/muse/elements/beta_session_score", beta_session_score_handler, "beta_session_score")
    dispatcher.map("/muse/elements/delta_session_score", delta_session_score_handler, "delta_session_score")
    dispatcher.map("/muse/elements/theta_session_score", theta_session_score_handler, "theta_session_score")
    dispatcher.map("/muse/elements/gamma_session_score", gamma_session_score_handler, "gamma_session_score")

    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
