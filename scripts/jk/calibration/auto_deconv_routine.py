
import sys
sys.path.append(r'C:\pythonLab\pulsegen')
sys.path.append(r'C:\pythonLab\pulseseq\pulseseq')
import sequencer

from auto_deconv import Auto_Deconv
import numpy as np
import h5py
from measurement import Measurement
fg='aieee'
import time
lt = time.localtime()
date_str = str(lt.tm_year-2000)+str(lt.tm_mon)+str(lt.tm_mday)
hd5_path = 'C:\\pythonLab\\'
hd5_file = 'blah_'+date_str+'.hdf5'
try: hdf5f = h5py.File(hd5_path+hd5_file, 'a')
except: hdf5f = h5py.File(hd5_path+hd5_file, 'w')
import matplotlib.pyplot as plt

import gc

#try:
#    del awg1
#    print 'deleted awg1'
#except:
#    print 'failed to del awg1'
#    pass
#try:
#    del fs
#    print 'deleted fs'
#except:
#    print 'failed to del fs'
#    pass
#
#gc.collect()


create_ins = 1
if create_ins:
    import AWG
    awg1=AWG.AWG('TCPIP0::172.28.140.179::4101::SOCKET')

    import Agilent_DCAX86100D
    fs = Agilent_DCAX86100D.FastScope('TCPIP0::172.28.141.133::inst0::INSTR')

Measurement.initialize({
    awg1:{1:1, 2:2, 3:3, 4:4}
    }, fg, hdf5f)
time.sleep(0.5)


kernel_path = r'C:\qrlab\scripts\calibration\deconv_kernel.csv'

gen_new_kernel_and_test = 1
if gen_new_kernel_and_test:
    wait_time = 200 #20 is ok.

    a = Auto_Deconv(fs,awg1,flux_chan=1)

    a.generate(kernel_path=None)
    a.setup()
    time.sleep(wait_time)
    x_old,y_old = a.take_trace()

    a.xs = x_old
    a.ys = y_old

    a.deconv_analysis(plot_all=False,plot_results=False,save_kernel=kernel_path)
    a.generate(kernel_path=kernel_path)
    a.setup()
    time.sleep(wait_time)
    x_new,y_new = a.take_trace()

    ranges = [[0,9999],[4000,4965],[4996,5990]]
    titles = ['full range','before the step','after the step']
    for idx,val in enumerate(ranges):
        r_min,r_max = val
        plt.plot(x_old[r_min:r_max]*1e9,y_old[r_min:r_max])
        plt.plot(x_new[r_min:r_max]*1e9,y_new[r_min:r_max])

        avg_old = np.average(y_old[r_min:r_max])
        plt.plot([x_old[r_min]*1e9,x_old[r_max]*1e9],[avg_old,avg_old])

        avg_new = np.average(y_new[r_min:r_max])
        plt.plot([x_new[r_min]*1e9,x_new[r_max]*1e9],[avg_new,avg_new])

        plt.title(titles[idx])
        plt.show()

    r_min = 4000
    r_max = 4965
    data = y_new[r_min:r_max]
    avg = np.average(data)
    plt.plot(x_new[r_min:r_max]*1e9,(data-avg))
    plt.title('(deviation) Before the step')
    plt.show()

    r_min = 4996
    r_max = 5990
    data = y_new[r_min:r_max]
    avg = np.average(data)
    plt.plot(x_new[r_min:r_max]*1e9,(data-avg))
    plt.title('(deviation) After the step')
    plt.show()

testcomb = 0
if testcomb:
    wait_time = 60
    a = Auto_Deconv(fs,awg1,flux_chan=1)

    a.generate(pattern = 'testcomb', kernel_path=None)
    a.setup()
    time.sleep(wait_time)
    x_old,y_old = a.take_trace()

    a.generate(pattern = 'testcomb', kernel_path=kernel_path)
    a.setup()
    time.sleep(wait_time)
    x_new,y_new = a.take_trace()

    plt.plot(x_old*1e9,y_old)
    plt.plot(x_new*1e9,y_new)
    plt.title('testcomb')
    plt.show()

    plt.plot(x_old*1e9,y_old)
    plt.plot(x_new*1e9,y_new)
    plt.ylim(.242,.252)
    plt.title('testcomb')
    plt.show()

