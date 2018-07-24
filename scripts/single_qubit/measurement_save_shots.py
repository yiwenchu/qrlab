import mclient
import config
import types
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import logging
logging.getLogger().setLevel(logging.INFO)
import signal
import objectsharer as objsh
from lib import jsonext
import os
import pulseseq
import awgloader
from PyQt4 import QtGui
from pulseseq import sequencer
from pulseseq import pulselib


def Measurement_shots(measurement, make_fig = True):
    
    alz = measurement.instruments['alazar']
    measurement.setup_measurement()
    
#    measurement.generate()
#    measurement.setup_measurement()
#    measurement.start_awgs()
#    measurement.start_funcgen()  

    if measurement._funcgen:
        measurement.start_awgs()
        time.sleep(0.1)

    # Setup and arm alazar

    alz.setup_experiment(measurement.cyclelen)

    alz.set_interrupt(False)
    # Estimate a capture timeout, mostly because the AWG is a bit slow...
    timeout = min(50000 + 2000 * measurement.cyclelen, 600000)
    alz.set_timeout(timeout)
    time.sleep(0.1)

#    # Capture CTRL-C and connect callbacks
#    measurement.capture_ctrlc()
#    progress_hid = alz.connect('capture-progress', self._capture_progress_cb)
#    dataupd_hid = self.data.connect('changed', self._data_changed_cb)

    # Start measurement, either by starting the AWG or the function generator
    if measurement._funcgen:
        self.start_funcgen()
    else:
        self.start_awgs()          
    
    shotsReal = alz.take_experiment_shots(acqtimeout=1000000)
    
    measurement.data.create_dataset('shots', 
                                    data = shotsIQr, 
                                    dtype=np.complex)
    
    if make_fig == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(shotsReal)        
        
        ax.set_title = measurement.title + ' data in %s' % measurement.data.get_fullname()
        ax.set_ylabel('Amplitide')
        ax.set_xlabel('Intensity')                                          
    

