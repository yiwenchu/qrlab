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
from pulseseq.sequencer import *
from pulseseq.pulselib import *


#IFPeriod = 10.0
awg_fileload = True
dot_awg_path = r'C:\awg_seqs'
dot_awg_path_awg = r'Y:\\'

class MeasurementOM(object):
    
    def __init__(self, infos, **kwargs):
            
        self.instruments = mclient.instruments
#        self.readout_info = mclient.get_readout_info(readout)
#        self._funcgen = mclient.instruments.get('funcgen')
        self.seq = None
        self._awgloader = None
        
        if infos is None:
            infos = []
        elif type(infos) is types.TupleType:
            infos = list(infos)
        elif type(infos) is not types.ListType:
            infos = [infos,]
        self.infos = infos

    def generate(self, s):

        s = self.get_sequencer(s)
        seq = s.render()
        self.seq = seq
        return seq
        
    def get_sequencer(self, seqs=None):
        s = Sequencer(seqs)
        
        for i in self.infos:
            if i.ssb:
                s.add_ssb(i.ssb)
            if i.marker and i.marker['channel'] != '':
                s.add_marker(i.marker['channel'], i.channels[0],
                             ofs=i.marker['ofs'], bufwidth=i.marker['bufwidth'])
                s.add_marker(i.marker['channel'], i.channels[1],
                             ofs=i.marker['ofs'], bufwidth=i.marker['bufwidth'])
        
        for ch in [1, 2, 3, 4,]:
            s.add_required_channel(ch)
            
        # Add master/slave settings to sequencer
        if hasattr(config, 'slave_triggers'):
            slave_chan = int(config.slave_triggers[0][0].split('m')[0])
            master_awg = ((slave_chan - 1) / 4) + 1
            logging.info('AWG %d seems to be the master' % master_awg)
            for i in range(4):
                ch = 4 * (master_awg - 1) + i + 1
                s.add_master_channel(ch)
                s.add_master_channel('%dm1'%ch)
                s.add_master_channel('%dm2'%ch)

            for chan, delay in config.slave_triggers:
                s.add_slave_trigger(chan, delay)

        # Add channel delays settings to sequencer
        if hasattr(config, 'channel_delays'):
            for ch, delay in config.channel_delays:
                s.add_channel_delay(ch, delay)

        if hasattr(config, 'flatten_waveforms'):
            s.set_flatten(config.flatten_waveforms)

        if hasattr(config, 'channel_convolutions'):
            s.set_flatten(True)
            for ch, path in config.channel_convolutions:
                kernel = np.loadtxt(path)
                s.add_convolution(ch, kernel)
                logging.info('adding convolution channel: %d' % ch)

        return s
        
    def get_awg_loader(self):
        '''
        Detect all AWGs and map channels:
        AWG1 gets channel 1-4, AWG2 5-8, etc.
        '''
        if self._awgloader:
            return self._awgloader

#        if hasattr(config,'awg_fileload') and hasattr(config,'dot_awg_path'):            
#            fl = config.awg_fileload
#            fp = config.dot_awg_path
#            fpawg = config.dot_awg_path_awg
#        else:
#            fl = False
#            fp = None
#            fpawg = None
#
#        print fl, fp, fpawg
        l = awgloader.AWGLoader(bulkload=config.awg_bulkload,
                                fileload=awg_fileload, dot_awg_path=dot_awg_path,
                                dot_awg_path_awg=dot_awg_path_awg)
        base = 1
        for i in range(1, 5):
            awg = self.instruments['AWG%d'%i]
            if awg:
                chanmap = {1:base, 2:base+1, 3:base+2, 4:base+3}
                logging.info('Adding AWG%d, channel map: %s', i, chanmap)
                l.add_awg(awg, chanmap)
                base += 4

        self._awgloader = l
        return l

    def load(self, run=False, ntries=4):
        '''
        Load sequences <seqs> to awgs.
        awgs are located from the instruments list and should be named
        AWG1, AWG2, ... (up to 4 currently).
        '''
        l = self.get_awg_loader()       
        l.load(self.seq)
        if run:
            self.start_awgs()
            
    def start_awgs(self):
        l = self.get_awg_loader()
        l.run()            
            
            
            
            
            
            
            
            
            
            
            
            