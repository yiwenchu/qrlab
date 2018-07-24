# NI_USB_6251.py
#
# Idea stolen from:
# http://www.scipy.org/Cookbook/Data_Acquisition_with_NIDAQmx#
#
# This used to be a near-verbatim translation of the example program
# C:\Program Files\National Instruments\NI-DAQ\Examples\DAQmx
#    ANSI C\Analog In\Measure Voltage\Acq-Int Clk\Acq-IntClk.c
# Adapted by Martijn Schaafsma April 2011

#### Import libraries ####
from instrument import Instrument
import types
import logging
import ctypes
import numpy
from pylab import plot, show, subplot
from time import sleep
import qt

class NI_USB_6251(Instrument):

    def __init__(self,name, num_samples=1000,samplerate = 1000.0,channellist=[0,8]):
        logging.info(__name__ + ' : Initializing instrument NI_USB_6251')
        Instrument.__init__(self, name, tags=['physical'])

        #### Load dll ####
        self._nidaq = ctypes.windll.nicaiu

        # the typedefs
        self._int32 = ctypes.c_long
        self._uInt32 = ctypes.c_ulong
        self._uInt64 = ctypes.c_ulonglong
        self._float64 = ctypes.c_double
        self._TaskHandle = self._uInt32

        # the constants
        self._DAQmx_Val_Cfg_Default = self._int32(-1)
        self._DAQmx_Val_Diff = 10106
        self._DAQmx_Val_Volts = 10348
        self._DAQmx_Val_Rising = 10280
        self._DAQmx_Val_FiniteSamps = 10178
        self._DAQmx_Val_GroupByChannel = 0

        # Add functions
        self.add_function('FetchMean')
        self.add_function('FetchAll')
        self.add_function('UpdateChannels')
        self.add_function('GetChannels')

        # Add parameters
        self.add_parameter('max_num_samples',
          flags=Instrument.FLAG_GETSET, units='', minval=0, maxval=16000, type=types.IntType)
        self.add_parameter('samplerate',
          flags=Instrument.FLAG_GETSET, units='Hz', minval=0, maxval=16000, type=types.FloatType)

        # Set properties
        self.set_max_num_samples(num_samples)
        self.set_samplerate(samplerate)
        self.UpdateChannels(channellist)

    def GetChannels(self):
      return self._channellist

    def do_set_max_num_samples(self,nums):
        self._max_num_samples = nums

    def do_get_max_num_samples(self):
        return self._max_num_samples

    def do_set_samplerate(self,rate):
        self._samplerate = rate

    def do_get_samplerate(self):
        return self._samplerate

    def UpdateChannels(self, channellist=''):
        if len(channellist)==0:
          if len(self._channellist)==0:
            self._channellist = [0,8]
        else:
          self._channellist = channellist

        self._numch = len(self._channellist)
        channelstr = ''
        for ch in self._channellist:
          channelstr = '%s, THz_setup/ai%d' %(channelstr,ch)
        self._channelstr = channelstr[2:]
        return self._channelstr

    def CHK(self, err):
        """a simple error checking routine"""
        if err < 0:
            buf_size = 100
            buf = ctypes.create_string_buffer('\000' * buf_size)
            self._nidaq.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
            raise RuntimeError('nidaq call failed with error %d: %s'%(err,repr(buf.value)))

    def TestMeasure(self):
        numpts = 600
        waittime = 0.5

        self.set_max_num_samples(10)
        self.set_samplerate(1000.0)

        vals = []
        for i in range(numpts):
          vals.append(self.FetchMean())
          qt.msleep(waittime)
        plot(range(numpts),vals)

    def FetchMean(self):
        t = self.FetchAll()
        r = []
        for a in t:
          r.append(numpy.mean(t[a]))

        return r

    def FetchAll(self):

        taskHandle = self._TaskHandle(0)
        data = numpy.zeros((self._numch*self._max_num_samples),dtype=numpy.float64)

        #### Acquire data ####
        self.CHK(self._nidaq.DAQmxCreateTask("",ctypes.byref(taskHandle)))
        self.CHK(self._nidaq.DAQmxCreateAIVoltageChan(taskHandle,self._channelstr,"",
                                           self._DAQmx_Val_Cfg_Default, # _DAQmx_Val_Cfg_Default _DAQmx_Val_Diff
                                           self._float64(-10.0),self._float64(10.0),
                                           self._DAQmx_Val_Volts,None))
        self.CHK(self._nidaq.DAQmxCfgSampClkTiming(taskHandle,"",self._float64(self._samplerate),
                                        self._DAQmx_Val_Rising,self._DAQmx_Val_FiniteSamps,
                                        self._uInt64(self._max_num_samples)));
        self.CHK(self._nidaq.DAQmxStartTask(taskHandle))
        read = self._int32()
        self.CHK(self._nidaq.DAQmxReadAnalogF64(taskHandle,self._max_num_samples,self._float64(10.0),
                                     self._DAQmx_Val_GroupByChannel,data.ctypes.data,
                                     self._numch*self._max_num_samples,ctypes.byref(read),None))
#        print "Acquired %d points"%(read.value)

        if taskHandle.value != 0:
            self._nidaq.DAQmxStopTask(taskHandle)
            self._nidaq.DAQmxClearTask(taskHandle)

        vals = {}
        for i in range(self._numch):
          t = data[i*self._max_num_samples:(i+1)*self._max_num_samples]
          vals[i]=t
        return vals