# coding: utf-8 

#import sys
import os
import time
import ctypes
import types
import numpy as np
import matplotlib.pyplot as plt
from instrument import Instrument
#import logging

sa_dll = os.path.dirname(__file__)
sa_dll += r'\sa_api.dll'
sa_dll = ctypes.cdll.LoadLibrary(sa_dll) 

#try:
#    sa_dll = 'instrument_plugins\\sa_api.dll'
#    sa_dll = ctypes.cdll.LoadLibrary(sa_dll)
#except:
#    sa_dll = 'sa_api.dll'
#    sa_dll = ctypes.cdll.LoadLibrary(sa_dll)

SA_TRUE  = 1
SA_FALSE = 0

#Modes
SA_IDLE      = -1
SA_SWEEPING  = 0x0
SA_REAL_TIME = 0x1
SA_IQ        = 0x2
SA_AUDIO     = 0x3
SA_TG_SWEEP  = 0x4

#Limits
SA124_MIN_FREQ       = 100.0e3
SA124_MAX_FREQ       = 13.0e9
SA_MIN_SPAN          = 1.0
SA_MAX_REF           = 20 # dBm
SA_MAX_ATTEN         = 3
SA_MAX_GAIN          = 2
SA_MIN_RBW           = 0.1
SA_MAX_RBW           = 6.0e6
SA_MIN_RT_RBW        = 100.0
SA_MAX_RT_RBW        = 10000.0
SA_MIN_IQ_BANDWIDTH  = 100.0
SA_MAX_IQ_DECIMATION = 128
SA_IQ_SAMPLE_RATE    = 486111.111

# Scales
SA_LOG_SCALE      = 0x0
SA_LIN_SCALE      = 0x1
SA_LOG_FULL_SCALE = 0x2
SA_LIN_FULL_SCALE = 0x3

SA_REF_UNUSED       = 0
SA_REF_INTERNAL_OUT = 1
SA_REF_EXTERNAL_IN  = 2 

# Levels
SA_AUTO_ATTEN = -1
SA_AUTO_GAIN  = -1

# Detectors
SA_MIN_MAX = 0x0
SA_AVERAGE = 0x1

NO_ERROR = 'No error'

def saGetAPIVersion():
    '''
    This code was written with version 3.0.11
    '''
    f = getattr(sa_dll, 'saGetAPIVersion')
    f.restype = ctypes.c_char_p
    return f()

 
def saGetErrorString(code):
    f = getattr(sa_dll, 'saGetErrorString')
    f.restype = ctypes.c_char_p
    
    err_str = f(code)
    if err_str != NO_ERROR:
        print 'SA Error: %s' % err_str
        
    return err_str
    
    
def error_check(err_code):
    return saGetErrorString(err_code)


def saGetSerialNumberList():
    '''
    List of the serial numbers of currently connected devices, max 8.
    '''
    dev_count = ctypes.c_int32()
    s_nums = np.zeros(8, dtype=np.int32)
    
    f = getattr(sa_dll, 'saGetSerialNumberList')
    
    err = f(s_nums.ctypes, 
            ctypes.byref(dev_count))
    error_check(err)
    
    return s_nums
    

def saOpenDeviceBySerialNumber(s_num):
    '''
    Opens device by serial number, returns a device ID (or handle) for 
    communication afterwards.  May take several seconds.
    '''
    dev_id = ctypes.c_int32(0)

    f = getattr(sa_dll, 'saOpenDeviceBySerialNumber')
    
    err = f(ctypes.byref(dev_id), 
            ctypes.c_int32(s_num))
    err_code = error_check(err)
        
#    print 'Opened device.  SN: %d  Dev_ID: %d' % (s_num, dev_id.value)
    
    return err_code, dev_id.value
   
       
def sa_call(dev_id, func_call, *args):
    '''
    Standard format for calls to an already open device.
    '''
    f = getattr(sa_dll, func_call)

    err = f(ctypes.c_int32(dev_id), *args)
    return error_check(err)
    

def saCloseDevice(dev_id):
    sa_call(dev_id, 'saCloseDevice')
     
    print 'Closed Device.  Dev_ID: %d' % (dev_id)
    

def saInitiate(dev_id, mode=SA_SWEEPING):
    '''
    Sets mode of operation for the device.  We will normally use SA_SWEEPING
    which is a sweep on-demand.  Other options are externally triggered
    sweeps and streaming data.
    
    Must be called after changing configuration settings.
    '''
    return sa_call(dev_id, 'saInitiate', 
                   ctypes.c_int32(mode), 
                   ctypes.c_uint32(0))
    
    print 'Initiated.  Dev_ID: %d   Mode: %d' % (dev_id, mode)


def saConfigAcquisition(dev_id, detector=SA_AVERAGE, scale=SA_LOG_SCALE):
    '''
    Configures aquisition mode.  We'll almost always use SA_AVERAGE
    and SA_LOG_SCALE, but you can also obtain min/max data or data in a linear
    scale.  In SA_LOG_SCALE data is given in nominally calibrated dBm, in
    SA_LIN_SCALE data is in millivolts.  In the _FULL scales the data are
    directly from the ADC, which is not terribly useful.
    '''
    assert detector == SA_AVERAGE, 'Min/max not implemented.'
    
    return sa_call(dev_id, 'saConfigAcquisition', 
                   ctypes.c_int32(detector),
                   ctypes.c_int32(scale))


def saQuerySweepInfo(dev_id, get_xaxis=False):
    '''
    Returns the current sweep start, number of steps, and step size
    '''    
    sweep_length = ctypes.c_int32(0)
    start_freq = ctypes.c_double(0)
    bin_size = ctypes.c_double(0)
    
    err_code = sa_call(dev_id, 'saQuerySweepInfo', 
                       ctypes.byref(sweep_length),
                       ctypes.byref(start_freq),
                       ctypes.byref(bin_size))
            
    sweep_length = sweep_length.value
    start_freq = start_freq.value
    bin_size = bin_size.value
    
    if get_xaxis:
        return err_code, np.arange(sweep_length)*bin_size + start_freq
        
    return err_code, sweep_length, start_freq, bin_size       


def saGetSweep(dev_id, get_xs=True, sweeplen=None):
    '''
    This assumes we're in average mode.  If you really want min/max the 
    buffers have different data and the return needs to be modified.
    '''
    if get_xs:
        err_code, xaxis = saQuerySweepInfo(dev_id, get_xaxis=True)
        sweeplen = len(xaxis)
    else:
        if sweeplen is None:
            raise Exception('must specify length or get xs')
    
    min_buff = np.empty(sweeplen, dtype=np.double)
    max_buff = np.empty(sweeplen, dtype=np.double)
#    print '00', sweeplen

    err_code = sa_call(dev_id, 'saGetSweep_64f', 
                       min_buff.ctypes,
                       max_buff.ctypes)
#    print '11'
    if get_xs:
        return err_code, xaxis, min_buff
    else:
        return err_code, min_buff    
    
def saQueryTemperature(dev_id):
    '''
    This gives you the internal temperature (C) of the device.  For reasons.
    Device has to be in idle mode (aborted) to get a new value.
    '''
    temp = ctypes.c_float(0)

    err_code = sa_call(dev_id, 'saQueryTemperature', 
                       ctypes.byref(temp))

    return err_code, temp.value
    

def saAbort(dev_id):
    '''
    Puts the device in idle mode.  Needs to be initiated to use again.
    '''
    return sa_call(dev_id, 'saAbort')


def saSetTimebase(dev_id, timebase=SA_REF_EXTERNAL_IN):
    '''
    Calling with SA_REF_EXTERNAL_IN attempts to use the external 10 MHz
    reference.  None is found the error will reflect that, and the internal
    reference will be used.  SA_REF_EXTERNAL_OUT is if you want the SA124 to 
    generate a 10 MHz reference, which seems unlikely.
    '''
    return sa_call(dev_id, 'saSetTimebase',
                   ctypes.c_int32(timebase))    
                   

def saConfigSweepCoupling(dev_id, rbw, vbw, reject=SA_TRUE):
    '''
    The resolution bandwidth, or RBW, represents the bandwidth of spectral 
    energy represented in each frequency bin. For example, with an RBW of 10 
    kHz, the amplitude value for each bin would represent the total energy 
    from 5 kHz below to 5 kHz above the bins center.
    
    The video bandwidth, or VBW, is applied after the signal has been converted
    to frequency domain as power, voltage, or log units. It is implemented as
    a simple rectangular window, averaging the amplitude readings for each 
    frequency bin over several overlapping FFTs. A signal whose amplitude is 
    modulated at a much higher frequency than the VBW will be shown as an 
    average, whereas amplitude modulation at a lower frequency will be shown 
    as a minimum and maximum value.
    
    Available RBWs are [0.1Hz – 100kHz] and 250kHz. '''
    assert rbw > SA_MIN_RBW, 'RBW below minimum: %f < %f' % (rbw, SA_MIN_RBW)
    assert rbw < SA_MAX_RBW, 'RBW above maximum: %f > %f' % (rbw, SA_MAX_RBW)

    
    return sa_call(dev_id, 'saConfigSweepCoupling',
                   ctypes.c_double(rbw),
                   ctypes.c_double(vbw),
                   ctypes.c_bool(reject)) 
                   

def saConfigCenterSpan(dev_id, center, span):
    '''
    Set the sweep frequencies.  The precise settings will not be what are
    assigned here, but they can be found from saQuerySweepInfo.
    
    '''
    f_min = center-span
    f_max = center+span
    
    assert f_min > SA124_MIN_FREQ, 'f_min below minimum: %f < %f' % (f_min, SA124_MIN_FREQ)
    assert f_max < SA124_MAX_FREQ, 'f_max above maximum: %f > %f' % (f_max, SA124_MAX_FREQ)
    assert span > 1.0, 'Minimum span is 1 Hz'
    
    return sa_call(dev_id, 'saConfigCenterSpan',
                   ctypes.c_double(center),
                   ctypes.c_double(span)) 
                       
    
def saConfigLevel(dev_id, level):
    '''
    Sets reference level in dBm.  
    '''
    assert level < SA_MAX_REF, 'level above maximum: %f > %f' % (level, SA_MAX_REF)

    return sa_call(dev_id, 'saConfigLevel',
                   ctypes.c_double(level)) 

def saReadSingleFreq(dev_id, freq, n_av=1, verify_freq=True,
                     set_zeroif_params=True):
    '''
    This function is for mixer tuning.  It's effectively zero-IF mode.
    The SA124B has a zero IF mode you can access from the Spike software,
    but it's not documented in the API.  (It is documented for the BB
    series analyzers.)
    
    This function becomes notably faster without the verification step,
    as well as without setting the zeroif_params.  If you're going to be
    looking at a frequency more than once (as you do when tuning a mixer),
    there's no need to reset the zeroif_params or re-verify the frequency.
    I recommend only doing those on the first call.
    
    There's not too much of a need for averaging because the device is
    quite sensitive.
    '''
    if set_zeroif_params:
        saConfigSweepCoupling(dev_id, 250e3, 250e3)
    err_code = saConfigCenterSpan(dev_id, freq, 250e3)
    saInitiate(dev_id, SA_SWEEPING)
    time.sleep(0.25)
    
    if verify_freq:
        err_code, sweep_length, start_freq, bin_size = saQuerySweepInfo(dev_id)
        actual_freq = start_freq + round(sweep_length/2)*bin_size
        diff = np.abs(freq-actual_freq)
        if diff > 100e3:
            raise Exception('Frequency set inaccurate')
            
    ys = []
    err_code, xs, y = saGetSweep(dev_id, get_xs=True)
#        if len(y) > 1:
#            print 'SA: settings lead to multiple measurements'
    
    idx = len(y)//2
    ys.append(y[idx])
    if n_av > 1:
        for i in range(n_av-1):                
#            ys.append(self.sweep(get_xs=False, sweeplen=len(y))[idx])
            err_code, y = saGetSweep(dev_id, get_xs=False, sweeplen=len(y))
            ys.append(y[idx])            
    else:
        pass
    ys = 10**np.array(np.array(ys)/10.0)
    y = np.mean(ys)
    return 10 * np.log10(y)                           


class SA124B(Instrument):

    def __init__(self, name, serial=None, **kwargs):
        super(SA124B, self).__init__(name)
        self.error_list = []
        self.vbw, self.rbw= 250e3, 250e3

        s_nums = saGetSerialNumberList()
        if not (serial in s_nums):
            raise Exception('Unable to find SA124B with serial %s' % serial)
        self._serialno = serial

        err, self._devid = saOpenDeviceBySerialNumber(serial)
        if err != NO_ERROR:
            raise Exception(err)

        print saSetTimebase(self._devid)
        print saConfigAcquisition(self._devid)     
        print saConfigCenterSpan(self._devid, 6e9, 10e6)
        self._initiate()
        
        self.add_parameter('serial', type=types.IntType,
            flags=Instrument.FLAG_GET, value=self._serialno)
        self.add_parameter('dev_id', type=types.IntType,
            flags=Instrument.FLAG_GET, value=self._devid)
        self.add_parameter('temperature', type=types.FloatType,
            flags=Instrument.FLAG_GET, units='C')            
        self.add_parameter('center_frequency', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='Hz',
            minval=SA124_MIN_FREQ, maxval=SA124_MAX_FREQ,
            display_scale=6)

        self.add_parameter('span', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='Hz',
            minval=1.0, maxval=12.4e9,
            display_scale=6)

        self.add_parameter('ext_ref', type=types.StringType,
            flags=Instrument.FLAG_GET)
            
        self.add_parameter('error', type=types.StringType,
            flags=Instrument.FLAG_GET, value='No error')
            
        self.add_parameter('vbw', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='Hz',
            display_scale=6)#, value=self.vbw)
            
        self.add_parameter('rbw', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='Hz',
            minval=SA_MIN_RBW, maxval=SA_MAX_RBW,
            display_scale=6)#, value=self.rbw)
            
        self.get_all()

        
    def _close(self):
        saCloseDevice(self._devid)
        
        
    def _initiate(self):
        saInitiate(self._devid, SA_SWEEPING)
        
        
    def do_get_error(self):
        '''
        Error list is update every call.  There is no device query.
        '''
        while len(self.error_list) > 1 and self.error_list[-1] == NO_ERROR:
            self.error_list.pop()
                
        if len(self.error_list) > 1:
            return self.error_list.pop()
        else:
            return 'Error list empty'
         
         
    def do_get_dev_id(self):
        return self._devid
    
    
    def do_get_serial(self):
        return self._serialno


    def do_get_vbw(self):
        # The API doesn't have a device query.
        return self.vbw
    
    
    def do_get_rbw(self):
        # The API doesn't have a device query.
        return self.rbw
     
       
    def do_get_temperature(self):
        saAbort(self._devid)
        err_code, temp = saQueryTemperature(self._devid)
        self.error_list.append(err_code)
        self._initiate()
        return temp
      
      
    def do_get_center_frequency(self):
        err_code, sweep_length, start_freq, bin_size = saQuerySweepInfo(self._devid)
        self.error_list.append(err_code)
        return start_freq + round(sweep_length/2)*bin_size
        
        
    def do_get_span(self):
        err_code, sweep_length, start_freq, bin_size = saQuerySweepInfo(self._devid)
        self.error_list.append(err_code)     
        return sweep_length*bin_size


    def do_set_center_frequency(self, freq):
        span = self.do_get_span()
        
        err_code = saConfigCenterSpan(self._devid, freq, span)
        self._initiate()
        self.error_list.append(err_code)
        return freq
        
        
    def do_set_span(self, span):
        center_frequency = self.do_get_center_frequency()
        
        err_code = saConfigCenterSpan(self._devid, center_frequency, span)
        self._initiate()
        self.error_list.append(err_code)
        return span

        
    def do_get_ext_ref(self):
        err_code = saSetTimebase(self._devid)
        self._initiate()
        return err_code


    def do_set_vbw(self, vbw):
        err_code = saConfigSweepCoupling(self._devid, self.rbw, vbw)
        self._initiate()
        if err_code == NO_ERROR:
            self.vbw = vbw
        self.error_list.append(err_code)
        return vbw

    
    def do_set_rbw(self, rbw):
        err_code = saConfigSweepCoupling(self._devid, rbw, self.vbw)
        self._initiate()
        if err_code == NO_ERROR:
            self.rbw = rbw
        self.error_list.append(err_code)
        return rbw
 

    def sweep(self, get_xs=True, sweeplen=None):
        if get_xs:
            err_code, xs, ys = saGetSweep(self._devid, get_xs=True)
            self.error_list.append(err_code)
            return xs, ys
        else:
            assert not (sweeplen is None)
            err_code, ys = saGetSweep(self._devid, get_xs=False, sweeplen=sweeplen)
            self.error_list.append(err_code)
            return ys                
            
    def read_single_freq(self, freq, n_av=1, verify_freq=True,
                         set_zeroif_params=True):
        '''
        This function is for mixer tuning.  It's effectively zero-IF mode.
        The SA124B has a zero IF mode you can access from the Spike software,
        but it's not documented in the API.  (It is documented for the BB
        series analyzers.)
        
        This function becomes notably faster without the verification step,
        as well as without setting the zeroif_params.  If you're going to be
        looking at a frequency more than once (as you do when tuning a mixer),
        there's no need to reset the zeroif_params or re-verify the frequency.
        I recommend only doing those on the first call.
        
        There's not too much of a need for averaging because the device is
        quite sensitive.
        '''
        if set_zeroif_params:
            self.do_set_rbw(250e3)
            self.do_set_span(250e3)
            self.do_set_vbw(250e3)
        
        self.do_set_center_frequency(freq)
        if verify_freq:
            actual_freq = self.do_get_center_frequency()
            diff = np.abs(freq-actual_freq)
            if diff > 100e3:
                raise Exception('Frequency set inaccurate')
                
        ys = []
        xs, y = self.sweep(get_xs=True)
        
#        if len(y) > 1:
#            print 'SA: settings lead to multiple measurements'
        
        idx = len(y)//2
        ys.append(y[idx])
        if n_av > 1:
            for i in range(n_av-1):                
                ys.append(self.sweep(get_xs=False, sweeplen=len(y))[idx])
                
        else:
            pass
        ys = 10**np.array(np.array(ys)/10.0)
        y = np.mean(ys)
        return 10 * np.log10(y)
                
        
  
  
if __name__ == '__main__':    
#    saCloseDevice(0)
#    bl

    assert saGetAPIVersion() == '3.0.11'
    
    s_nums = saGetSerialNumberList()
    s_num = s_nums[0]
    assert s_num != 0
     
    err, did = saOpenDeviceBySerialNumber(s_num)
    saSetTimebase(did)
    
    saConfigAcquisition(did)
#    saConfigSweepCoupling(did, 100, 100) # num pts set by RBW, VBW and span
#    saConfigCenterSpan(did, 5e9, 100)
#    saInitiate(did, SA_SWEEPING)
    time.sleep(1)
    #saQuerySweepInfo(did)
    try:
        print saReadSingleFreq(did, 6e9)
    except:
        pass
    finally:
        saCloseDevice(did)

#    start = time.time()
#    for i in range(10):
#        err_code, xs, ys = saGetSweep(did)
#    print (time.time()-start)/10.0
#    print ys
#    fig,ax=plt.subplots()
#    ax.plot(xs,ys)
#    
#    saAbort(did)
#    err_code, temp = saQueryTemperature(did)
#    print 'Device temperature: %.2f C' % temp
    
#    saCloseDevice(did)


