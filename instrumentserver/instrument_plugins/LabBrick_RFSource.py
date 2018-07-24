# LabBrick RF Source DLL driver
#
# There are different DLL versions around. Most of them use MS Visual C name
# mangling (a bit inconvenient), but newer ANSI versions are around as well.
# It seems the newer builds need the MS Visual C++ 2013 redistributable
# package, which you can get at http://www.microsoft.com/en-us/download/details.aspx?id=40784


import sys
import time
import ctypes
import types
import numpy as np
from instrument import Instrument
import logging

SUCCESS = 0
NO_DEVICE = 0

LB_DLL = 'vnx_fmsynth.dll'
try:
    lb_dll = ctypes.cdll.LoadLibrary(LB_DLL)
except Exception, e:
    raise ValueError('Unable to load LabBrick DLL, please put vnx_fmsynth.dll in instrumentserver directory (%s)'%str(e))

# Many versions of the labbrick DLL are compiled using MS Visual C decorators...
DECORATOR_MAP = {
    'fnLMS_SetTestMode': '?fnLMS_SetTestMode@@YAX_N@Z',
    'fnLMS_GetNumDevices': '?fnLMS_GetNumDevices@@YAHXZ',
    'fnLMS_GetDevInfo': '?fnLMS_GetDevInfo@@YAHPAI@Z',
    'fnLMS_GetSerialNumber': '?fnLMS_GetSerialNumber@@YAHI@Z',
    'fnLMS_GetModelNameA': '?fnLMS_GetModelName@@YAHIPAD@Z',
    'fnLMS_InitDevice': '?fnLMS_InitDevice@@YAHI@Z',
    'fnLMS_CloseDevice': '?fnLMS_CloseDevice@@YAHI@Z',
    'fnLMS_GetMaxFreq': '?fnLMS_GetMaxFreq@@YAHI@Z',
    'fnLMS_GetMinFreq': '?fnLMS_GetMinFreq@@YAHI@Z',
    'fnLMS_GetMaxPwr': '?fnLMS_GetMaxPwr@@YAHI@Z',
    'fnLMS_GetMinPwr': '?fnLMS_GetMinPwr@@YAHI@Z',
    'fnLMS_GetRF_On': '?fnLMS_GetRF_On@@YAHI@Z',
    'fnLMS_SetRFOn': '?fnLMS_SetRFOn@@YAHI_N@Z',
    'fnLMS_GetUseInternalRef': '?fnLMS_GetUseInternalRef@@YAHI@Z',
    'fnLMS_SetUseInternalRef': '?fnLMS_SetUseInternalRef@@YAHI_N@Z',
    'fnLMS_GetUseInternalPulseMod': '?fnLMS_GetUseInternalPulseMod@@YAHI@Z',
    'fnLMS_SetUseExternalPulseMod': '?fnLMS_SetUseExternalPulseMod@@YAHI_N@Z',
    'fnLMS_GetFrequency': '?fnLMS_GetFrequency@@YAHI@Z',
    'fnLMS_SetFrequency': '?fnLMS_SetFrequency@@YAHIH@Z',
    'fnLMS_GetPowerLevel': '?fnLMS_GetPowerLevel@@YAHI@Z',
    'fnLMS_SetPowerLevel': '?fnLMS_SetPowerLevel@@YAHIH@Z',
    'fnLMS_GetDeviceStatus': '?fnLMS_GetDeviceStatus@@YAHI@Z',
}

STATUS_INV_DEVID        = 0x80000000
STATUS_DEV_CONNECTED    = 0x00000001
STATUS_DEV_OPENED       = 0x00000002
STATUS_SWP_ACTIVE       = 0x00000004
STATUS_SWP_UP           = 0x00000008
STATUS_SWP_REPEAT       = 0x00000010
STATUS_SWP_BIDIR        = 0x00000020
STATUS_PLL_LOCKED       = 0x00000040
STATUS_FAST_PULSE_OPT   = 0x00000080

def get_lb_func(funcname, argtypes=[ctypes.c_uint32]):
    '''
    Get function from labbrick DLL.
    First try normal name (ANSI DLL version), otherwise decorated name (DLL
    with MS Visual C name mangling).
    Set function argument types to <argtypes>.
    '''

    try:
        f = getattr(lb_dll, funcname)
    except:
        if funcname not in DECORATOR_MAP:
            raise ValueError('Decorated version of function %s not known' % funcname)
        f = getattr(lb_dll, DECORATOR_MAP[funcname])

    f.argtypes = argtypes
    return f

def do_set_test_mode(val):
    val = int(val)
    f = get_lb_func('fnLMS_SetTestMode', [ctypes.c_uint32])
    return f(val)

def do_get_num_devices():
    f = get_lb_func('fnLMS_GetNumDevices', [])
    return f()

def do_get_device_info():
    FP = ctypes.POINTER(ctypes.c_uint32)
    f = get_lb_func('fnLMS_GetDevInfo', [FP])
    num_devices = do_get_num_devices()
    device_info = np.zeros([num_devices], dtype=np.uint32)
    f(ctypes.cast(device_info.ctypes.data, FP))
    return device_info

def do_get_serial_number(devid):
    f = get_lb_func('fnLMS_GetSerialNumber')
    return f(devid)

def do_get_model_name(devid):
    f = get_lb_func('fnLMS_GetModelNameA', [ctypes.c_uint32, ctypes.c_char_p])
    model_name = ' '*32
    name_len = f(devid, model_name)
    return model_name[:name_len]

def find_labbricks():
    do_set_test_mode(False)
    d_IDs = do_get_device_info()

    ret = {}
    for idx in d_IDs:
        sn = do_get_serial_number(idx)
        mn = do_get_model_name(idx)
        ret[idx] = (sn, mn)
    return ret

class LabBrick_RFSource(Instrument):

    def __init__(self, name, devid=None, serial=None, **kwargs):
        super(LabBrick_RFSource, self).__init__(name)

        if devid is None and serial is None:
            raise Exception('Labbrick driver needs devid or serial as parameter')

        if serial is not None:
            serial = int(serial)
            for did, props in find_labbricks().iteritems():
                if props[0] == serial:
                    devid = did
                    break

        if devid is None:
            raise Exception('Unable to find Labbrick with serial %s'%serial)

        self._devid = devid
        self._serialno = do_get_serial_number(devid)
        self._modelname = do_get_model_name(devid)

        if self._serialno is NO_DEVICE:
            raise Exception('invalid device ID, try again.')

        val = self._init()
        if val is not SUCCESS:
            print 'labbrick (sn: %d) device already opened, reopening' % \
                        (self._serialno)
            self._close()
            time.sleep(0.01)
            self._init()

        self._max_freq = self._get_max_freq()
        self._min_freq = self._get_min_freq()
        self._max_power = self._get_max_power()
        self._min_power  = self._get_min_power()
        logging.debug('Frequency range: %.03f - %.03f MHz' % (self._min_freq/1e6, self._max_freq/1e6))
        logging.debug('Power range: %.01f - %.01f dBm' % (self._min_power, self._max_power))

        self.add_parameter('serial', type=types.IntType,
            flags=Instrument.FLAG_GET, value=self._serialno)
        self.add_parameter('model', type=types.StringType,
            flags=Instrument.FLAG_GET, value=self._modelname)

        self.add_parameter('rf_on', type=types.BooleanType,
            flags=Instrument.FLAG_GETSET)
        self.add_parameter('ext_locked', type=types.BooleanType,
            flags=Instrument.FLAG_GET)
        self.add_parameter('fast_pulse_option', type=types.BooleanType,
            flags=Instrument.FLAG_GET)
        self.add_parameter('power', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='dBm',
            minval=self._min_power, maxval=self._max_power,
            format='%.02f')
        self.add_parameter('frequency', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='Hz',
            minval=self._min_freq, maxval=self._max_freq,
            display_scale=6)
        self.add_parameter('use_extref', type=types.BooleanType,
            flags=Instrument.FLAG_GETSET)
        self.add_parameter('pulse_on', type=types.BooleanType,
            flags=Instrument.FLAG_GETSET)

        if kwargs.pop('reset', False):
            self.reset()
        else:
            self.get_all()
        self.set(kwargs)

    def do_get_serial(self):
        return self._serialno

    def do_get_model(self):
        return self._modelname

    def _init(self):
        f = get_lb_func('fnLMS_InitDevice')
        return f(self._devid)

    def _close(self):
        f = get_lb_func('fnLMS_CloseDevice')
        return f(self._devid)

    def _get_max_freq(self):
        f = get_lb_func('fnLMS_GetMaxFreq')
        return f(self._devid) * 10.0

    def _get_min_freq(self):
        f = get_lb_func('fnLMS_GetMinFreq')
        return f(self._devid) * 10.0

    def _get_status(self):
        f = get_lb_func('fnLMS_GetDeviceStatus')
        return f(self._devid)

    def _get_max_power(self):
        f = get_lb_func('fnLMS_GetMaxPwr')
        return f(self._devid) * 0.25

    def _get_min_power(self):
        f = get_lb_func('fnLMS_GetMinPwr')
        return f(self._devid) * 0.25

    def convert_to_units(self, power):
        '''Convert power in dBm to dll integer units'''
        return int(round(power * 4))

    def convert_to_dBm(self, val):
        '''Cnverts dll units to dBm'''
        return self._max_power - 0.25 * val

    def do_get_rf_on(self):
        f = get_lb_func('fnLMS_GetRF_On')
        return f(self._devid) == 1

    def do_set_rf_on(self, val):
        f = get_lb_func('fnLMS_SetRFOn', [ctypes.c_uint32, ctypes.c_bool])
        return f(self._devid, val)

    def do_get_ext_locked(self):
        return bool(self._get_status() & STATUS_PLL_LOCKED)

    def do_get_fast_pulse_option(self):
        return bool(self._get_status() & STATUS_FAST_PULSE_OPT)

    def do_get_use_extref(self):
        f = get_lb_func('fnLMS_GetUseInternalRef')
        return f(self._devid) == 0

    def do_set_use_extref(self, val):
        f = get_lb_func('fnLMS_SetUseInternalRef', [ctypes.c_uint32, ctypes.c_bool])
        return f(self._devid, not val)

    def do_get_pulse_on(self):
        f = get_lb_func('fnLMS_GetUseInternalPulseMod')
        return not f(self._devid)

    def do_set_pulse_on(self, val):
        f = get_lb_func('fnLMS_SetUseExternalPulseMod', [ctypes.c_uint32, ctypes.c_bool])
        return f(self._devid, val)

    def do_get_frequency(self):
        f = get_lb_func('fnLMS_GetFrequency')
        return f(self._devid) * 10.0

    def do_set_frequency(self, freq_Hz):
        val = int(round(freq_Hz / 10.0))
        f = get_lb_func('fnLMS_SetFrequency', [ctypes.c_uint32, ctypes.c_int32])
        return f(self._devid, val)

    def do_get_power(self):
        f = get_lb_func('fnLMS_GetPowerLevel')
        return self.convert_to_dBm(f(self._devid))

    def do_set_power(self, power):
        val = self.convert_to_units(power)
        f = get_lb_func('fnLMS_SetPowerLevel', [ctypes.c_uint32, ctypes.c_int32])
        return f(self._devid, val)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    lb = Instrument.test(LabBrick_RFSourceDLL)
