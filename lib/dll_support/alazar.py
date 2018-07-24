import ctypes
import numpy as np
import types
import time
import logging

import win32api
import msvcrt

# Load DLL, 64-bit seems to require windll, 32-bit cdll.
_DLL_FILE = r'c:\windows\system32\ATSApi.dll'
try:
    ats = ctypes.windll.LoadLibrary(_DLL_FILE)
    ats.AlazarGetStatus(0)
except:
    ats = ctypes.cdll.LoadLibrary(_DLL_FILE)

class Constants:
    CHANNEL_ALL     = 0
    CHANNEL_A       = 1
    CHANNEL_B       = 2
    CHANNEL_C       = 4
    CHANNEL_D       = 8
    CHANNEL_E       = 16
    CHANNEL_F       = 32
    CHANNEL_G       = 64
    CHANNEL_H       = 128
    CHANNEL_AB      = 3

    COUPLING_AC     = 1
    COUPLING_DC     = 2

    IMPEDANCE_1k    = 1
    IMPEDANCE_50    = 2
    IMPEDANCE_75    = 4
    IMPEDANCE_300   = 8

    RANGE_0020      = 1
    RANGE_0040      = 2
    RANGE_0050      = 3
    RANGE_0080      = 4
    RANGE_0100      = 5
    RANGE_0200      = 6
    RANGE_0400      = 7
    RANGE_0500      = 8
    RANGE_0800      = 9
    RANGE_1         = 10
    RANGE_2         = 11
    RANGE_4         = 12
    RANGE_5         = 13
    RANGE_8         = 14
    RANGE_10        = 15
    RANGE_20        = 16
    RANGE_40        = 17
    RANGE_16        = 18

    CLKSRC_INT      = 1
    CLKSRC_FASTEXT  = 2
    CLKSRC_MEDEXT   = 3
    CLKSRC_SLOWEXT  = 4
    CLKSRC_EXT_10M  = 7     # Generate 1 GHz from 10 MHz ext.

    CLKEDG_RISE     = 0
    CLKEDG_FALL     = 1

    SR_1k           = 0x01
    SR_2k           = 0x02
    SR_5k           = 0x04
    SR_10k          = 0x08
    SR_20k          = 0x0A
    SR_50k          = 0x0C
    SR_100k         = 0x0E
    SR_1M           = 0x14
    SR_2M           = 0x18
    SR_5M           = 0x1A
    SR_10M          = 0x1C
    SR_20M          = 0x1E
    SR_50M          = 0x22
    SR_100M         = 0x24
    SR_250M         = 0x2B
    SR_500M         = 0x30
    SR_1G           = 0x35
    SR_1G_FROM_EXT10 = 1000000000

    TRIG_ENGINE_OP_J       = 0
    TRIG_ENGINE_OP_K       = 1
    TRIG_ENGINE_OP_J_OR_K  = 2
    TRIG_ENGINE_OP_J_AND_K = 3
    TRIG_ENGINE_OP_J_XOR_K = 4
    TRIG_ENGINE_OP_J_AND_NOT_K = 5
    TRIG_ENGINE_OP_NOT_J_AND_K = 6

    TRIG_ENGINE_J       = 0
    TRIG_ENGINE_K       = 1

    TRIG_CHAN_A     = 0
    TRIG_CHAN_B     = 1
    TRIG_EXTERNAL   = 2
    TRIG_DISABLE    = 3
    TRIG_CHAN_C     = 4
    TRIG_CHAN_D     = 5

    TRIG_SLP_POS        = 1
    TRIG_SLP_NEG        = 2

    ETR_5V              = 0
    ETR_1V              = 1

    # Parameters that can be extracted
    GET_DATA_WIDTH = 0x10000009
    GET_MEMORY_SIZE = 0x1000002A
    SETGET_ASYNC_BUFFSIZE_BYTES = 0x10000039
    SETGET_ASYNC_BUFFCOUNT = 0x10000040
    GET_DATA_FORMAT = 0x10000042
    GET_SAMPLES_PER_TIMESTAMP_CLOCK = 0x10000044
    GET_RECORDS_CAPTURED = 0x10000045
    GET_ASYNC_BUFFERS_PENDING = 0x10000050
    GET_ASYNC_BUFFERS_PENDING_FULL = 0x10000051
    GET_ASYNC_BUFFERS_PENDING_EMPTY = 0x10000052
    GET_ECC_MODE = 0x10000048
    GET_AUX_INPUT_LEVEL = 0x10000049

    GET_CLOCK_SOURCE = 0x1000000C
    GET_CLOCK_SLOPE = 0x1000000D

def CHK(ret):
    if ret != 512:
        raise ValueError("Alazar error %s: %s" % (ret, get_error(ret)))

def get_error(errcode):
    ats.AlazarErrorToText.restype = ctypes.c_char_p
    ret = ats.AlazarErrorToText(errcode)
    return str(ret)

class Alazar:

    def __init__(self, systemid=1, boardid=1):
        self._systemid = systemid
        self._boardid = boardid
        self.init_alazar(systemid, boardid)
        self.capture_channels = Constants.CHANNEL_A
        self._posted_buffers = []

    def init_alazar(self, systemid, boardid):
        self.handle = ats.AlazarGetBoardBySystemID(systemid, boardid)

# To allow creation of subprocess the file handle should not be inherited.
#        win32api.SetHandleInformation(
#            msvcrt.get_osfhandle(self.handle),
#            win32api.HANDLE_FLAG_INHERIT, 0)

    def get_status(self):
        return ats.AlazarGetStatus(self.handle)

    def set_capture_channels(self, chan):
        self.capture_channels = chan

    def set_ch_props(self, chan, vrange, coupling=Constants.COUPLING_DC,
                     impedance=Constants.IMPEDANCE_50):
        ats.AlazarInputControl(self.handle, chan, coupling, vrange, impedance)

    def set_ch_bw_limit(self, chan, val):
        ats.AlazarSetBWLimit(self.handle, chan, val)

    def set_clock(self, rate):
        err = ats.AlazarSetCaptureClock(self.handle,
            Constants.CLKSRC_INT, rate, Constants.CLKEDG_RISE, 0)
        CHK(err)

    def set_1GHz(self):
        err = ats.AlazarSetCaptureClock(self.handle,
                            Constants.EXTERNAL_CLOCK_10MHz_REF,
                            1000000000, Constants.CLOCK_EDGE_RISING, 1)

    def set_trigger_ext(self, lev, coupling=Constants.COUPLING_DC,
                        trigscale=Constants.ETR_5V):
        err = ats.AlazarSetTriggerOperation(
                self.handle, Constants.TRIG_ENGINE_OP_J,
                Constants.TRIG_ENGINE_J, Constants.TRIG_EXTERNAL, Constants.TRIG_SLP_POS, lev,
                Constants.TRIG_ENGINE_K, Constants.TRIG_DISABLE, Constants.TRIG_SLP_POS, lev)
        CHK(err)
        err = ats.AlazarSetExternalTrigger(self.handle, coupling, trigscale)
        err = ats.AlazarSetTriggerDelay(self.handle, 0)
        CHK(err)

    def set_led(self, on=True):
        ats.AlazarSetLED(self, self.handle, int(on))

    def prepare_capture(self, samples_per_rec, rec_per_buf, rec_per_acq, trig_delay=0, pre_trig_samples=0):
        '''
        pre_trig cannot be more than number of samples per record - 64
        '''
        ret = ats.AlazarSetTriggerDelay(self.handle, trig_delay)
        if ret == 572:
            self.init_alazar(self._systemid, self._boardid)
            ret = ats.AlazarSetTriggerDelay(self.handle, trig_delay)
        CHK(ret)

        dma_flag = 0x001 | 0x200
        ret = ats.AlazarBeforeAsyncRead(self.handle,
            self.capture_channels, ctypes.c_long(pre_trig_samples),
                samples_per_rec, rec_per_buf, rec_per_acq, dma_flag)
        CHK(ret)
        ret = ats.AlazarSetRecordSize(self.handle, pre_trig_samples, samples_per_rec - pre_trig_samples)
        CHK(ret)

    def post_buffers(self, bufs):
        if bufs is None:
            return
        if type(bufs) not in (types.ListType, types.TupleType):
            bufs = (bufs, )
        else:
            if len(bufs) == 0:
                return
        for buf in bufs:
            bufsize = len(buf)
            ret = ats.AlazarPostAsyncBuffer(self.handle, buf.ctypes.data, bufsize)
            if ret == 512:
                self._posted_buffers.append(buf)
            elif ret == 518:
                print 'Error, unable to post buffer!'
            else:
                CHK(ret)
        return True

    def get_parameter(self, chan, p):
        ret = ctypes.c_long(0)
        ats.AlazarGetParameter(self.handle, ctypes.c_uint8(chan), p, ctypes.pointer(ret))
        return ret

    def get_channel_info(self):
        bps = np.array([0], dtype=np.uint8)
        max_s = np.array([0], dtype=np.uint32)
        success = ats.AlazarGetChannelInfo(
                self.handle,
                max_s.ctypes.data,
                bps.ctypes.data)
        bits_per_sample = bps[0]
        max_samples_per_record = max_s[0]
        bytes_per_sample = (bps[0]+7)/8

    def start_capture(self):
        ret = ats.AlazarStartCapture(self.handle)
        CHK(ret)

    def end_capture(self):
        self._posted_buffers = []
        ret = ats.AlazarAbortAsyncRead(self.handle)
        if ret == 572:
            self.init_alazar(self._systemid, self._boardid)
            return
        CHK(ret)

    def get_next_buffer(self, timeout=200):
        '''
        Wait for a buffer from the Alazar to become ready and return it.
        Timeout is in ms.
        '''
        if len(self._posted_buffers) == 0:
            return None
        buf = self._posted_buffers[0]
        ret = ats.AlazarWaitAsyncBufferComplete(self.handle, buf.ctypes.data, int(timeout))
        if ret == 512:
            del self._posted_buffers[0]
            return buf
        elif ret == 579:
            logging.debug('WaitAsyncBufferComplete timed out!')
            return None
        else:
            CHK(ret)
            return None

    def wait_for_buffer(self, buf, timeout=200):
        ret = ats.AlazarWaitAsyncBufferComplete(self.handle, buf.ctypes.data, int(timeout))
        if ret == 579:
            logging.warning('Warning: timed out!')
            return None
        else:
            CHK(ret)
        return buf

    def post_alaz_buffer(self, buf):
        ret = ats.AlazarPostAsyncBuffer(self.handle, buf.ctypes.data, len(buf))
        CHK(ret)

    def test_alazar(self):
        import matplotlib.pyplot as plt

        self.set_capture_channels(Constants.CHANNEL_A)
        self.set_ch_range(Constants.CHANNEL_A, Constants.RANGE_0050)
        self.set_ch_bw_limit(Constants.CHANNEL_A, 0)
        self.set_clock(Constants.SR_1G)
        self.set_trigger_ext(128)

        RECPERBUF = 2
        NBUF = 4
        SAMPLES = 1024
        BUFPERACQ = 4
        ars = [np.zeros([SAMPLES*RECPERBUF,], dtype=np.uint8) for i in range(NBUF)]
        self.prepare_capture(0, SAMPLES, RECPERBUF, RECPERBUF*NBUF*BUFPERACQ)
        self.post_buffers(ars)
        self.start_capture()
        i = 0
        j = 0
        while i < NBUF*BUFPERACQ and j < 20:
            j += 1
            buf = self.get_next_buffer()
            if buf is not None:
                print 'Buf: %s' % (buf, )
                plt.figure()
                plt.plot(buf)
                plt.savefig('samples.png')
                plt.close()
                i += 1
            self.post_buffers(buf)
        self.end_capture()

        return ars

if __name__ == "__main__":
    a = Alazar()
    a.test_alazar()
