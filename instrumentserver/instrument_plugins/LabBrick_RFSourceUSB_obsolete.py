# easy_install pywinusb

from instrument import Instrument
import pywinusb.hid
import time
import types
import threading

LMS_802 = 0x1221
LMS_103 = 0x1220
LMS_123 = 0x1222
LMS_163 = 0x1224
LMS_203 = 0x1223
BRICKS = (LMS_802, LMS_103, LMS_123, LMS_163, LMS_203)
MAX_POWER = 13

def find_labbricks(devid=None):
    l = pywinusb.hid.find_all_hid_devices()
    ret = []
    for i in l:
        if i.vendor_id == 0x41F and i.product_id in BRICKS:
            ret.append(i)
    return ret

def find_serial_brick(serial):
    bricks = find_labbricks()
    for brick in bricks:
        if str(serial) in brick.serial_number:
            return brick
    return None

class LabBrick_RFSource(Instrument):

    def __init__(self, name, serial=None, devid=None):
        super(LabBrick_RFSource, self).__init__(name)

        if serial:
            self._dev = find_serial_brick(serial)
            if self._dev is None:
                raise Exception('Brick %s not found' % serial)
        else:
            devs = find_labbricks()
            self._dev = devs[int(devid)]

        self._nstatus = 0
        self._last_replies = {}
        self.open()

# TODO: fix problem reading these
        minfreq = self.do_get_min_frequency()
        maxfreq = self.do_get_max_frequency()
        print 'Min freq: %.03e, max freq: %.03e' % (minfreq, maxfreq)

        self.add_parameter('serial', type=types.StringType,
            flags=Instrument.FLAG_GET)
        self.add_parameter('rf_on', type=types.BooleanType,
            flags=Instrument.FLAG_GETSET)
        self.add_parameter('ext_locked', type=types.FloatType,
            flags=Instrument.FLAG_GETSET)
        self.add_parameter('power', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='dBm', minval=-135, maxval=16)
        self.add_parameter('frequency', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='Hz')

        self.get_all()

    def open(self):
        self._dev.open()
        self._dev.set_raw_data_handler(self.data_handler)

    def close(self):
        self._dev.set_raw_data_handler(None)
        self._dev.close()

    def data_handler(self, data):
        # Ignore status reports for now
        if (data[1] & 0xf) == 0xe:
            self._nstatus += 1
#            if (self._nstatus % 100) == 0:
#                print 'Ignored 100 reports'
            return
        nbytes = data[2]
        val = 0
        for i in range(nbytes):
            val <<= 8
            val += data[3+nbytes-1-i]
#        print 'Got reply for %s, val %s, nbytes: %s: data %r' % (data[1], val, nbytes, data)
        self._last_replies[data[1]] = val

    def do_cmd(self, cmd, count=0, data=0, get_reply=True):
        buffer = [0] * 9
        buffer[1] = cmd
        buffer[2] = count
        buffer[3] = data & 0xff
        buffer[4] = data>>8 & 0xff
        buffer[5] = data>>16 & 0xff
        buffer[6] = data>>24 & 0xff
        self._dev.send_output_report(buffer)

        ret = None
        if get_reply:
            time.sleep(0.05)
            if cmd in self._last_replies:
                ret = self._last_replies[cmd]
                del self._last_replies[cmd]
        return ret

    def do_get_serial(self):
        return self._dev.serial_number

    def do_get_rf_on(self):
        return bool(self.do_cmd(0x0A))

    def do_set_rf_on(self, on):
        self.do_cmd(0x8A, 1, int(on))
        return bool(on)

    def do_get_min_frequency(self):
        val = self.do_cmd(0x20)
        if val is not None:
            return val * 1e5

    def do_get_max_frequency(self):
        val = self.do_cmd(0x21)
        if val is not None:
            return val * 1e5

    def do_get_frequency(self):
        val = self.do_cmd(0x44)
        if val is not None:
            return val * 10

    def do_set_frequency(self, freq):
        f = int(round(freq / 10))
        self.do_cmd(0xc4, 4, f)
        return f * 10

    def do_get_power(self):
        '''Return output power in dBm.'''
        val = self.do_cmd(0x0D)
        if val is not None:
            return MAX_POWER - val * 0.25

    def do_set_power(self, power):
        '''Set output power in dBm.'''
        power -= MAX_POWER
        if power > 0:
            raise ValueError('Power should be < %s' % MAX_POWER)
        p = int(round(-power / 0.25))
        self.do_cmd(0x8D, 1, p)
        return MAX_POWER - p * 0.25

    def do_get_ext_locked(self):
        # Need to figure out the right command...
        val = self.do_cmd(0x44)
        if val is not None:
            return val * 10

    def do_set_ext_locked(self, lock):
        # Need to figure out the right command...
        f = int(round(freq / 10))
        self.do_cmd(0xc4, 4, f)
        return f * 10

if __name__ == '__main__':
    brick = Instrument.test(LabBrick_RFSource)
    brick.close()
