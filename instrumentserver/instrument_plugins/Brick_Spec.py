# easy_install pywinusb

import pywinusb.hid
import time
from instrument import Instrument

LMS_802 = 0x1221
LMS_103 = 0x1220
LMS_123 = 0x1222
LMS_163 = 0x1224
LMS_203 = 0x1223
BRICKS = (LMS_802, LMS_103, LMS_123, LMS_163, LMS_203)

def find_labbricks():
    l = pywinusb.hid.find_all_hid_devices()
    ret = []
    for i in l:
        if i.vendor_id == 0x41F and i.product_id in BRICKS:
            ret.append(i)
    return ret

class Brick_Spec(Instrument):

    def __init__(self, name, brickdev, specdev):
        super(Brick_Spec, self).__init__(name)

        self._brickdev = brickdev
        self._specdev = specdev
        self._nstatus = 0
        self.last_replies = {}
        self.open()

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
        self.last_replies[data[1]] = val

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
            if cmd in self.last_replies:
                ret = self.last_replies[cmd]
                del self.last_replies[cmd]
        return ret

    def get_rf_on(self):
        return bool(self.do_cmd(0x0A))

    def set_rf_on(self, on):
        self.do_cmd(0x8A, 1, int(on))
        return bool(on)

    def get_frequency(self):
        return self.do_cmd(0x44) * 10

    def set_frequency(self, freq):
        f = int(round(freq / 10))
        self.do_cmd(0xc4, 4, f)
        return f * 10

    def get_power(self):
        '''Return output power in dBm with respect to maximum output power.'''
        return -self.do_cmd(0x0D) * 0.25

    def set_power(self, power):
        '''Set output power in dBm with respect to maximum output power.'''
        if power > 0:
            raise ValueError('Power should be < 0')
        p = int(round(-power / 0.25))
        self.do_cmd(0x8D, 1, p)
        return -p * 0.25

    def get_ext_locked(self):
        return self.do_cmd(0x44) * 10

    def set_ext_locked(self, lock):
        f = int(round(freq / 10))
        self.do_cmd(0xc4, 4, f)
        return f * 10

    def get_min_frequency(self):
        return self.do_cmd(0x20) * 100

    def get_max_frequency(self):
        return self.do_cmd(0x21) * 100
