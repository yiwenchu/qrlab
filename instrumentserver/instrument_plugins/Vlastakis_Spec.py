from instrument import Instrument
from visainstrument import VisaInstrument
import types
import time
import logging
logging.getLogger().setLevel(logging.INFO)
import numpy as np

class Vlastakis_Spec(VisaInstrument):

    def __init__(self, name, address, **kwargs):
        VisaInstrument.__init__(self, name, address=address, term_chars='')
        self.baud_rate = 9600
        self.data_bits = 8
        self.stop_bits = 1
        self.parity = 0

        self.rf_ins = None

        self.add_parameter('rfsource', type=types.StringType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET)
        self.add_parameter('df0', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True,
                value=10.6e6)
        self.add_parameter('frequency', type=types.FloatType,
                flags=Instrument.FLAG_GETSET)
        self.add_parameter('rf_on', type=types.BooleanType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_GET)
        self.add_parameter('Navg', type=types.IntType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True,
                value=5)
        self.add_parameter('power', type=types.FloatType,
                flags=Instrument.FLAG_GET)
        self.add_parameter('delay', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True, value=0.05)

        self.set(kwargs)

    def do_set_rfsource(self, val):
        srv = self.get_instruments()
        if srv:
            self.rf_ins = srv.get(val)
        else:
            self.rf_ins = None

    def _get_ins(self):
        if self.rf_ins is None:
            self.do_set_rfsource(self.get_rfsource())
        return self.rf_ins

    def do_get_frequency(self):
        ins = self._get_ins()
        if ins:
            return ins.get_frequency() - self.get_df0()

    def do_set_frequency(self, val):
        ins = self._get_ins()
        if ins:
            df0 = self.get_df0()
            logging.debug('Setting RF source to %.03f MHz', (val+df0)/1e6)
            ins.set_frequency(val + df0)
            i = 0
            while i < 10:
                val = ins.get_frequency()
                if val is not None:
                    break
            logging.debug('  Source @ %.03f MHz + df0', (val/1e6))
        else:
            logging.warning('RF Source not available')

    def do_get_rf_on(self):
        ins = self._get_ins()
        if ins:
            return ins.get_rf_on()

    def do_set_rf_on(self, val):
        ins = self._get_ins()
        if ins:
            return ins.set_rf_on(val)

    def find_offset(self, f0, freqrange=1e6, N=21):
        '''
        Find (and set) spectrum analyzer frequency offset.
        - f0: center frequency of test signal
        - df0: offset of spectrum analyzer, if None use currently set value
        - freqrange: +/- range of frequencies.
        - N: number of steps

        Sets and returns optimum offset from f0
        '''
        logging.info('Finding spectrum analyzer offset...')

        df0 = self.get_df0()

        fs, ps = self.get_spectrum(f0 - freqrange, f0 + freqrange, N)
        imax = np.argmax(ps)
        fmax = fs[imax]

        df0 += fmax - f0
        self.set_df0(df0)
        return df0

    def do_get_power(self):
        self.clear()
        Navg = self.get_Navg()
        plevel = 0
        i = 0
        n = 0
        while n < Navg and i < 2 * Navg:
            try:
                ret = int(self.ask('1'))
                plevel += ret
                n += 1
            except:
                time.sleep(0.005)
            i += 1
        if n != 0:
            plevel = float(plevel) / n
        logging.debug('Power %s, %d reads (%d tries)', plevel, i, n)
        return plevel

    def get_power_at(self, f):
        self.set_frequency(f)
        if not self.get_rf_on():
            self.set_rf_on(True)
        time.sleep(self.get_delay())
        return self.get_power()

    def get_spectrum(self, fmin, fmax, N):
        self.set_rf_on(True)

        fs = np.linspace(fmin, fmax, N)
        ps = []
        for f in fs:
            plevel = self.get_power_at(f)
            ps.append(plevel)

        self.set_rf_on(False)

        return fs, np.array(ps)

    def find_peak(self, f, freqrange=1e6, N=21):
        fs, ps = self.get_spectrum(f - freqrange, f + freqrange, N)
        imax = np.argmax(ps)
        return fs[imax]

if __name__ == '__main__':
    ins = Vlastakis_Spec('vspec', 'COM6')
    print ins.get_all()
    ins.close()
