from visainstrument import VisaInstrument, Instrument
import time
import visa
import types

class Agilent_FuncGen33250A(VisaInstrument):

    def __init__(self, name, address, **kwargs):
        super(Agilent_FuncGen33250A, self).__init__(name, address=address, term_chars='\n', **kwargs)

        self.add_visa_parameter('output_on',
            'OUTP?', 'OUTP %d',
            type=types.BooleanType,
            flags=Instrument.FLAG_GETSET)
        self.add_parameter('function',
            type=types.StringType,
            flags=Instrument.FLAG_GETSET,
            format_map={
                'SIN': 'SIN',
                'SQU': 'SQUARE',
                'RAMP': 'RAMP',
                'PULS': 'PULSE',
                'NOIS': 'NOISE',
                'DC': 'DC',
                'USER': 'USER',
            })
        self.add_parameter('burst_on',
            type=types.StringType,
            flags=Instrument.FLAG_GETSET,
            format_map={
                'OFF': 'False',
                'ON': 'True',
            })
        self.add_visa_parameter('sync_on',
            'OUTP:SYNC?', 'OUTP:SYNC %d',
            type=types.BooleanType,
            flags=Instrument.FLAG_GETSET)
        self.add_parameter('frequency', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='Hz', minval=0, maxval=50e6)
        self.add_parameter('period_us', type=types.FloatType,
            flags=Instrument.FLAG_GETSET)
        self.add_visa_parameter('Vhigh',
            'VOLT:HIGH?', 'VOLT:HIGH %.06f',
            type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='V')
        self.add_visa_parameter('Vlow',
            'VOLT:LOW?', 'VOLT:LOW %.06f',
            type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='V')
        self.add_visa_parameter('DCOffset',
            'VOLT:OFFS?', 'VOLT:OFFS %.06f',
            type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='V')
        self.add_visa_parameter('edgetime',
            'PULS:TRAN?', 'PULS:TRAN %.06e',
            type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='sec')
        self.add_visa_parameter('pulsewidth',
            'PULS:WIDTH?', 'PULS:WIDTH %.06e',
            type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='sec')

        self.get_all()
        self.set(kwargs)

    def do_get_frequency(self):
        val = self.ask('FREQ?')
        val = float(val)
        self.set_period_us(1e6 / val, update=False)
        return val

    def do_set_frequency(self, freq, update=False):
        self.write('FREQ %.06f' % freq)
        self.get_period_us()
        time.sleep(0.05)

    def do_set_period_us(self, period, update=True):
        freq =  1.0 / (period * 1e-6)
        if update:
            self.set_frequency(freq)

    def do_get_period_us(self):
        return 1e6 / self.get_frequency()

    def do_set_function(self, val):
        self.write('BURS:STAT OFF')
        freq = self.get_frequency()
        vhigh = self.get_Vhigh()
        vlow = self.get_Vlow()
        amp = vhigh - vlow
        ofs = (vhigh + vlow)/2.0
        self.write('APPL:%s %s, %s, %s' % (val, freq, amp, ofs))

    def do_get_function(self):
        val = self.ask('APPL?')
        return val.strip('"').split(' ')[0]
        
    def do_set_burst_on(self, val):
        if val != 'ON':
            self.write('BURS:STAT OFF')
        else:
            freq = self.get_frequency()
            vhigh = self.get_Vhigh()
            vlow = self.get_Vlow()
            amp = vhigh - vlow
            ofs = (vhigh + vlow)/2.0
            func = self.get_function()
            self.write('APPL:%s %s, %s, %s\n' % (func, freq, amp, ofs))
            self.write('BURS:MODE TRIG')
            self.write('BURS:NCYC 1')
            self.write('TRIG:SOUR EXT')
            self.write('BURS:STAT ON')
        

    def do_get_burst_on(self):
        val = self.ask('BURS:STAT?')
        return val

    def query_config(self):
        return self.ask('APPL?')
        time.sleep(0.05)
