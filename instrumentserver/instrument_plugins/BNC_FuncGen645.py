import time
import types
import visa
from visainstrument import VisaInstrument, Instrument

class BNC_FuncGen645(VisaInstrument):

    def __init__(self, name, address, **kwargs):
        super(BNC_FuncGen645, self).__init__(name, address=address, term_chars='\n', **kwargs)

        self.add_visa_parameter('output_on',
            'OUTP?', 'OUTP %s',
            type=types.BooleanType,
            flags=Instrument.FLAG_GETSET)
        self.add_visa_parameter('function',
            'FUNC?', 'FUNC %s',
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

        self.add_visa_parameter('frequency',
            'FREQ?', 'FREQ %.06f',
            type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='Hz')

        self.add_parameter('amplitude', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='V')
        self.add_parameter('offset', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='V')

        self.add_parameter('Vlow', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='V')
        self.add_parameter('Vhigh', type=types.FloatType,
            flags=Instrument.FLAG_GETSET, units='V')

        self.add_visa_parameter('sync_on',
            'OUTP:SYNC?', 'OUTP:SYNC %d',
            type=types.BooleanType,
            flags=Instrument.FLAG_GETSET,
            help='Whether sync output is enabled')

        self.add_visa_parameter('trigger_source',
            'TRIG:SOUR?', 'TRIG:SOUR %s',
            type=str, flags=Instrument.FLAG_GETSET,
            format_map={'IMM':'Immediate', 'EXT': 'External', 'BUS': 'Bus'})
        self.add_visa_parameter('burst_on',
            'BURST:STAT?', 'BURST:STAT %s',
            type=bool, flags=Instrument.FLAG_GETSET)
        self.add_visa_parameter('burst_on',
            'BURST:MODE?', 'BURST:MODE %s',
            type=str, flags=Instrument.FLAG_GETSET,
            format_map={'TRIG': 'Triggered', 'GAT': 'Gated'})
        self.add_visa_parameter('burst_cycles',
            'BURST:NCYC?', 'BURST:NCYC %d',
            type=float, flags=Instrument.FLAG_GETSET)
        self.add_function('trigger')

        self.get_all()
        self.set(kwargs)

    def do_get_Vhigh(self):
        val = self.ask('VOLT:HIGH?\n')
        return float(val)

    def do_set_Vhigh(self, val):
        self.write('VOLT:HIGH %.06f\n' % val)
        self.get_amplitude()
        self.get_offset()

    def do_get_Vlow(self):
        val = self.ask('VOLT:LOW?\n')
        return float(val)

    def do_set_Vlow(self, val):
        self.write('VOLT:LOW %.06f\n' % val)
        self.get_amplitude()
        self.get_offset()

    def do_set_offset(self, val):
        self.write('VOLT:OFFS %.06f\n' % val)
        self.get_Vlow()
        self.get_Vhigh()

    def do_get_offset(self):
        val = self.ask('VOLT:OFFS?\n')
        return float(val)

    def do_set_amplitude(self, val):
        self.write('VOLT %.06f\n' % val)
        self.get_Vlow()
        self.get_Vhigh()

    def do_get_amplitude(self):
        val = self.ask('VOLT?\n')
        return float(val)

    def trigger(self):
        self.write('*TRG')

if __name__ == '__main__':
    Instrument.test(BNC_FuncGen645)

