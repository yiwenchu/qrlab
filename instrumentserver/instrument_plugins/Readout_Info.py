from instrument import Instrument
import types

class Readout_Info(Instrument):

    def __init__(self, name, **kwargs):
        Instrument.__init__(self, name, tags=['virtual'])

        self.add_parameter('rotype', type=types.StringType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                option_list=('High-power', 'Dispersive'),
                help='Read-out type to use')
        self.add_parameter('rfsource1', type=types.StringType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                help='RF-source for read-out pulse')
        self.add_parameter('rfsource2', type=types.StringType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                help='RF-source for demodulation')
        self.add_parameter('power', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_GET)
        self.add_parameter('frequency', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_GET)
        self.add_parameter('readout_chan', type=types.StringType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True)
        self.add_parameter('IQg', type=types.ComplexType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                help='IQ point of g')
        self.add_parameter('IQe', type=types.ComplexType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                help='IQ point of e')
        self.add_parameter('acq_chan', type=types.StringType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True)
        self.add_parameter('pulse_len', type=types.IntType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True, value=500)
        self.add_parameter('acq_len', type=types.IntType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                help='Acquisition length for FPGA',
                set_func=lambda x: True, value=3000)
        self.add_parameter('ref_len', type=types.IntType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                help='Reference length for FPGA',
                set_func=lambda x: True, value=500)
        self.add_parameter('naverages', type=types.IntType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                help='# averages for FPGA',
                set_func=lambda x: True, value=500)
        self.add_parameter('envelope', type=types.StringType,
                help='Envelope for FPGA',
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True, value='1')

        self.set(kwargs)

    def do_set_rfsource(self, val):
        srv = self.get_instruments()
        if srv:
            self._ins = srv.get(val)
        else:
            self._ins = None

    def _get_ins(self):
        if self._ins:
            return self._ins
        srv = self.get_instruments()
        if srv:
            self._ins = srv.get(self.get_rfsource())

    def do_get_power(self):
        ins = self._get_ins()
        if ins:
            return ins.get_power()

    def do_set_power(self, val):
        ins = self._get_ins()
        if ins:
            return ins.set_power(val)

    def do_get_frequency(self):
        ins = self._get_ins()
        if ins:
            return ins.get_frequency()

    def do_set_frequency(self, val):
        ins = self._get_ins()
        if ins:
            return ins.set_frequency(val)
