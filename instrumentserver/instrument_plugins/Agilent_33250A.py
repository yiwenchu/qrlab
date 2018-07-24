from visainstrument import SCPI_Instrument

class Agilent_33250A(SCPI_Instrument):
    def __init__(self, name, address, **kwargs):
        super(Agilent_33250A, self).__init__(name=name, address=address)

        self.add_parameter(
            'function', type=str,
            format_map={
                'SIN': 'SIN',
                'SQU': 'SQUARE',
                'RAMP': 'RAMP',
                'PULS': 'PULSE',
                'NOIS': 'NOISE',
                'DC': 'DC',
                'USER': 'USER',
                }
        )

        self.add_scpi_parameter(
            'frequency', 'FREQ', '%d', units='Hz', type=float, updates=['period']
        )

        self.add_parameter('period', type=float, units='us')

        self.add_scpi_parameter(
            'voltage', 'VOLT', '%.06f', units='V', type=float, updates=['v_high', 'v_low']
        )
        self.add_scpi_parameter(
            'v_offset', 'VOLT:OFFS', '%.06f', units='V', type=float, updates=['v_high', 'v_low']
        )
        self.add_scpi_parameter(
            'v_high', 'VOLT:HIGH', '%.06f', units='V', type=float, updates=['voltage', 'v_offset']
        )
        self.add_scpi_parameter(
            'v_low', 'VOLT:LOW', '%.06f', units='V', type=float, updates=['voltage', 'v_offset']
        )

        self.add_scpi_parameter('edgetime', 'PULS:TRAN', '%.06e', units='s', type=float)
        self.add_scpi_parameter('pulsewidth', 'PULS:WIDTH', '%.06e', units='s', type=float)
        self.add_scpi_parameter('output_on', 'OUTP', '%d', type=bool)
        self.add_scpi_parameter('sync_on', 'OUTP:SYNC', '%d', type=bool)

        self.get_all()
        self.set(kwargs)

    def do_get_function(self):
        val = self.ask('APPL?')
        return val.strip('"').split(' ')[0]

    def do_set_function(self, val):
        s = 'APPL:' + val
        self.write(s)

    def do_set_period(self, period):
        self.set_frequency(1e6 / period)

    def do_get_period(self):
        return 1e6 / self.get_frequency()
