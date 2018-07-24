from visainstrument import SCPI_Instrument

class Agilent_E3631A(SCPI_Instrument):

    def __init__(self, name, address, **kwargs):
        super(Agilent_E3631A, self).__init__(name=name, address=address)

        self.add_scpi_parameter('output_on', 'OUTP', '%d', type=bool)

        self.get_all()
        self.set(kwargs)

    def set_output_state(self, state):
    	self.write('OUTP ' + state)

    def set_output_off(self):
    	self.write('OUTP OFF')
