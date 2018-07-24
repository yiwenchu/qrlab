import numpy as np
from visainstrument import SCPI_Instrument

class Agilent_E5052B(SCPI_Instrument):
    def __init__(self, name, address):
        super(Agilent_E5052B, self).__init__(name, address=address)

        self.add_scpi_parameter(
            'trigger', 'TRIG:PN:SOUR', type=str,
            option_list=('INT', 'EXT', 'MAN', 'BUS')
        )
        self.add_scpi_parameter('averaging', 'SENS:PN:AVER:STAT', type=bool)
        self.add_scpi_parameter('n_averages', 'SENS:PN:AVER:COUN', type=int)
        self.add_scpi_parameter('n_correlations', 'SENS:PN:CORR:COUN', type=int)
        self.add_scpi_parameter(
            'start_freq', 'SENS:PN:FREQ:STAR', type=float,
            option_list=[1,10,100,1000]
        )
        self.add_scpi_parameter(
            'stop_freq', 'SENS:PN:FREQ:STOP', type=float,
            option_list=[1e5, 1e6, 5e6, 10e6, 20e6, 40e6, 100e6]
        )
        self.add_scpi_parameter('n_points', 'SENS:PN:SWE:POIN', type=int)

    def get_pn_data(self):
        x_strdata = self.ask('CALC:PN:DATA:XDAT?', timeout=0.1)
        x_data = np.array(map(float, x_strdata.split(',')))
        self.check_last_command()
        y_strdata = self.ask('CALC:PN:DATA:RDAT?', timeout=0.1)
        y_data = np.array(map(float, y_strdata.split(',')))
        self.check_last_command()
        return x_data, y_data

if __name__ == '__main__':
    ins = Agilent_E5052B('PNA', 'GPIB::28')
    # ins.test_commands()
    print ins.get_all()

    import matplotlib.pyplot as plt
    plt.plot(*ins.get_pn_data())
    plt.xscale('log')
    plt.xlabel('Offset Freq')
    plt.ylabel('Phase Noise (dBc/Hz)')
    plt.show()


